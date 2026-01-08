# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from datetime import datetime
# [新增] 引入 OmegaConf
from omegaconf import OmegaConf

# 从 models.py 文件中，把名为 DiT_models 的这个对象（变量、函数或类）引入到当前 train.py 的命名空间中
# DiT_models 是一个字典，把字符串名字（我们在命令行输入的）映射到对应的模型构造函数（代码里定义的函数）
from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from torch.optim.lr_scheduler import CosineAnnealingLR


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

class AcousticDataset(torch.utils.data.Dataset):
    def __init__(self, structure_path, target_path):
        self.structures = np.load(structure_path) 
        self.targets = np.load(target_path)       
        
        self.structures = torch.from_numpy(self.structures).float()
        
        # ------------------- 新增代码开始 -------------------
        # 检查维度：如果是 [N, 8, 8]，自动补齐为 [N, 1, 8, 8]
        if self.structures.ndim == 3:
            self.structures = self.structures.unsqueeze(1)
        # ------------------- 新增代码结束 -------------------

        # 归一化结构到 [-1, 1]
        self.structures = self.structures * 2 - 1
        self.targets = torch.from_numpy(self.targets).float()

    def __len__(self):
        return len(self.structures)

    def __getitem__(self, idx):
        return self.structures[idx], self.targets[idx]


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    # dist.init_process_group("nccl")
    # dist.init_process_group("gloo")  # <--- 改为 "gloo"
    # ------------------- 修改开始 -------------------
    # 使用本地文件进行初始化，彻底避开 Windows 环境变量和网络通信的麻烦
    dist.init_process_group(
        backend="gloo", 
        init_method="file:///C:/Users/Administrator/Documents/DiT/dist_lock_file", 
        rank=0, 
        world_size=1
    )
    # ------------------- 修改结束 -------------------
    # [修改] args.global_batch_size -> args.training.global_batch_size
    assert args.training.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    # [修改] args.training.global_seed
    seed = args.training.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    if rank == 0: # 如果是主进程
        # 检查文件夹 args.results_dir 是否存在，不存在则创建
        # [修改] args.experiment.results_dir
        os.makedirs(args.experiment.results_dir, exist_ok=True)
        # 计算当前 results 目录下已经有了多少个文件夹
        # experiment_index = len(glob(f"{args.results_dir}/*"))
        # # 字符串处理，文件系统路径里不能包含 / 字符，所以把它们替换为 -
        # model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        # # 拼接本次实验的专属路径，形如 results/000-DiT-XL-2
        # experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        # ------------------- 修改开始 -------------------
        # 1. 获取当前时间戳 (例如: 20251223-143005)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # 2. 处理模型名称 (把 / 换成 -)
        # [修改] args.model.name
        model_string_name = args.model.name.replace("/", "-")
        
        # 3. 构造文件夹名: 数据集-模型-时间戳 (按照你指定的顺序)
        # 例如: results/08-data-01_DiT-Tiny_20251223-143005
        # [修改] args.data.dataset
        experiment_name = f"{args.data.dataset}_{model_string_name}_{timestamp}"
        experiment_dir = f"{args.experiment.results_dir}/{experiment_name}"
        # ------------------- 修改结束 -------------------
        # 在实验文件夹内部再定义一个 checkpoints 子目录，专门用来放后面训练产生的 .pt 权重文件
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        # 主进程的 logger 会既把信息打印到屏幕，又把信息写入到 experiment_dir/log.txt 文件中
        # create_logger 函数定义在本文件的上方，是自定义的一个函数
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        # [新增] 极其重要：备份本次实验的配置文件！
        # 这样你永远知道这个文件夹是用什么参数跑出来的
        yaml_path = os.path.join(experiment_dir, "config.yaml")
        OmegaConf.save(config=args, f=yaml_path)
        logger.info(f"Configuration saved to {yaml_path}")

    else:
        logger = create_logger(None)

    # Create model:
    assert args.data.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    # latent_size = args.image_size // 8
    latent_size = args.image_size # (直接是8)
    # DiT_models 是在 models.py 文件末尾定义的一个 Python 字典 (dict)
    # 它的键 (Key) 是字符串，比如 "DiT-XL/2" 或 "DiT-B/4"
    # 它的值 (Value) 并不是字符串，而是 函数对象（构造函数）
    # 紧跟在字典取值后的括号 (...) 代表调用刚才取出的那个函数
    # [修改] args.model.name, args.model.learnable_null
    model = DiT_models[args.model.name](
        input_size=latent_size,
        learnable_null=args.model.learnable_null
    ).to(device)
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    # 加载一个 vae 的预训练模型，该模型是在图像数据集上训练的，要求输入的图像的通道数为3（RGB图像），返回的通道数为4
    # 对输入图片的宽高没有定死，只要求能被8整除，因为该 vae 中固定的下采样倍率是8
    # vae 是用于将原始图像压缩到潜空间，处理更短的序列，节省显存并提高计算速度
    # 但因为我们这里的任务不是标准的图像数据集的任务，通道数也不是3，维度本身也很小，所以我们不使用 vae
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # [修改] args.training.lr
    opt = torch.optim.AdamW(model.parameters(), lr=args.training.lr, weight_decay=0)

    # [新增] 定义余弦退火调度器
    # 假设 epochs=1400，它会让学习率从 1e-4 平滑下降到 1e-6
    # [修改] args.training.epochs
    scheduler = CosineAnnealingLR(opt, T_max=args.training.epochs, eta_min=1e-6)

    # Setup data:
    transform = transforms.Compose([
        # [修改] args.data.image_size
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.data.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    # dataset = ImageFolder(args.data_path, transform=transform)
    # dataset = AcousticDataset(args.structure_path, args.target_path)
    # ------------------- 修改开始 -------------------
    # 根据传入的 dataset 参数自动拼接路径
    # 例如：data/08-data-01/surrogate_structures.npy
    # [修改] args.data.dataset
    structure_path = os.path.join("data", args.data.dataset, "structures.npy")
    target_path = os.path.join("data", args.data.dataset, "properties.npy")

    # 检查一下文件是否存在，防止拼写错误
    if not os.path.exists(structure_path) or not os.path.exists(target_path):
        raise FileNotFoundError(f"Data files not found in data/{args.data.dataset}/. Please check the folder name.")

    dataset = AcousticDataset(structure_path, target_path)
    logger.info(f"Dataset loaded from: data/{args.data.dataset}")
    # ------------------- 修改结束 -------------------
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.training.global_seed
    )
    loader = DataLoader(
        dataset,
        # [修改] args.training.global_batch_size
        batch_size=int(args.training.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        # [修改] args.data.num_workers
        num_workers=args.data.num_workers,
        pin_memory=True,
        drop_last=True
    )
    # logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    # logger.info(f"Dataset contains {len(dataset):,} samples ({args.structure_path})")
    logger.info(f"Dataset contains {len(dataset):,} samples (from {args.data.dataset})")

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    # [修改] args.training.epochs
    logger.info(f"Training for {args.training.epochs} epochs...")
    for epoch in range(args.training.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            # vae.encode(x) 输入图像 x，形状如 (N, 3, 256, 256)，返回一个分布对象
            # .latent_dist 得到这个分布对象，sample() 从中采样得到潜空间表示
            # 再通过 .mul_(0.18215) 进行缩放，得到最终的潜空间表示
            # with torch.no_grad():
            #     # Map input images to latent space + normalize latents:
            #     x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            # torch.randint(low, high, size, ...) 是 PyTorch 生成随机整数的函数
            # 这里 0 是下限（包含），diffusion.num_timesteps 是上限（不包含）
            # x.shape[0] 是当前 Batch 的大小（Batch Size）。这意味着我们为 Batch 里的每一张图都独立挑选一个不同的 $t$。
            # device=device: 确保生成的 $t$ 直接位于 GPU 上，方便后续计算
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            # 因为本代码仓库基于 OpenAI 的 guided-diffusion 仓库修改而来
            # 所以 training_losses() 函数的最后一个参数是 model_kwargs 字典，旨在支持各种模型
            # 所以这里把 y 打包进字典而不是在函数签名中直接传递
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            # [修改] args.training.log_every
            if train_steps % args.training.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # [修改] args.training.ckpt_every
            if train_steps % args.training.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

        # [新增] 每个 Epoch 结束时更新学习率
        scheduler.step()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # 新的入口逻辑
    parser = argparse.ArgumentParser()
    # 只需要一个核心参数：配置文件路径
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to config file")
    # 允许通过命令行临时覆盖参数 (例如: training.global_seed=42)
    parser.add_argument("overrides", nargs=argparse.REMAINDER, help="Modify config options from command line")
    
    cmd_args = parser.parse_args()

    # 1. 读取基础配置文件
    base_conf = OmegaConf.load(cmd_args.config)
    
    # 2. 读取命令行覆盖的参数 (如: python train.py training.epochs=10)
    cli_conf = OmegaConf.from_cli(cmd_args.overrides)
    
    # 3. 合并配置 (命令行 > 配置文件)
    final_args = OmegaConf.merge(base_conf, cli_conf)
    
    # 4. 执行训练
    main(final_args)