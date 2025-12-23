# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
import os
import numpy as np  # <--- 必须加上这一行！
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    # latent_size = args.image_size // 8
    latent_size = args.image_size

    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    # ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    if args.ckpt is None:
        raise ValueError("Please specify a checkpoint path using --ckpt")
    
    # state_dict = find_model(ckpt_path)
    state_dict = find_model(args.ckpt)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Labels to condition the model with (feel free to change):
    # class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    target_real = args.target_real
    target_imag = args.target_imag
    print(f"Generating structures for target transmission: {target_real} + {target_imag}i")


    # 构造 batch 条件向量
    n = 4 # 生成样本数量
    # 创建一个 [1, 2] 的向量，然后复制 n 份变成 [n, 2]
    target_tensor = torch.tensor([target_real, target_imag], device=device).float()
    y = target_tensor.unsqueeze(0).repeat(n, 1)
    z = torch.randn(n, 1, latent_size, latent_size, device=device)

    # Create sampling noise:
    # n = len(class_labels)
    # z = torch.randn(n, 4, latent_size, latent_size, device=device)
    # y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    # z = torch.cat([z, z], 0)
    # y_null = torch.tensor([1000] * n, device=device)
    # y = torch.cat([y, y_null], 0)
    # model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
    model_kwargs = dict(y=y)

    # Sample images:
    # samples = diffusion.p_sample_loop(
    #     model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    # )
    # samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    # samples = vae.decode(samples / 0.18215).sample
    samples = diffusion.p_sample_loop(
        model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )

    # 7. 准备保存路径
    ckpt_abs_path = os.path.abspath(args.ckpt)
    ckpt_dir = os.path.dirname(ckpt_abs_path) 
    exp_dir = os.path.dirname(ckpt_dir) 
    
    # 在实验目录下新建 samples 文件夹
    save_dir = os.path.join(exp_dir, "samples")
    os.makedirs(save_dir, exist_ok=True)

    # 8. [修改点] 自动计算文件前缀 (Target在前，分组独立编号)
    # 逻辑：只检查当前 Target 下已经用到第几号了
    
    # 定义这个 Target 的专属前缀
    base_prefix = f"target_{target_real}_{target_imag}"
    
    file_idx = 1
    while True:
        # 构造文件名: target_0.85_-0.15_001.npy
        candidate_name = f"{base_prefix}_{file_idx:03d}"
        
        save_path_npy = os.path.join(save_dir, f"{candidate_name}.npy")
        
        # 如果文件不存在，说明这个编号可以用
        if not os.path.exists(save_path_npy):
            base_name = candidate_name
            save_path_img = os.path.join(save_dir, f"{base_name}.png")
            break
        
        # 如果存在，就试下一个编号 (001 -> 002 -> ...)
        file_idx += 1

    # 9. 保存结果
    samples_binary = (samples > 0).float()
    np.save(save_path_npy, samples_binary.cpu().numpy())
    
    print(f"\n[保存成功] 结果已保存至:")
    print(f"  -> {save_path_npy}")

    # 保存可视化图片
    try:
        from torchvision.utils import save_image
        save_image(samples, save_path_img, nrow=int(n**0.5), normalize=True, value_range=(-1, 1))
        print(f"  -> {save_path_img}")
    except ImportError:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-Tiny1")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)

    # ==========================================
    #在此处修改你的“硬编码”设置
    # ==========================================
    
    # 1.在此处填入你的 .pt 文件绝对路径或相对路径
    # 注意：Windows路径里的反斜杠 \ 最好改成正斜杠 /，或者用双反斜杠 \\
    MY_CKPT_PATH = "results/08-data-01_DiT-Tiny1_20251211-050607/checkpoints/0500000.pt" 
    
    # 2. 在此处填入你想要的 目标实部
    MY_TARGET_REAL = 0.85
    
    # 3. 在此处填入你想要的 目标虚部
    MY_TARGET_IMAG = -0.15

    # ==========================================


    parser.add_argument("--ckpt", type=str, default=MY_CKPT_PATH)
    parser.add_argument("--target-real", type=float, default=MY_TARGET_REAL)
    parser.add_argument("--target-imag", type=float, default=MY_TARGET_IMAG)
    
    args = parser.parse_args()
    main(args)
