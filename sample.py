# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
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
    target_real = 0.5
    target_imag = 0.5
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

    # Save and display images:
    # save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))
    # 后处理：二值化 (Thresholding)
    # 模型输出在 [-1, 1] 之间，我们将其映射回 0/1 结构
    samples_binary = (samples > 0).float()

    # [修改点8] 保存为 .npy 文件 (便于后续物理仿真验证)
    np.save("sampled_structures.npy", samples_binary.cpu().numpy())
    print("Saved samples to sampled_structures.npy")

    # 依然可以保存图片用于直观查看
    save_image(samples, "sample.png", nrow=2, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)
