"""
evaluate.py
功能: 
1. 加载 DiT 模型 Checkpoints。
2. 生成结构并计算二值化合规性 (Binary Validity)。
3. 通过 MATLAB Engine 调用外部物理仿真脚本计算 MAE (物理误差)。
"""

import torch
import numpy as np
import os
import argparse
import time
import csv
from glob import glob
import matlab.engine # 引入 MATLAB 引擎

from models import DiT_models
from diffusion import create_diffusion
from download import find_model

# ==========================================
# [配置区] 请修改这里
# ==========================================
# 填入你存放 solve_batch.m 和 src/ analysis/ 等文件夹的根目录路径
# 建议使用 r"..." 格式防止转义问题
MATLAB_SCRIPT_PATH = r"C:\Users\Administrator\Documents\Acoustic-Metamaterial" 


# ==========================================
# 物理仿真接口 (调用 MATLAB)
# ==========================================
def simulate_physics(structures, eng):
    """
    输入: 
        structures: (N, 1, 8, 8) Tensor, 值在 [0, 1] 之间
        eng: 启动好的 matlab.engine 对象
    输出: 
        pred_targets: (N, 2) Tensor, [real, imag]
    """
    device = structures.device
    N = structures.shape[0]
    
    # 1. 数据预处理: Tensor -> Numpy
    # DiT 输出是 (N, 1, 8, 8)，我们需要去掉通道维 -> (N, 8, 8)
    structures_np = structures.squeeze(1).cpu().numpy().astype(float)
    
    # 2. 维度调整: Python (N, H, W) -> MATLAB (H, W, N)
    # MATLAB 习惯列优先，且通常将 Batch 放在最后
    structures_permuted = structures_np.transpose(2, 1, 0)
    
    # 3. 数据类型转换: Numpy -> MATLAB Double
    # matlab.double() 需要接收 Python list，对于大数组这步可能稍慢，但最稳妥
    input_mat = matlab.double(structures_permuted.tolist())
    
    # 4. 调用 MATLAB 包装函数 solve_batch
    try:
        # nargout=1 表示我们需要 1 个返回值
        # solve_batch 会利用 parfor 并行计算
        results_mat = eng.solve_batch(input_mat, nargout=1)
        
        # 5. 结果回传: MATLAB List -> Numpy -> Tensor
        results_np = np.array(results_mat) # 应该是 (N, 2)
        pred_targets = torch.from_numpy(results_np).float().to(device)
        
        print("\n[Debug] MATLAB 返回的前5个结果:\n", results_np[:5])

        return pred_targets

    except Exception as e:
        print(f"\n[MATLAB Error] 调用出错: {e}")
        # 出错时返回全0，避免程序崩溃，方便排查
        return torch.zeros(N, 2, device=device)


def main(args):
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------
    # 1. 启动并配置 MATLAB 引擎
    # ------------------------------------------------
    print(f"正在启动 MATLAB 引擎 (目标路径: {MATLAB_SCRIPT_PATH})...")
    try:
        eng = matlab.engine.start_matlab()
        # 使用 genpath 递归添加所有子文件夹 (src, analysis, utils 等)
        # 这样确保 solve_batch.m 能调用到深层的 calculate_transmission.m
        eng.addpath(eng.genpath(MATLAB_SCRIPT_PATH), nargout=0)
        print("MATLAB 引擎启动成功！")
    except Exception as e:
        print(f"MATLAB 引擎启动失败: {e}")
        return

    # ------------------------------------------------
    # 2. 准备测试集 (Test Set)
    # ------------------------------------------------
    # 生成一个 10x10 的均匀网格作为标准考卷
    grid_size = 10 
    real_vals = np.linspace(-0.9, 0.9, grid_size)
    imag_vals = np.linspace(-0.9, 0.9, grid_size)
    
    grid_real, grid_imag = np.meshgrid(real_vals, imag_vals)
    test_targets = np.stack([grid_real.flatten(), grid_imag.flatten()], axis=1)
    test_targets = torch.from_numpy(test_targets).float().to(device)
    
    print(f"准备测试集: 共有 {len(test_targets)} 个目标点 (Grid {grid_size}x{grid_size})")

    # ------------------------------------------------
    # 3. 扫描 Checkpoints
    # ------------------------------------------------
    ckpt_list = []
    if os.path.isdir(args.ckpt):
        ckpt_list = sorted(glob(os.path.join(args.ckpt, "**/*.pt"), recursive=True))
    else:
        ckpt_list = [args.ckpt]
    
    if len(ckpt_list) == 0:
        print(f"错误: 在 {args.ckpt} 未找到任何 .pt 文件。")
        return

    print(f"待评估模型数量: {len(ckpt_list)}")
    
    # 结果保存路径
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"evaluation_report_{timestamp}.csv")
    
    # 初始化 CSV
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Experiment", "Checkpoint", "Binary Validity (%)", "Physics MAE (Mean)", "Physics MAE (Max)"])

    # ------------------------------------------------
    # 4. 循环评估
    # ------------------------------------------------
    for ckpt_path in ckpt_list:
        print(f"\n------------------------------------------------")
        print(f"正在评估: {ckpt_path}")
        
        # --- 加载模型 ---
        # 注意: 如果你修改了 hidden_size 或 depth，请确保这里的 args 参数与训练时一致
        # 或者在这里写死你的模型参数，例如 DiT-Tiny (hidden=64, depth=6)
        model = DiT_models[args.model](
            input_size=args.image_size,
            num_classes=args.num_classes
        ).to(device)
        
        state_dict = find_model(ckpt_path)
        model.load_state_dict(state_dict)
        model.eval()
        
        diffusion = create_diffusion(str(args.num_sampling_steps))
        
        # --- 批量生成 (Generation) ---
        n_samples = len(test_targets)
        z = torch.randn(n_samples, 1, args.image_size, args.image_size, device=device)
        y = test_targets
        
        # Setup CFG (Classifier-Free Guidance)
        if args.cfg_scale > 1.0:
            z = torch.cat([z, z], 0)
            
            # 构造 Null Token (空条件)
            # [重要] 必须与 models.py 里的实现一致
            # 如果你用的是 Learnable Embedding (zero), 用 zeros_like
            # 如果你用的是 [2, 2] 方案, 请解开下面这行的注释:
            # y_null = torch.tensor([[2.0, 2.0]], device=device).repeat(n_samples, 1)
            
            # 默认假设是 sample.py 里的 zeros 方案:
            y_null = torch.zeros_like(y) 
            
            y_combined = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y_combined, cfg_scale=args.cfg_scale)
            sample_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            sample_fn = model.forward

        # 执行采样
        start_time = time.time()
        samples = diffusion.p_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
        if args.cfg_scale > 1.0:
            samples, _ = samples.chunk(2, dim=0)
        
        print(f"生成耗时: {time.time() - start_time:.2f}s")

        # --- 计算指标 ---
        
        # 1. Binary Validity (二值化合规性)
        # 统计有多少像素落在模糊区间 (0.2 ~ 0.8)
        samples_01 = (samples + 1) / 2 # [-1,1] -> [0,1]
        invalid_mask = (samples_01 > 0.2) & (samples_01 < 0.8)
        validity_score = 100 * (1 - (invalid_mask.float().mean().item()))
        
        # 2. Physics MAE (物理误差) - 调用 MATLAB
        # 先将连续图像硬二值化 (Threshold = 0)
        structures_binary = (samples > 0).float()
        
        print("  -> 正在调用 MATLAB 计算物理属性...")
        pred_properties = simulate_physics(structures_binary, eng)
        
        # 计算绝对误差
        mae_errors = torch.abs(test_targets - pred_properties)
        mean_mae = mae_errors.mean().item()
        max_mae = mae_errors.max().item()
        
        print(f"  -> Validity: {validity_score:.2f}%")
        print(f"  -> Physics MAE: {mean_mae:.6f}")
        
        # --- 写入 CSV ---
        ckpt_name = os.path.basename(ckpt_path)
        # 获取上一级文件夹名作为实验名
        exp_name = os.path.dirname(os.path.dirname(ckpt_path)).split(os.sep)[-1]
        if not exp_name: exp_name = "default"
        
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([exp_name, ckpt_name, f"{validity_score:.2f}", f"{mean_mae:.6f}", f"{max_mae:.6f}"])

    print(f"\n所有评估完成！完整报告已保存至: {csv_path}")
    # 关闭引擎
    eng.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-Tiny1")
    parser.add_argument("--image-size", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=1000) # 这里实际没用到，因为我们替换了 Embedder
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    
    # 评估专用参数
    parser.add_argument("--ckpt", type=str, required=True, 
                        help="Path to a .pt file or a directory containing .pt files")
    parser.add_argument("--out-dir", type=str, default="eval_results",
                        help="Output directory for the CSV report")
    
    args = parser.parse_args()
    main(args)