"""
evaluate.py (Excel 自动格式化版)
功能: 
1. Best-of-N 策略 (N=5) 进行物理评估。
2. 为每个 checkpoint 的每次评估创建一个独立的文件夹 (带时间戳)。
3. 所有详细数据 (trials.csv, error_map, images) 都存入该独立文件夹。
4. 在根目录下维护 evaluation_summary.csv (数据源)。
5. [新增] 自动生成 evaluation_summary.xlsx (带自动列宽，方便查看)。
"""

import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import time
import math
import csv
import pandas as pd
from glob import glob
import matlab.engine 
from torchvision.utils import save_image
import matplotlib.pyplot as plt 

from models import DiT_models
from diffusion import create_diffusion
from download import find_model

# ==========================================
# [配置区]
# ==========================================
MATLAB_SCRIPT_PATH = r"C:\Users\Administrator\Documents\Acoustic-Metamaterial" 
NUM_REPEATS = 5       # 每个点重复生成几次
SUCCESS_THRESHOLD = 0.10  # 成功率阈值
DEFAULT_CFG_SCALE = 4.0
DEFAULT_CKPT_DIRS = [
    # "results/16-data-01_DiT-Tiny1_20251224-155744/checkpoints/",
    # "results/16-data-01_DiT-Tiny1_20251223-221753/checkpoints/",
    # "results/16-data-01_DiT-Tiny1_20251228-214600/checkpoints/", # 这个是没有学过无条件生成的
    "results/16-data-01_DiT-Tiny1_20251229-150508/checkpoints/", # 这个是有学过无条件生成的，且无条件向量为2，2
]


# ==========================================
# 辅助函数: 自动生成格式化的 Excel
# ==========================================
def update_summary_excel(csv_path):
    """
    读取 CSV 总表，生成一个带有自动列宽的 Excel 文件方便查看。
    包含防崩溃机制（防止用户正打开 Excel 文件导致写入失败）。
    """
    xlsx_path = csv_path.replace(".csv", ".xlsx")
    
    try:
        # 1. 读取最新的 CSV 数据
        if not os.path.exists(csv_path):
            return
        df = pd.read_csv(csv_path)
        
        # 2. 使用 xlsxwriter 引擎写入 Excel
        # 这种模式会完全覆盖旧的 xlsx 文件，确保数据和格式都是最新的
        with pd.ExcelWriter(xlsx_path, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Summary')
            
            # 3. 获取 workbook 和 worksheet 对象以调整格式
            workbook  = writer.book
            worksheet = writer.sheets['Summary']
            
            # 设置表头格式 (加粗, 居中)
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': False,
                'valign': 'vcenter',
                'align': 'center'
            })
            # 数据单元格居中
            cell_center_format = workbook.add_format({
                'valign': 'vcenter',
                'align': 'center'
            })
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            worksheet.set_row(0, None, header_format)

            # 4. 自动调整列宽
            for i, col in enumerate(df.columns):
                # 计算该列所有内容的最大长度 (包括表头)
                # astype(str) 确保数字也被转为字符串来计算长度
                # map(len) 计算每个单元格长度
                # max() 取最大值
                column_len = max(
                    df[col].astype(str).map(len).max(),  # 内容最大长度
                    len(col)  # 表头长度
                )
                # 设置列宽 (额外 +2 留点余量) + 居中格式
                worksheet.set_column(i, i, column_len + 2, cell_center_format)
        
        print(f"  -> [Excel] 已更新带格式的报表: {os.path.basename(xlsx_path)}")

    except PermissionError:
        print(f"\n[警告] 无法更新 Excel 报表 ({os.path.basename(xlsx_path)})。")
        print("       原因: 你可能正在打开该文件。请关闭文件后下次评估即可自动更新。")
    except Exception as e:
        print(f"\n[警告] 生成 Excel 报表失败: {e}")
        print("       (提示: 确保已安装 xlsxwriter: pip install xlsxwriter)")


# ==========================================
# 物理仿真接口
# ==========================================
def simulate_physics(structures, eng):
    device = structures.device
    N = structures.shape[0]
    structures_np = structures.squeeze(1).cpu().numpy().astype(float)
    # (N, H, W) -> (W, H, N) 物理转置修正
    structures_permuted = structures_np.transpose(2, 1, 0)
    input_mat = matlab.double(structures_permuted.tolist())
    
    try:
        results_mat = eng.solve_batch(input_mat, nargout=1)
        results_np = np.array(results_mat) 
        pred_targets = torch.from_numpy(results_np).float().to(device)
        return pred_targets
    except Exception as e:
        print(f"\n[MATLAB Error] {e}")
        return torch.zeros(N, 2, device=device)

def get_clean_exp_name(ckpt_path):
    path_norm = os.path.normpath(ckpt_path)
    parts = path_norm.split(os.sep)
    if len(parts) >= 3:
        return parts[-3]
    return "default_exp"

def infer_image_size_from_state_dict(state_dict):
    pos_embed = state_dict.get("pos_embed")
    proj_weight = state_dict.get("x_embedder.proj.weight")
    if pos_embed is None or proj_weight is None:
        return None

    try:
        num_patches = int(pos_embed.shape[1])
        grid_size = int(math.isqrt(num_patches))
        if grid_size * grid_size != num_patches:
            return None

        patch_h = int(proj_weight.shape[2])
        patch_w = int(proj_weight.shape[3])
        if patch_h != patch_w:
            return None

        return grid_size * patch_h
    except Exception:
        return None

def main(args):
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 启动 MATLAB
    print(f"正在启动 MATLAB 引擎 (目标路径: {MATLAB_SCRIPT_PATH})...")
    if not os.path.exists(MATLAB_SCRIPT_PATH):
        print(f"错误: 找不到路径 {MATLAB_SCRIPT_PATH}")
        return
    try:
        eng = matlab.engine.start_matlab()
        eng.addpath(eng.genpath(MATLAB_SCRIPT_PATH), nargout=0)
        print("MATLAB 引擎启动成功！")
    except Exception as e:
        print(f"MATLAB 引擎启动失败: {e}")
        return

    # 2. 准备测试集
    grid_size = 10 
    real_vals = np.linspace(-0.9, 0.9, grid_size)
    imag_vals = np.linspace(-0.9, 0.9, grid_size)
    grid_real, grid_imag = np.meshgrid(real_vals, imag_vals)
    
    raw_targets = np.stack([grid_real.flatten(), grid_imag.flatten()], axis=1)
    magnitudes = np.linalg.norm(raw_targets, axis=1)
    valid_mask = magnitudes < 1.0 
    filtered_targets = raw_targets[valid_mask]
    
    test_targets_original = torch.from_numpy(filtered_targets).float().to(device)
    num_original_targets = len(test_targets_original)
    
    print(f"准备测试集: {num_original_targets} 个目标点 (Best-of-{NUM_REPEATS})")

    test_targets_expanded = test_targets_original.repeat_interleave(NUM_REPEATS, dim=0)

    # 3. 扫描 Checkpoints
    ckpt_list = []
    for ckpt_root in args.ckpt_dirs:
        if os.path.isdir(ckpt_root):
            ckpt_list.extend(sorted(glob(os.path.join(ckpt_root, "**/*.pt"), recursive=True)))
        else:
            ckpt_list.append(ckpt_root)
    
    if len(ckpt_list) == 0:
        print(f"错误: 未找到 .pt 文件")
        return

    # 4. 初始化总表 (CSV)
    os.makedirs(args.out_dir, exist_ok=True)
    summary_csv_path = os.path.join(args.out_dir, "evaluation_summary.csv")
    
    if not os.path.exists(summary_csv_path):
        with open(summary_csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Timestamp", "Experiment", "Checkpoint", 
                "Physics MAE (Mean)", "Physics MAE (Max)", 
                "Success Rate (%)", "Avg Diversity", "Avg Fill Factor"
            ])

    # 5. 循环评估
    for ckpt_path in ckpt_list:
        print(f"\n------------------------------------------------")
        print(f"正在评估: {ckpt_path}")
        
        # --- 创建本次评估的专属文件夹 ---
        eval_timestamp = time.strftime("%Y%m%d-%H%M%S")
        exp_name = get_clean_exp_name(ckpt_path)
        ckpt_name = os.path.basename(ckpt_path)
        
        eval_folder_name = f"eval_{exp_name}_{ckpt_name}_{eval_timestamp}"
        current_eval_dir = os.path.join(args.out_dir, eval_folder_name)
        os.makedirs(current_eval_dir, exist_ok=True)
        
        print(f"  -> 本次评估结果将保存在: {current_eval_dir}")

        # --- 加载模型与生成 ---
        state_dict = find_model(ckpt_path)
        image_size = args.image_size
        if image_size is None:
            image_size = infer_image_size_from_state_dict(state_dict)
            if image_size is None:
                print("错误: 无法从checkpoint推断 image_size，请手动指定 --image-size")
                return

        model = DiT_models[args.model](
            input_size=image_size,
            num_classes=args.num_classes
        ).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        diffusion = create_diffusion(str(args.num_sampling_steps))
        
        # 生成
        # n_samples_total = len(test_targets_expanded)
        z = torch.randn(len(test_targets_expanded), 1, image_size, image_size, device=device)
        y = test_targets_expanded
        n_samples = len(test_targets_expanded)

        # Setup CFG (Classifier-Free Guidance)
        if DEFAULT_CFG_SCALE > 1.0:
            z = torch.cat([z, z], 0)
            
            # [修正] 必须使用 [2.0, 2.0] 作为 Null Token，与 training/sample 保持一致
            # 创建一个 [1, 2] 的向量然后复制 n_samples 份
            y_null = torch.tensor([[2.0, 2.0]], device=device).repeat(n_samples, 1)
            # y_null = torch.zeros_like(y) # 使用全零作为 Null Token
            
            y_combined = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y_combined, cfg_scale=DEFAULT_CFG_SCALE)
            sample_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y) # 原始版本
            sample_fn = model.forward # 原始版本
            # # ================= [实验修改] =================
            # # 进入这个分支意味着 args.cfg_scale <= 1.0 (通常是 1.0)
            # # 我们在这里强行执行 "噪声放大实验"
            
            # # 定义放大的倍数 (因为此时 args.cfg_scale 是 1.0，不能用它做倍数，必须手写 4.0)
            # FORCED_SCALE = 4.0 
            
            # print(f" [实验模式] 进入标准分支 (Scale=1.0)，但强制放大噪声 {FORCED_SCALE} 倍")

            # def forward_with_forced_scaling(x, t, y, **kwargs):
            #     # 1. 正常的单次前向传播
            #     model_out = model(x, t, y)
                
            #     # 2. 拆分噪声(eps)和方差(rest)
            #     C = model.in_channels
            #     eps, rest = model_out[:, :C], model_out[:, C:]
                
            #     # 3. [核心] 强制放大
            #     eps = eps * FORCED_SCALE
                
            #     # 4. 拼回去
            #     return torch.cat([eps, rest], dim=1)

            # sample_fn = forward_with_forced_scaling
            # model_kwargs = dict(y=y)
            # # ================= [修改结束] =================

        start_time = time.time()
        samples = diffusion.p_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
        )
        if DEFAULT_CFG_SCALE > 1.0:
            samples, _ = samples.chunk(2, dim=0)
        print(f"生成耗时: {time.time() - start_time:.2f}s")

        # --- 物理仿真 ---
        structures_binary_all = (samples > 0).float()
        print(f"正在计算物理属性...")
        pred_properties_all = simulate_physics(structures_binary_all, eng)
        
        # --- 统计指标 ---
        errors_all = torch.abs(test_targets_expanded - pred_properties_all)
        error_mag_all = torch.sqrt(errors_all[:, 0]**2 + errors_all[:, 1]**2)
        
        error_mag_grouped = error_mag_all.view(num_original_targets, NUM_REPEATS)
        min_error_per_target, best_local_indices = torch.min(error_mag_grouped, dim=1)
        
        success_count = (min_error_per_target < SUCCESS_THRESHOLD).sum().item()
        success_rate = 100.0 * success_count / num_original_targets
        
        structures_grouped = structures_binary_all.view(num_original_targets, NUM_REPEATS, -1)
        diversity_score = structures_grouped.std(dim=1).mean().item()
        
        offsets = torch.arange(num_original_targets, device=device) * NUM_REPEATS
        best_global_indices = offsets + best_local_indices
        best_structures = structures_binary_all[best_global_indices]   
        best_errors_abs = errors_all[best_global_indices]              
        best_samples_raw= samples[best_global_indices]

        avg_fill_factor = best_structures.mean().item()

        samples_01 = (best_samples_raw + 1) / 2
        invalid_mask = (samples_01 > 0.2) & (samples_01 < 0.8)
        validity_score = 100 * (1 - (invalid_mask.float().mean().item()))
        mean_mae = best_errors_abs.mean().item()
        max_mae = best_errors_abs.max().item()
        
        print(f"  -> MAE: {mean_mae:.4f} | Success: {success_rate:.1f}% | Div: {diversity_score:.3f}")

        # --- 绘图 & 保存 ---
        # 1. Error Map
        targets_x = test_targets_original[:, 0].cpu().numpy()
        targets_y = test_targets_original[:, 1].cpu().numpy()
        errors_val = min_error_per_target.cpu().numpy() 
        
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        circle = plt.Circle((0, 0), 1, color='gray', fill=False, linestyle='--', linewidth=1.5)
        ax.add_patch(circle)
        sc = plt.scatter(targets_x, targets_y, c=errors_val, cmap='Reds', vmin=0, s=60, edgecolors='k', linewidth=0.5)
        plt.xlim(-1.1, 1.1); plt.ylim(-1.1, 1.1)
        plt.xlabel('Re(T)'); plt.ylabel('Im(T)')
        plt.title(f'{exp_name}\nMAE:{mean_mae:.4f} | Success:{success_rate:.1f}%')
        plt.grid(True, linestyle=':', alpha=0.6); plt.axis('equal')
        plt.colorbar(sc, fraction=0.046, pad=0.04).set_label('Error Magnitude')
        plt.savefig(os.path.join(current_eval_dir, "error_map.png"), dpi=100, bbox_inches='tight')
        plt.close()

        # 2. Images
        images_dir = os.path.join(current_eval_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        for i in range(num_original_targets):
            tr = test_targets_original[i, 0].item()
            ti = test_targets_original[i, 1].item()
            img = best_structures[i].transpose(1, 2)
            img = F.interpolate(img.unsqueeze(0), size=(256, 256), mode='nearest').squeeze(0)
            save_image(1.0 - img, os.path.join(images_dir, f"sample_{i:03d}_{tr:.2f}_{ti:.2f}.png"))

        # 3. Trials CSV
        col_target_re = test_targets_expanded[:, 0].cpu().numpy()
        col_target_im = test_targets_expanded[:, 1].cpu().numpy()
        col_pred_re   = pred_properties_all[:, 0].cpu().numpy()
        col_pred_im   = pred_properties_all[:, 1].cpu().numpy()
        col_error     = error_mag_all.cpu().numpy()
        col_fill      = structures_binary_all.mean(dim=[1, 2, 3]).cpu().numpy()
        col_repeat_id = np.tile(np.arange(NUM_REPEATS), num_original_targets)
        col_sample_id = np.repeat(np.arange(num_original_targets), NUM_REPEATS)

        df_trials = pd.DataFrame({
            'Sample_ID': col_sample_id,
            'Repeat_ID': col_repeat_id,
            'Target_Real': col_target_re,
            'Target_Imag': col_target_im,
            'Pred_Real': col_pred_re,
            'Pred_Imag': col_pred_im,
            'Error': col_error,
            'Fill_Factor': col_fill
        })
        df_trials.to_csv(os.path.join(current_eval_dir, "trials.csv"), index=False)

        # 4. 追加写入 CSV 总表
        with open(summary_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                eval_timestamp, exp_name, ckpt_name, 
                f"{mean_mae:.6f}", f"{max_mae:.6f}",
                f"{success_rate:.1f}", 
                f"{diversity_score:.4f}", 
                f"{avg_fill_factor:.4f}"
            ])
            
    # [关键] 每次所有 Checkpoints 跑完后，刷新一下 Excel 格式化文件
    update_summary_excel(summary_csv_path)

    print(f"\n所有评估完成！")
    eng.quit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-Tiny1")
    parser.add_argument("--image-size", type=int, default=None, help="Auto-infer from checkpoint if omitted")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--ckpt-dirs", nargs="*", default=DEFAULT_CKPT_DIRS, help="Checkpoint files or directories")
    parser.add_argument("--out-dir", type=str, default="eval_results", help="Output directory")
    
    args = parser.parse_args()
    main(args)
