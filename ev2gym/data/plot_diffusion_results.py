import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('default')
# 手动配置网格样式，与"whitegrid"效果一致
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["grid.color"] = "#dddddd"
plt.rcParams["axes.facecolor"] = "white"

plt.rcParams.update({
    "font.family": "Arial",  # 英文标准字体
    "font.size": 11,         # 基础字体大小
    "axes.titlesize": 13,    # 标题字体大小
    "axes.labelsize": 12,    # 坐标轴标签大小
    "legend.fontsize": 10,   # 图例字体大小
    "xtick.labelsize": 10,   # X轴刻度大小
    "ytick.labelsize": 10,   # Y轴刻度大小
    "axes.edgecolor": "#333333",  # 坐标轴边框颜色
    "grid.alpha": 0.2        # 网格透明度（不抢主体）
})

# 核心指标（和你的数据集列名对应）
MAIN_VARS = ["price", "pv", "load", "lambda"]
# 专业配色（对比明显且不刺眼）
COLOR_REAL = "#1f77b4"    # 真实数据：深蓝色
COLOR_DIFFUSION = "#ff7f0e"  # 扩散模型：橙色
COLOR_BEST = "#2ca02c"    # 最优损失：绿色


def plot_single_day_comparison(real_df, gen_df, save_dir):
    """Single day time-series comparison (real vs diffusion-generated)"""
    os.makedirs(save_dir, exist_ok=True)
    # Get day 0 data
    real_day0 = real_df[real_df["day_id"] == 0].sort_values("t").reset_index(drop=True)
    gen_day0 = gen_df[gen_df["day_id"] == 0].sort_values("t").reset_index(drop=True)

    for var in MAIN_VARS:
        if var not in real_df.columns or var not in gen_df.columns:
            print(f"Warning: Column '{var}' not found, skipping plot.")
            continue
        
        # 调整图尺寸（更修长，适合汇报）
        plt.figure(figsize=(14, 4.5))
        # Plot curves (更细的线条+标记，增强可读性)
        plt.plot(real_day0["t"], real_day0[var], 
                 label="Real Data", color=COLOR_REAL, linewidth=2.2, 
                 marker="o", markersize=3.5, markevery=4)  # 每4步显示一个标记
        plt.plot(gen_day0["t"], gen_day0[var], 
                 label="Diffusion-Generated", color=COLOR_DIFFUSION, linewidth=2.2,
                 marker="s", markersize=3.5, markevery=4)
        
        # 坐标轴与标题
        plt.xlabel("Time Step (15min interval, 96 steps/day)", fontweight="light")
        plt.ylabel(var.replace("_", " ").title())  # 美化指标名（如price_eur→Price Eur）
        plt.title(f"Single-Day Time-Series Comparison: {var.replace('_', ' ').title()}", 
                  fontweight="bold", pad=15)
        # 图例（放在右上角，不遮挡曲线）
        plt.legend(loc="upper right", framealpha=0.9, edgecolor="#dddddd")
        # 调整X轴刻度（避免密集）
        plt.xticks(np.arange(0, 97, 12))  # 每12步显示一个刻度
        
        # 保存（高分辨率+无白边）
        plt.savefig(os.path.join(save_dir, f"diffusion_{var}_day0_comparison.png"), 
                    dpi=350, bbox_inches="tight", facecolor="white")
        plt.close()
    print("✅ Single-day time-series plots saved.")


def compute_stats_table(real_df, gen_df, save_dir):
    """Statistical distribution comparison table (mean/std/quantiles)"""
    stats_list = []
    for var in MAIN_VARS:
        if var not in real_df.columns or var not in gen_df.columns:
            continue
        
        # Calculate stats
        real_mean = real_df[var].mean()
        real_std = real_df[var].std()
        real_p50 = real_df[var].quantile(0.5)
        real_p90 = real_df[var].quantile(0.9)
        
        gen_mean = gen_df[var].mean()
        gen_std = gen_df[var].std()
        gen_p50 = gen_df[var].quantile(0.5)
        gen_p90 = gen_df[var].quantile(0.9)
        
        # Calculate percentage difference
        mean_diff_pct = (gen_mean - real_mean) / real_mean * 100
        std_diff_pct = (gen_std - real_std) / real_std * 100
        p50_diff_pct = (gen_p50 - real_p50) / real_p50 * 100
        p90_diff_pct = (gen_p90 - real_p90) / real_p90 * 100
        
        stats_list.append({
            "Variable": var.replace("_", " ").title(),
            "Real Mean": round(real_mean, 3),
            "Gen Mean": round(gen_mean, 3),
            "Mean Diff (%)": round(mean_diff_pct, 2),
            "Real Std": round(real_std, 3),
            "Gen Std": round(gen_std, 3),
            "Std Diff (%)": round(std_diff_pct, 2),
            "Real P50": round(real_p50, 3),
            "Gen P50": round(gen_p50, 3),
            "P50 Diff (%)": round(p50_diff_pct, 2),
            "Real P90": round(real_p90, 3),
            "Gen P90": round(gen_p90, 3),
            "P90 Diff (%)": round(p90_diff_pct, 2)
        })
    
    stats_df = pd.DataFrame(stats_list)
    # Save CSV
    stats_df.to_csv(os.path.join(save_dir, "diffusion_stats_comparison.csv"), 
                    index=False, encoding="utf-8")
    
    # Plot table (更美观的样式)
    fig, ax = plt.subplots(figsize=(16, len(stats_df)*1.4))
    ax.axis("off")  # 隐藏坐标轴
    # Create table
    table = ax.table(cellText=stats_df.values, 
                     colLabels=stats_df.columns, 
                     cellLoc="center", 
                     loc="center",
                     colColours=[COLOR_REAL]*len(stats_df.columns))  # 表头用真实数据配色
    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 2.2)  # 调整单元格大小
    # Format header text
    for i in range(len(stats_df.columns)):
        table[(0, i)].set_text_props(weight="bold", color="white")
    
    plt.title("Statistical Distribution Comparison: Real vs Diffusion-Generated Data", 
              fontweight="bold", pad=25)
    plt.savefig(os.path.join(save_dir, "diffusion_stats_table.png"), 
                dpi=350, bbox_inches="tight", facecolor="white")
    plt.close()
    print("✅ Statistical comparison table saved (CSV + plot).")


def plot_training_curve(loss_history_path, save_dir):
    """Diffusion model training curve (train/val loss)"""
    if not os.path.exists(loss_history_path):
        print("Warning: Loss history file not found, skipping training curve.")
        return
    
    loss_history = np.load(loss_history_path, allow_pickle=True).item()
    train_loss = loss_history["train"]
    val_loss = loss_history["val"]
    epochs = len(train_loss)
    best_val_loss = min(val_loss)
    best_epoch = val_loss.index(best_val_loss) + 1

    plt.figure(figsize=(12, 5))
    # Plot loss curves
    plt.plot(range(1, epochs+1), train_loss, 
             label="Training Loss", color=COLOR_REAL, linewidth=2, linestyle="-")
    plt.plot(range(1, epochs+1), val_loss, 
             label="Validation Loss", color=COLOR_DIFFUSION, linewidth=2, linestyle="-")
    # Mark best validation loss
    plt.scatter(best_epoch, best_val_loss, 
                color=COLOR_BEST, s=80, zorder=5, edgecolor="white", linewidth=1)
    plt.annotate(f"Best Val Loss: {best_val_loss:.4f}\nEpoch: {best_epoch}",
                 xy=(best_epoch, best_val_loss), xytext=(20, -20),
                 textcoords="offset points",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc"),
                 arrowprops=dict(arrowstyle="->", color="#666666"))
    
    plt.xlabel("Training Epoch", fontweight="light")
    plt.ylabel("Loss (MSE)", fontweight="light")
    plt.title("Diffusion Model Training Curve", fontweight="bold", pad=15)
    plt.legend(loc="upper right")
    plt.xticks(np.arange(0, epochs+1, epochs//10))  # 均匀显示10个刻度
    
    plt.savefig(os.path.join(save_dir, "diffusion_training_curve.png"), 
                dpi=350, bbox_inches="tight", facecolor="white")
    plt.close()
    print("✅ Training curve saved.")


def plot_acf_comparison(real_df, gen_df, save_dir, max_lag=24):
    """ACF (Auto-Correlation Function) comparison"""
    def acf(x, max_lag):
        x = x - x.mean()
        ac = [1.0]
        # 先计算全局分母（避免重复计算）
        denominator = np.dot(x, x)
        # 处理分母为0的极端情况
        denominator = denominator if denominator != 0 else 1
        for lag in range(1, max_lag+1):
            if len(x) <= lag:
                numerator = 0
            else:
                numerator = np.dot(x[:-lag], x[lag:])
            ac.append(numerator / denominator)
        return np.array(ac)
    
    for var in MAIN_VARS:
        if var not in real_df.columns or var not in gen_df.columns:
            continue
        
        real_acf = acf(real_df[var].values, max_lag)
        gen_acf = acf(gen_df[var].values, max_lag)
        lags = np.arange(0, max_lag+1)

        plt.figure(figsize=(13, 4.5))
        # Plot ACF curves
        plt.plot(lags, real_acf, 
                 label="Real Data", color=COLOR_REAL, linewidth=2,
                 marker="o", markersize=4, markevery=2)
        plt.plot(lags, gen_acf, 
                 label="Diffusion-Generated", color=COLOR_DIFFUSION, linewidth=2,
                 marker="s", markersize=4, markevery=2)
        plt.axhline(y=0, color="#888888", linestyle="--", linewidth=1.2)  # 零线
        
        plt.xlabel(f"Lag (Max: {max_lag})", fontweight="light")
        plt.ylabel("Auto-Correlation Coefficient", fontweight="light")
        plt.title(f"Auto-Correlation Function (ACF) Comparison: {var.replace('_', ' ').title()}",
                  fontweight="bold", pad=15)
        plt.legend(loc="upper right")
        plt.xticks(lags[::2])  # 每2个lag显示一个刻度
        
        plt.savefig(os.path.join(save_dir, f"diffusion_{var}_acf_comparison.png"),
                    dpi=350, bbox_inches="tight", facecolor="white")
        plt.close()
    print("✅ ACF comparison plots saved.")


def main(args):
    print("Loading data...")
    real_df = pd.read_csv(args.real_csv).sort_values(["day_id", "t"]).reset_index(drop=True)
    gen_files = sorted(glob.glob(os.path.join(args.gen_dir, "gen_day_*.csv")))
    if not gen_files:
        raise FileNotFoundError("No diffusion-generated files found. Run sample_diffusion.py first.")
    gen_df = pd.concat([pd.read_csv(f) for f in gen_files], ignore_index=True).sort_values(["day_id", "t"]).reset_index(drop=True)
    
    plot_single_day_comparison(real_df, gen_df, args.save_dir)
    compute_stats_table(real_df, gen_df, args.save_dir)
    plot_training_curve(args.loss_history, args.save_dir)
    plot_acf_comparison(real_df, gen_df, args.save_dir)
    
    print(f"\n🎉 All plots saved to: {args.save_dir}")
    print("\n=== Comparison Dimensions for Report ===")
    print("1. Time-Series Continuity: Smoothness of curves & rationality of peaks/valleys.")
    print("2. Statistical Consistency: Smaller % difference in mean/std/quantiles = better.")
    print("3. Training Stability: Fast convergence + no overfitting (val loss not rising).")
    print("4. Temporal Correlation: ACF curve trend consistency (higher correlation for small lags).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion Model Result Plotter (English + Professional Style)")
    parser.add_argument("--real_csv", default="dataset/scenario_train.csv", help="Path to real dataset CSV")
    parser.add_argument("--gen_dir", default="generated/samples", help="Directory of diffusion-generated files")
    parser.add_argument("--save_dir", default="generated/diffusion_plots", help="Directory to save plots")
    parser.add_argument("--loss_history", default="generated/diffusion/diffusion_loss_history.npy", help="Path to diffusion loss history")
    args = parser.parse_args()
    main(args)