"""
============================================================================
Generative AI for EV Charging Data — Comprehensive Metric Evaluation Script
============================================================================
毕设: Generative AI and Reinforcement Learning for Electric Vehicle Management

评估维度:
  1) 边缘分布保真度  — KS statistic, Wasserstein-1
  2) 变量间依赖结构  — 相关矩阵 Frobenius 差异
  3) 时序结构保持度  — 自相关函数 MAE
  4) 联合分布覆盖度  — Coverage (真实数据被"覆盖"的比例)
  5) 极端值捕获能力  — Tail divergence (q99/q01 相对误差)

输出: 每个 (模型, 场景, 变量) 组合的指标 → 保存为 CSV
============================================================================
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist
import warnings
import os
import json

warnings.filterwarnings("ignore")


# ============================================================================
# 0. 配置区 — 请根据你的实际文件路径和列名修改
# ============================================================================

CONFIG = {
    # 模型名称列表 (real 作为 ground truth, 其余为待评估的生成模型)
    "models": ["real", "CVAE-RunB", "GAN-tuned1", "diffusion"],

    # 场景列表
    "scenarios": ["mainB", "stressA"],

    # 需要评估的变量列名 (与你的 CSV 文件中的列名一致)
    "variables": ["price", "load", "lambda"],

    # 数据文件路径模板 — 用 {model} 和 {scenario} 占位
    # 例如: "data/mainB/real.csv", "data/stressA/CVAE-RunB.csv"
    "data_path_template": "data/{scenario}/{model}.csv",

    # 自相关评估的最大滞后步数
    "acf_max_lag": 48,

    # Coverage 指标的 k-NN 参数
    "coverage_k": 5,

    # Coverage 采样数 (加速计算, 设为 None 则用全量)
    "coverage_sample_n": 2000,
}


# ============================================================================
# 1. 数据加载
# ============================================================================

def load_data(model: str, scenario: str) -> pd.DataFrame:
    """
    加载指定模型在指定场景下的数据。
    返回 DataFrame, 列名需包含 CONFIG["variables"] 中的变量。

    --- 请根据你的实际数据格式修改此函数 ---
    """
    path = CONFIG["data_path_template"].format(model=model, scenario=scenario)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到文件: {path}\n"
            f"请修改 CONFIG['data_path_template'] 或将数据放到对应路径。"
        )
    df = pd.read_csv(path)
    missing = [v for v in CONFIG["variables"] if v not in df.columns]
    if missing:
        raise ValueError(f"文件 {path} 缺少以下列: {missing}")
    return df


# ============================================================================
# 2. 指标计算函数
# ============================================================================

# ---------- 2.1 边缘分布保真度 ----------

def compute_ks(real: np.ndarray, gen: np.ndarray) -> float:
    """KS 统计量 (↓ 越小越好): 两个样本的经验CDF最大差异"""
    stat, _ = stats.ks_2samp(real, gen)
    return round(stat, 6)


def compute_wasserstein1(real: np.ndarray, gen: np.ndarray) -> float:
    """Wasserstein-1 距离 (↓): Earth Mover's Distance 的一维版本"""
    return round(stats.wasserstein_distance(real, gen), 6)


# ---------- 2.2 变量间依赖结构 ----------

def compute_corr_diff(real_df: pd.DataFrame, gen_df: pd.DataFrame,
                      variables: list) -> float:
    """
    相关矩阵 Frobenius 范数差异 (↓ 越小越好)。
    衡量生成数据是否保留了变量间的相关结构。
    """
    corr_real = real_df[variables].corr().values
    corr_gen = gen_df[variables].corr().values
    diff = np.linalg.norm(corr_real - corr_gen, "fro")
    # 归一化: 除以矩阵维度, 使得不同变量数量下可比
    n = len(variables)
    return round(diff / n, 6)


# ---------- 2.3 时序结构保持度 ----------

def _acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    """计算自相关函数 (unnormalized → normalized)"""
    n = len(x)
    x_centered = x - x.mean()
    acf_vals = np.correlate(x_centered, x_centered, mode="full")
    acf_vals = acf_vals[n - 1:]  # 取正半轴
    acf_vals = acf_vals / acf_vals[0] if acf_vals[0] != 0 else acf_vals
    return acf_vals[: max_lag + 1]


def compute_acf_mae(real: np.ndarray, gen: np.ndarray,
                    max_lag: int = 48) -> float:
    """
    自相关函数 MAE (↓ 越小越好)。
    比较真实数据和生成数据在 [0, max_lag] 内自相关曲线的平均绝对差异。
    """
    acf_r = _acf(real, max_lag)
    acf_g = _acf(gen, min(max_lag, len(gen) - 1))
    min_len = min(len(acf_r), len(acf_g))
    mae = np.mean(np.abs(acf_r[:min_len] - acf_g[:min_len]))
    return round(mae, 6)


# ---------- 2.4 联合分布覆盖度 ----------

def compute_coverage(real_df: pd.DataFrame, gen_df: pd.DataFrame,
                     variables: list, k: int = 5,
                     sample_n: int = None) -> float:
    """
    Coverage (↑ 越高越好, 范围 [0, 1])。
    对真实数据中的每个点, 找其在生成数据中的 k-NN 距离;
    如果该距离 ≤ 该点在真实数据自身中的 k-NN 距离, 则认为被"覆盖"。
    用于检测 mode collapse。
    """
    real_vals = real_df[variables].values.astype(np.float64)
    gen_vals = gen_df[variables].values.astype(np.float64)

    # 标准化 (用真实数据的 mean/std)
    mu = real_vals.mean(axis=0)
    sigma = real_vals.std(axis=0) + 1e-8
    real_norm = (real_vals - mu) / sigma
    gen_norm = (gen_vals - mu) / sigma

    # 采样加速
    if sample_n and sample_n < len(real_norm):
        idx_r = np.random.choice(len(real_norm), sample_n, replace=False)
        real_norm = real_norm[idx_r]
    if sample_n and sample_n < len(gen_norm):
        idx_g = np.random.choice(len(gen_norm), sample_n, replace=False)
        gen_norm = gen_norm[idx_g]

    # 真实→真实 k-NN 距离
    dist_rr = cdist(real_norm, real_norm, metric="euclidean")
    np.fill_diagonal(dist_rr, np.inf)
    knn_rr = np.sort(dist_rr, axis=1)[:, k - 1]  # 第 k 近邻距离

    # 真实→生成 k-NN 距离
    dist_rg = cdist(real_norm, gen_norm, metric="euclidean")
    knn_rg = np.sort(dist_rg, axis=1)[:, min(k - 1, dist_rg.shape[1] - 1)]

    coverage = np.mean(knn_rg <= knn_rr)
    return round(coverage, 6)


# ---------- 2.5 极端值捕获能力 ----------

def compute_tail_divergence(real: np.ndarray, gen: np.ndarray) -> dict:
    """
    尾部散度: q01 和 q99 的相对误差 (↓ 越接近0越好)。
    返回 {"q01_rel_err": ..., "q99_rel_err": ...}
    """
    results = {}
    for q, label in [(1, "q01"), (99, "q99")]:
        r_val = np.percentile(real, q)
        g_val = np.percentile(gen, q)
        denom = max(abs(r_val), 1e-8)
        rel_err = (g_val - r_val) / denom
        results[f"{label}_rel_err"] = round(rel_err, 6)
    return results


# ============================================================================
# 3. 主评估流程
# ============================================================================

def evaluate_all() -> pd.DataFrame:
    """
    遍历所有 (模型, 场景) 组合, 计算全部指标, 返回汇总 DataFrame。
    """
    rows = []
    real_cache = {}  # 缓存 real 数据

    for scenario in CONFIG["scenarios"]:
        # 加载 real 数据
        if scenario not in real_cache:
            real_cache[scenario] = load_data("real", scenario)
        real_df = real_cache[scenario]

        for model in CONFIG["models"]:
            if model == "real":
                continue  # 跳过 real vs real

            print(f"  评估中: {model} @ {scenario} ...")
            gen_df = load_data(model, scenario)

            # --- 联合指标 (所有变量一起) ---
            corr_diff = compute_corr_diff(
                real_df, gen_df, CONFIG["variables"]
            )
            coverage = compute_coverage(
                real_df, gen_df, CONFIG["variables"],
                k=CONFIG["coverage_k"],
                sample_n=CONFIG["coverage_sample_n"]
            )

            # --- 逐变量指标 ---
            for var in CONFIG["variables"]:
                r = real_df[var].dropna().values
                g = gen_df[var].dropna().values

                ks = compute_ks(r, g)
                w1 = compute_wasserstein1(r, g)
                acf_mae = compute_acf_mae(
                    r, g, max_lag=CONFIG["acf_max_lag"]
                )
                tail = compute_tail_divergence(r, g)

                rows.append({
                    "scenario": scenario,
                    "model": model,
                    "variable": var,
                    "KS↓": ks,
                    "W1↓": w1,
                    "ACF_MAE↓": acf_mae,
                    "q01_rel_err": tail["q01_rel_err"],
                    "q99_rel_err": tail["q99_rel_err"],
                    "CorrMatrix_FroDiff↓": corr_diff,
                    "Coverage↑": coverage,
                })

    df = pd.DataFrame(rows)
    return df


# ============================================================================
# 4. 输出格式化与保存
# ============================================================================

def format_and_save(df: pd.DataFrame, output_path: str = "genai_metrics.csv"):
    """保存结果到 CSV, 并在终端打印格式化输出。"""

    # 保存 CSV
    df.to_csv(output_path, index=False, float_format="%.6f")
    print(f"\n✅ 指标已保存至: {output_path}\n")

    # 终端打印 (按场景分块)
    for scenario in CONFIG["scenarios"]:
        print(f"\n{'='*80}")
        print(f"  场景: {scenario}")
        print(f"{'='*80}")
        sub = df[df["scenario"] == scenario]
        print(sub.to_string(index=False))

    # 额外: 按模型汇总 (所有变量均值)
    print(f"\n{'='*80}")
    print("  模型级汇总 (各变量指标均值)")
    print(f"{'='*80}")
    metric_cols = ["KS↓", "W1↓", "ACF_MAE↓", "CorrMatrix_FroDiff↓", "Coverage↑"]
    summary = df.groupby(["scenario", "model"])[metric_cols].mean().round(6)
    print(summary.to_string())


# ============================================================================
# 5. 入口
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  GenAI EV Charging Data — Metric Evaluation")
    print("=" * 60)

    # --- 如果你的数据还没准备好, 可以先跑 demo 模式 ---
    DEMO_MODE = False  # 改为 False 后使用真实数据

    if DEMO_MODE:
        print("\n⚠️  DEMO 模式: 使用随机生成的模拟数据演示指标计算\n")
        np.random.seed(42)
        N = 5000

        # 模拟 real 数据: 三个变量有一定相关性和时序结构
        t = np.arange(N)
        price_real = 0.5 + 0.3 * np.sin(2 * np.pi * t / 24) + 0.05 * np.random.randn(N)
        load_real = 50 + 30 * np.sin(2 * np.pi * t / 24 + 1) + 5 * np.random.randn(N)
        lam_real = 0.1 + 0.05 * np.sin(2 * np.pi * t / 24) + 0.01 * np.random.randn(N)

        real_data = {
            "mainB": pd.DataFrame({"price": price_real, "load": load_real, "lambda": lam_real}),
            "stressA": pd.DataFrame({
                "price": price_real * 1.5 + 0.1 * np.random.randn(N),
                "load": load_real * 2 + 10 * np.random.randn(N),
                "lambda": lam_real * 1.3 + 0.02 * np.random.randn(N),
            }),
        }

        def make_gen(real_df, noise_scale, corr_distort, acf_distort):
            """模拟不同质量的生成数据"""
            gen = real_df.copy()
            for col in gen.columns:
                vals = gen[col].values
                # 加噪声 (影响边缘分布)
                gen[col] = vals + noise_scale * vals.std() * np.random.randn(N)
                # 打乱一部分时序 (影响自相关)
                if acf_distort > 0:
                    n_swap = int(N * acf_distort)
                    idx = np.random.choice(N, n_swap, replace=False)
                    gen.loc[idx, col] = np.random.permutation(gen.loc[idx, col].values)
            # 打乱部分变量间关系 (影响相关矩阵)
            if corr_distort > 0:
                n_swap = int(N * corr_distort)
                idx = np.random.choice(N, n_swap, replace=False)
                gen.loc[idx, "price"] = np.random.permutation(gen.loc[idx, "price"].values)
            return gen

        # 模拟三个生成模型 (质量从高到低: diffusion > CVAE-RunB > GAN-tuned1)
        gen_configs = {
            "CVAE-RunB":  {"noise_scale": 0.08, "corr_distort": 0.05, "acf_distort": 0.10},
            "GAN-tuned1": {"noise_scale": 0.15, "corr_distort": 0.12, "acf_distort": 0.20},
            "diffusion":  {"noise_scale": 0.03, "corr_distort": 0.02, "acf_distort": 0.03},
        }

        # 构建结果
        rows = []
        for scenario in CONFIG["scenarios"]:
            real_df = real_data[scenario]
            for model, cfg in gen_configs.items():
                gen_df = make_gen(real_df, **cfg)

                corr_diff = compute_corr_diff(real_df, gen_df, CONFIG["variables"])
                coverage = compute_coverage(
                    real_df, gen_df, CONFIG["variables"],
                    k=CONFIG["coverage_k"], sample_n=CONFIG["coverage_sample_n"]
                )

                for var in CONFIG["variables"]:
                    r = real_df[var].values
                    g = gen_df[var].values
                    tail = compute_tail_divergence(r, g)
                    rows.append({
                        "scenario": scenario,
                        "model": model,
                        "variable": var,
                        "KS↓": compute_ks(r, g),
                        "W1↓": compute_wasserstein1(r, g),
                        "ACF_MAE↓": compute_acf_mae(r, g, CONFIG["acf_max_lag"]),
                        "q01_rel_err": tail["q01_rel_err"],
                        "q99_rel_err": tail["q99_rel_err"],
                        "CorrMatrix_FroDiff↓": corr_diff,
                        "Coverage↑": coverage,
                    })

        result_df = pd.DataFrame(rows)
        format_and_save(result_df, "genai_metrics_demo.csv")

    else:
        # 真实数据模式
        result_df = evaluate_all()
        format_and_save(result_df, "genai_metrics.csv")

    print("\n✅ 完成!")
