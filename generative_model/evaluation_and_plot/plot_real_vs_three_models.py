# plot_real_vs_three_models.py
# 画：real vs CVAE-RunB vs GAN-tuned1 vs Diffusion Model（四图例）——直方图/剖面/日总量
# 用法：在 __main__ 底部配好路径，分别对 mainB / stressA 各跑一次即可
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
 
DT_HOUR = 0.25  # 15min=0.25h（若你不是15min，改这里）
 
TARGET_COLS = ["price", "load", "lambda"]
 
# ========== 四条图例的颜色 / 线型 ==========
MODEL_STYLES = {
    "Real": dict(color="#1f77b4", linewidth=2.0, linestyle="-",
                 hist_kw=dict(alpha=0.25, color="#1f77b4")),           # 蓝色填充
    "CVAE-RunB": dict(color="#d62728", linewidth=2.2, linestyle="--",
                      hist_kw=dict(histtype="step", color="#d62728", linewidth=2.2, linestyle="--")),
    "GAN-tuned1": dict(color="#2ca02c", linewidth=2.2, linestyle="-.",
                       hist_kw=dict(histtype="step", color="#2ca02c", linewidth=2.2, linestyle="-.")),
    "Diffusion Model": dict(color="#9467bd", linewidth=2.2, linestyle=":",
                            hist_kw=dict(histtype="step", color="#9467bd", linewidth=2.2, linestyle=":")),
}
 
# ========== 工具函数 ==========
 
def _as_list(x):
    if isinstance(x, np.ndarray):
        return [str(i) for i in x.tolist()]
    return list(x)
 
 
def load_real_as_df(real_path: Path) -> pd.DataFrame:
    """
    real_path:
      - CSV: 需要至少包含 day_id,t,price,load,lambda，并有 day_of_week 或 is_weekend
      - NPZ: 用 1_prepare_npz_cvae_from_scene.py 生成的格式
    """
    real_path = Path(real_path)
 
    if real_path.suffix.lower() == ".csv":
        df = pd.read_csv(real_path)
        need = {"day_id", "t"} | set(TARGET_COLS)
        miss = need - set(df.columns)
        if miss:
            raise ValueError(f"[REAL CSV] missing columns: {miss}")
        if "is_weekend" not in df.columns:
            if "day_of_week" not in df.columns:
                raise ValueError("[REAL CSV] need day_of_week or is_weekend.")
            df["is_weekend"] = (pd.to_numeric(df["day_of_week"]).fillna(0).astype(int) >= 5).astype(int)
        if "day_of_week" not in df.columns:
            df["day_of_week"] = 0
        df["is_weekend"] = pd.to_numeric(df["is_weekend"], errors="coerce").fillna(0).astype(int)
        df["day_of_week"] = pd.to_numeric(df["day_of_week"], errors="coerce").fillna(0).astype(int)
        n_wd = (df["is_weekend"] == 0).sum()
        n_we = (df["is_weekend"] == 1).sum()
        print(f"  [load_real] {real_path.name}: weekday={n_wd}, weekend={n_we}, "
              f"is_weekend dtype={df['is_weekend'].dtype}, unique={df['is_weekend'].unique()}")
        return df[["day_id", "t", "day_of_week", "is_weekend"] + TARGET_COLS].copy()
 
    if real_path.suffix.lower() == ".npz":
        z = np.load(real_path, allow_pickle=True)
        Xn = z["X"]
        mean = z["x_mean"].astype(np.float32)
        std  = z["x_std"].astype(np.float32)
        x_cols = _as_list(z["x_cols"])
        log1p_cols = _as_list(z["log1p_x_cols"]) if "log1p_x_cols" in z else []
 
        X = Xn * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)
        for c in log1p_cols:
            if c in x_cols:
                k = x_cols.index(c)
                X[:, :, k] = np.expm1(X[:, :, k])
                X[:, :, k] = np.clip(X[:, :, k], 0.0, None)
 
        C = z["C"]
        c_cols = _as_list(z["c_cols"])
        if "is_weekend" not in c_cols:
            raise ValueError("[REAL NPZ] c_cols has no 'is_weekend'")
        iw = c_cols.index("is_weekend")
        is_weekend = C[:, :, iw].astype(int)
 
        dow_cols = [f"dow_{k}" for k in range(7)]
        if all(c in c_cols for c in dow_cols):
            idxs = [c_cols.index(c) for c in dow_cols]
            dow = np.argmax(C[:, 0, idxs], axis=1).astype(int)
        else:
            dow = np.zeros((X.shape[0],), dtype=int)
 
        N, T, D = X.shape
        day_id = np.repeat(np.arange(N, dtype=int), T)
        t = np.tile(np.arange(T, dtype=int), N)
        df = pd.DataFrame({"day_id": day_id, "t": t})
        df["day_of_week"] = np.repeat(dow, T)
        df["is_weekend"]  = is_weekend.reshape(-1)
        for v in TARGET_COLS:
            if v not in x_cols:
                raise ValueError(f"[REAL NPZ] x_cols missing '{v}'")
            k = x_cols.index(v)
            df[v] = X[:, :, k].reshape(-1)
        return df[["day_id", "t", "day_of_week", "is_weekend"] + TARGET_COLS].copy()
 
    raise ValueError(f"Unsupported real file: {real_path}")
 
 
def load_gen_as_df(gen_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(gen_csv)
    need = {"day_id", "t", "is_weekend"} | set(TARGET_COLS)
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"[GEN CSV] {gen_csv} missing columns: {miss}")
    if "day_of_week" not in df.columns:
        df["day_of_week"] = 0
    # 强制转 int，防止 float/str/bool 导致筛选失败
    df["is_weekend"] = pd.to_numeric(df["is_weekend"], errors="coerce").fillna(0).astype(int)
    df["day_of_week"] = pd.to_numeric(df["day_of_week"], errors="coerce").fillna(0).astype(int)
    # 兼容 t 为全局递增编号（0..N*96-1）的情况，统一转为日内编号 0..95
    if df["t"].max() > 95:
        df["t"] = df["t"] % 96
    # 诊断：打印周末/工作日行数，方便排查
    n_wd = (df["is_weekend"] == 0).sum()
    n_we = (df["is_weekend"] == 1).sum()
    print(f"  [load_gen] {gen_csv.name}: weekday={n_wd}, weekend={n_we}, "
          f"is_weekend dtype={df['is_weekend'].dtype}, unique={df['is_weekend'].unique()}, "
          f"t range=[{df['t'].min()}..{df['t'].max()}]")
    return df[["day_id", "t", "day_of_week", "is_weekend"] + TARGET_COLS].copy()
 
 
# ========== 直方图：四图例 ==========
 
def plot_hist_4way(dfs: dict, var: str, out_path: Path, bins=60):
    """
    dfs: {"Real": df_real, "CVAE-RunB": df_cvae, "GAN-tuned1": df_gan, "Diffusion Model": df_diff}
    图例：Real 保持默认填充色块；三个模型用空心矩形框（edgecolor=颜色, facecolor='none'）
    """
    arrays = {name: df[var].to_numpy() for name, df in dfs.items()}
    lo = np.nanmin([a.min() for a in arrays.values()])
    hi = np.nanmax([a.max() for a in arrays.values()])
 
    plt.figure(dpi=250)
    legend_handles = []
    for name, arr in arrays.items():
        sty = MODEL_STYLES[name]
        if name == "Real":
            # Real：带填充的直方图，图例自动生成
            _, _, patches = plt.hist(arr, bins=bins, range=(lo, hi), density=True,
                                     alpha=0.25, color=sty["color"], label="_nolegend_")
            legend_handles.append(mpatches.Patch(
                facecolor=sty["color"], alpha=0.25, edgecolor=sty["color"], label=name))
        else:
            # 三个模型：step 直方图，图例用空心矩形框
            plt.hist(arr, bins=bins, range=(lo, hi), density=True,
                     histtype="step", linewidth=2.2, color=sty["color"], label="_nolegend_")
            legend_handles.append(mpatches.Patch(
                facecolor="none", edgecolor=sty["color"], linewidth=2.2, label=name))
    plt.title(f"Histogram: {var}")
    plt.xlabel(var)
    plt.ylabel("Density")
    plt.legend(handles=legend_handles)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
 
 
# ========== 日内剖面：四图例 ==========
 
def mean_profile(df, var, is_weekend_value: int):
    g = df[df["is_weekend"] == is_weekend_value].groupby("t")[var].mean()
    return g.reindex(range(96)).to_numpy()
 
 
def plot_profile_4way(dfs: dict, var: str, is_weekend_value: int, out_path: Path):
    x = np.arange(96)
    tag = "weekend" if is_weekend_value == 1 else "weekday"
 
    plt.figure(dpi=250)
    for name, df in dfs.items():
        y = mean_profile(df, var, is_weekend_value)
        sty = MODEL_STYLES[name]
        # Real 保持原样；三个模型统一用实线
        plt.plot(x, y, label=name,
                 color=sty["color"], linewidth=sty["linewidth"], linestyle="-")
    plt.title(f"Mean profile ({tag}): {var}")
    plt.xlabel("t (15-min slot)")
    plt.ylabel(var)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
 
 
# ========== 日总量分布：四图例 ==========
 
def daily_totals(df):
    g = df.groupby("day_id", as_index=False)
    out = g.agg(load_total=("load", "sum"), lambda_total=("lambda", "sum"))
    tmp = df.copy()
    tmp["pl"] = tmp["price"] * tmp["load"]
    pl = tmp.groupby("day_id")["pl"].sum()
    ld = tmp.groupby("day_id")["load"].sum().replace(0, np.nan)
    out["price_weighted"] = out["day_id"].map(pl) / out["day_id"].map(ld)
    tmp["cost_step"] = tmp["price"] * tmp["load"] * DT_HOUR
    cost = tmp.groupby("day_id")["cost_step"].sum()
    out["day_cost"] = out["day_id"].map(cost)
    return out
 
 
def plot_daily_total_4way(dfs: dict, col: str, out_path: Path, bins=60):
    arrays = {name: daily_totals(df)[col].to_numpy() for name, df in dfs.items()}
    lo = np.nanmin([a.min() for a in arrays.values()])
    hi = np.nanmax([a.max() for a in arrays.values()])
 
    plt.figure(dpi=250)
    legend_handles = []
    for name, arr in arrays.items():
        sty = MODEL_STYLES[name]
        if name == "Real":
            plt.hist(arr, bins=bins, range=(lo, hi), density=True,
                     alpha=0.25, color=sty["color"], label="_nolegend_")
            legend_handles.append(mpatches.Patch(
                facecolor=sty["color"], alpha=0.25, edgecolor=sty["color"], label=name))
        else:
            plt.hist(arr, bins=bins, range=(lo, hi), density=True,
                     histtype="step", linewidth=2.2, color=sty["color"], label="_nolegend_")
            legend_handles.append(mpatches.Patch(
                facecolor="none", edgecolor=sty["color"], linewidth=2.2, label=name))
    plt.title(f"Daily distribution: {col}")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.legend(handles=legend_handles)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
 
 
# ========== 主入口 ==========
 
def make_all(real_path, cvae_csv, gan_csv, diff_csv, out_dir, tag):
    """
    real_path : 真实数据 CSV / NPZ
    cvae_csv  : CVAE-RunB 的 generated_days.csv
    gan_csv   : GAN-tuned1 的 generated_days.csv
    diff_csv  : Diffusion Model 的 generated_days.csv
    out_dir   : 输出目录
    tag       : 文件名后缀，如 "mainB" / "stressA"
    """
    real = load_real_as_df(Path(real_path))
    cvae = load_gen_as_df(Path(cvae_csv))
    gan  = load_gen_as_df(Path(gan_csv))
    diff = load_gen_as_df(Path(diff_csv))
    out_dir = Path(out_dir)
 
    # 用 OrderedDict 风格保证图例顺序：Real 在最前
    dfs = {
        "Real":             real,
        "CVAE-RunB":        cvae,
        "GAN-tuned1":       gan,
        "Diffusion Model":  diff,
    }
 
    # ---- Fig2-6：三变量直方图 ----
    for v in TARGET_COLS:
        plot_hist_4way(dfs, v, out_dir / f"Fig2-6_{tag}_hist_{v}.png")
 
    # ---- Fig2-7：weekday/weekend 日内剖面 ----
    for v in ["load", "lambda"]:
        plot_profile_4way(dfs, v, 0, out_dir / f"Fig2-7_{tag}_profile_{v}_weekday.png")
        plot_profile_4way(dfs, v, 1, out_dir / f"Fig2-7_{tag}_profile_{v}_weekend.png")
 
    # ---- Fig2-8：日总量分布 ----
    for col in ["load_total", "lambda_total", "price_weighted"]:
        plot_daily_total_4way(dfs, col, out_dir / f"Fig2-8_{tag}_daily_{col}.png")
 
    print(f"[DONE] all figures saved to {out_dir}/  (tag={tag})")


# ========================================================================
# ====== 你只需要改下面的路径，然后用命令行选场景 ======
# ========================================================================
if __name__ == "__main__":
    import argparse

    # ---------- 公共：真实数据路径 ----------
    REAL_PATH = r"/opt/data/private/KYC-260209LoadDataCode/dataset_China_all.csv"

    # ---------- 各场景下三个模型的 generated_days.csv ----------
    SCENE_CFG = {
        "mainB": {
            "cvae": r"/opt/data/private/KYC-260209LoadDataCode/260216-CVAEandGAN/runs/china_cvae/20260216_150617_cvae_beta0.001_z128_hd1024-512-256_pmproject_levels_zsposterior_zsc1_extload_sum_top0.15_w10/gen_20260216_150637_mainB/generated_days.csv",
            "gan":  r"/opt/data/private/KYC-260209LoadDataCode/260216-CVAEandGAN/runs/china_gan/20260224_111733__cganWGP__z=256__g=512__d=512__bs=64__ep=500__nc=3__gp=5.0__note=china_gan_tuned1/gen_20260224_113018_mainB/generated_days.csv",
            "diff": r"/opt/data/private/KYC-260209LoadDataCode/260216-CVAEandGAN/runs/diffusion_model/generated_mainB.csv",
            "out":  "./paper_figs_3model_mainB",
        },
        "stressA": {
            "cvae": r"/opt/data/private/KYC-260209LoadDataCode/260216-CVAEandGAN/runs/china_cvae/20260216_150617_cvae_beta0.001_z128_hd1024-512-256_pmproject_levels_zsposterior_zsc1_extload_sum_top0.15_w10/gen_20260216_151209_stressA/generated_days.csv",
            "gan":  r"/opt/data/private/KYC-260209LoadDataCode/260216-CVAEandGAN/runs/china_gan/20260224_111733__cganWGP__z=256__g=512__d=512__bs=64__ep=500__nc=3__gp=5.0__note=china_gan_tuned1/gen_20260224_113154_stressA/generated_days.csv",
            "diff": r"/opt/data/private/KYC-260209LoadDataCode/260216-CVAEandGAN/runs/diffusion_model/generated_stressA.csv",
            "out":  "./paper_figs_3model_stressA",
        },
    }

    # ---------- 命令行参数 ----------
    parser = argparse.ArgumentParser(
        description="Real vs 3-model comparison plots (4 legends)")
    parser.add_argument(
        "--scene", type=str, required=True,
        choices=list(SCENE_CFG.keys()) + ["all"],
        help="选择场景：mainB / stressA / all（两个都跑）")
    args = parser.parse_args()

    scenes_to_run = list(SCENE_CFG.keys()) if args.scene == "all" else [args.scene]

    for sc in scenes_to_run:
        cfg = SCENE_CFG[sc]
        print(f"\n{'='*50}")
        print(f"  Running scene: {sc}")
        print(f"{'='*50}")
        make_all(REAL_PATH, cfg["cvae"], cfg["gan"], cfg["diff"],
                 out_dir=cfg["out"], tag=sc)