import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from scipy.stats import gaussian_kde

# ====================== M4 专用设置 ======================
plt.rcParams["font.family"] = "Arial"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("使用设备:", device)

# ====================== 修复后的数据预处理 ======================
class TimeSeriesPreprocessor:
    def __init__(self, df):
        self.num_features = ["price", "pv", "load", "lambda", "t"]
        # 注意：day_id是日期标识，换成“周内第几天”的周期特征（否则模型学错）
        self.cat_features = ["day_of_week", "is_weekend"]  # 移除day_id，用day_of_week代替周期
        
        self.raw_num = df[self.num_features].values

        # 数值特征分特征归一化（解决不同尺度问题）
        self.num_scalers = {}
        for col in self.num_features:
            scaler = MinMaxScaler(feature_range=(-1, 1))
            df[col] = scaler.fit_transform(df[[col]])
            self.num_scalers[col] = scaler

        # 类别特征OneHot编码
        self.cat_encoder = OneHotEncoder(sparse_output=False)
        cat_oh = self.cat_encoder.fit_transform(df[self.cat_features])

        self.all = np.hstack([df[self.num_features].values, cat_oh])
        self.feature_dim = self.all.shape[1]

        self.window = 96
        self.n_days = len(self.all) // self.window
        self.data = self.all[:self.n_days * self.window].reshape(-1, self.window, self.feature_dim)

    def inv(self, x):
        num = x[..., :5]
        # 分特征逆变换
        for i, col in enumerate(self.num_features):
            num[..., i] = self.num_scalers[col].inverse_transform(num[..., i].reshape(-1, 1)).reshape(num[..., i].shape)
        return num.reshape(-1, 96, 5)

# ====================== 增强版M4模型（加时序位置嵌入） ======================
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        emb = torch.exp(torch.arange(half, device=t.device) * (-np.log(10000) / half))
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

# 新增：时序位置嵌入（让模型知道时间步的顺序）
class PosEmbedding(nn.Module):
    def __init__(self, dim, max_len=96):
        super().__init__()
        self.emb = nn.Embedding(max_len, dim)
    def forward(self, x):
        pos = torch.arange(x.shape[1], device=x.device)  # 0-95时间步
        return self.emb(pos)[None, :, :]  # [1, 96, dim]

class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        # 新增：局部注意力（只关注相邻时间步，增强细粒度关联）
        self.local_attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.global_attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
    def forward(self, x):
        # 局部注意力：只看前后5个时间步
        batch, seq_len, dim = x.shape
        local_mask = torch.ones(seq_len, seq_len, device=x.device)
        local_mask = torch.triu(local_mask, diagonal=6) + torch.tril(local_mask, diagonal=-6)
        local_mask = local_mask.bool()
        x = x + self.local_attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=local_mask)[0]
        
        # 全局注意力：看整个序列
        x = x + self.global_attn(self.norm2(x), self.norm2(x), self.norm2(x))[0]
        x = x + self.mlp(self.norm3(x))
        return x

class ImprovedTimeDiT(nn.Module):
    def __init__(self, feat_dim, emb=96):
        super().__init__()
        self.proj_in = nn.Linear(feat_dim, emb)
        self.t_emb = TimeEmbedding(emb)
        self.t_mlp = nn.Sequential(nn.Linear(emb, emb*2), nn.GELU(), nn.Linear(emb*2, emb))
        self.pos_emb = PosEmbedding(emb)  # 新增时序位置嵌入
        self.blocks = nn.ModuleList([Block(emb) for _ in range(4)])  # 加1个Block增强容量
        self.norm = nn.LayerNorm(emb)
        self.proj_out = nn.Linear(emb, feat_dim)
    def forward(self, x, t):
        x = self.proj_in(x)
        x = x + self.pos_emb(x)  # 注入时序位置信息
        t = self.t_mlp(self.t_emb(t))
        x = x + t[:, None, :]
        for b in self.blocks:
            x = b(x)
        return self.proj_out(self.norm(x))

# ====================== 修复后扩散过程（余弦调度+充分采样） ======================
class Diffusion:
    def __init__(self, model, T=200):  # 增加扩散步数到200
        self.model = model
        self.T = T
        # 换成余弦噪声调度（对时序数据更友好）
        self.betas = self._cosine_beta_schedule(T)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]])

    def _cosine_beta_schedule(self, T):
        steps = torch.arange(T+1, device=device)/T
        alphas_cumprod = torch.cos((steps + 0.008)/1.008 * np.pi/2)**2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:]/alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.02)

    def add_noise(self, x, t):
        a = self.alphas_cumprod[t][:, None, None]
        noise = torch.randn_like(x)
        return torch.sqrt(a)*x + torch.sqrt(1-a)*noise, noise

    def loss(self, x):
        t = torch.randint(0, self.T, (x.shape[0],), device=device)
        xn, n = self.add_noise(x, t)
        return nn.functional.mse_loss(self.model(xn, t), n)

    def sample(self, n_samples, feat_dim):
        x = torch.randn(n_samples, 96, feat_dim, device=device)
        # 减少跳步（每2步采一次），充分去噪
        for i in reversed(range(self.T)):
            if i % 2 != 0: continue
            t = torch.tensor([i]*n_samples, device=device)
            with torch.no_grad():
                pred = self.model(x, t)
            alpha = self.alphas[i]
            alpha_prev = self.alphas_cumprod_prev[i]
            x = (x - torch.sqrt(1-alpha)*pred)/torch.sqrt(alpha)
            if i > 0:
                x = x * torch.sqrt(alpha_prev) + torch.sqrt(1-alpha_prev)*torch.randn_like(x)
        return x.cpu().numpy()

# ====================== 可视化（不变） ======================
def show_plots(raw, gen, feat_names):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0,0].plot(raw[0, :, 0], label="Real", linewidth=2)
    axes[0,0].plot(gen[0, :, 0], label="Generated", linestyle="--", linewidth=2)
    axes[0,0].set_title("Price (1 day)")
    axes[0,0].legend()

    r = raw[..., 0].flatten()
    g = gen[..., 0].flatten()
    sns.kdeplot(r, ax=axes[0,1], label="Real", color="blue")
    sns.kdeplot(g, ax=axes[0,1], label="Generated", color="red")
    axes[0,1].set_title("Price Distribution")
    axes[0,1].legend()

    axes[1,0].plot(raw[..., 2].mean(0), label="Real Load")
    axes[1,0].plot(gen[..., 2].mean(0), label="Generated Load", linestyle="--")
    axes[1,0].set_title("Daily Load Pattern")
    axes[1,0].legend()

    corr_r = pd.DataFrame(raw.reshape(-1,5), columns=feat_names).corr()
    sns.heatmap(corr_r, annot=True, cmap="coolwarm", ax=axes[1,1], vmin=-1, vmax=1)
    axes[1,1].set_title("Correlation (Real)")

    plt.tight_layout()
    plt.show()

# ====================== 主程序 ======================
if __name__ == "__main__":
    df = pd.read_csv("diffusion_model/scenario_train.csv")  # 你的数据路径
    prep = TimeSeriesPreprocessor(df)
    data = torch.tensor(prep.data, dtype=torch.float32).to(device)

    model = ImprovedTimeDiT(prep.feature_dim, emb=96).to(device)
    diff = Diffusion(model, T=200)
    opt = optim.AdamW(model.parameters(), lr=8e-5)  # 调小学习率，避免震荡
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)
    epochs = 80
    batch = 12  # 稍微减小batch，M4更稳
    loader = torch.utils.data.DataLoader(data, batch_size=batch, shuffle=True)

    print("开始训练...")
    model.train()
    for e in range(epochs):
        loss_sum = 0
        for x in loader:
            opt.zero_grad()
            loss = diff.loss(x)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
        avg_loss = loss_sum/len(loader)
        scheduler.step()
        print(f"Epoch {e+1:2d} | Loss: {avg_loss:.4f}")
        # 当loss降到0.1以下，说明模型开始学到规律了
        if avg_loss < 0.1 and e > 30:
            break

    print("生成新数据...")
    gen = diff.sample(10, prep.feature_dim)
    gen_real = prep.inv(gen)
    raw_real = prep.inv(prep.data)

    show_plots(raw_real, gen_real, prep.num_features)
    pd.DataFrame(gen_real.reshape(-1,5), columns=prep.num_features).to_csv("generated_M4_fixed.csv", index=False)