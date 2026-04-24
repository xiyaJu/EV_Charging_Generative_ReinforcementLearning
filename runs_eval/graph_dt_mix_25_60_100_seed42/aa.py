import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('/Users/xiya/Downloads/final/runs_eval/origin_model_70/ori70.per_replay.csv')

# 要计算平均值的列
target_columns = [
    'total_energy_charged',
    'total_energy_discharged', 
    'average_user_satisfaction',
    'total_reward'
]

# 检查列是否存在
for col in target_columns:
    if col not in df.columns:
        print(f"警告: 列 '{col}' 不存在于数据中")
        print(f"可用列: {list(df.columns)}")
        
# 计算每列的平均值
results = {}
for col in target_columns:
    if col in df.columns:
        # 计算平均值，并处理可能的缺失值
        mean_val = df[col].mean()
        # 也可以计算其他统计量
        std_val = df[col].std()
        min_val = df[col].min()
        max_val = df[col].max()
        
        results[col] = {
            '平均值': mean_val,
            '标准差': std_val,
            '最小值': min_val,
            '最大值': max_val
        }
        
        # 打印结果
        print(f"{col}:")
        print(f"  平均值: {mean_val:.4f}")
        print(f"  标准差: {std_val:.4f}")
        print(f"  最小值: {min_val:.4f}")
        print(f"  最大值: {max_val:.4f}")
        print(f"  样本数: {len(df[col].dropna())}")
        print()
