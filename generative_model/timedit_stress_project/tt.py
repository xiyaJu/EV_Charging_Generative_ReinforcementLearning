import pandas as pd
import numpy as np
from timedit_stress.stress import StressScorer, StressConfig

def generate_daily_stress_scores(input_file='daily.csv', output_file='daily_stress_scores.csv'):
    """
    从 daily.csv 生成 daily_stress_scores.csv 文件

    Args:
        input_file: 输入的 daily.csv 文件路径
        output_file: 输出的 daily_stress_scores.csv 文件路径
    """
    try:
        # 1. 读取原始数据
        print(f"正在读取数据: {input_file}")
        df = pd.read_csv(input_file)

        print(f"数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")

        # 2. 检查必要的列是否存在
        required_columns = {'day_id', 'day_of_week', 'is_weekend', 'price', 'load', 'lambda', 't'}
        missing_columns = required_columns - set(df.columns)

        if missing_columns:
            print(f"错误: 数据中缺少以下必要列: {missing_columns}")
            print("请确保 daily.csv 包含以下列: day_id, day_of_week, is_weekend, price, load, lambda, t")
            return False

        # 3. 计算 step_in_day 列（StressScorer 需要的列）
        # 假设 t 是时间戳或时间索引，我们将其转换为一天中的时间步（0-95）
        # 如果 t 是连续的时间值，我们可以用 t 对 96 取模
        print("正在计算 step_in_day...")

        # 方法1: 如果 t 是连续的时间值（例如，从0开始的索引）
        if 't' in df.columns:
            # 假设每天有96个时间步，t 对 96 取模得到 step_in_day
            df['step_in_day'] = df['t'] % 96

            # 检查 step_in_day 的范围
            print(f"step_in_day 范围: {df['step_in_day'].min()} 到 {df['step_in_day'].max()}")

            # 如果 step_in_day 的范围不是0-95，可能需要调整
            if df['step_in_day'].max() > 95 or df['step_in_day'].min() < 0:
                print("警告: step_in_day 的范围超出预期（0-95），可能需要调整计算方法")

        # 4. 验证数据
        print(f"数据验证:")
        print(f"  - 总行数: {len(df)}")
        print(f"  - 总天数: {df['day_id'].nunique()}")
        print(f"  - 每天的时间步数: {df.groupby('day_id').size().mean():.1f}")

        # 5. 配置 StressScorer
        print("\n正在配置 StressScorer...")
        config = StressConfig(
            steps_per_day=96,  # 每天96个时间步
            min_group_size=6,
            active_threshold=1.5,
            deviation_clip=8.0
        )

        # 6. 创建并训练 StressScorer
        print("正在训练 StressScorer...")
        scorer = StressScorer(config=config)

        # 使用 fit_transform 方法，它会返回包含 stress_score 的完整数据集
        scored_df = scorer.fit_transform(df)

        # 7. 提取每天的 stress_score
        print("提取每日 stress_score...")
        # 每天的 stress_score 是相同的，所以我们可以按 day_id 分组并取第一条记录
        daily_scores = scored_df.groupby('day_id').first()[['stress_score']].reset_index()

        # 8. 保存到 CSV 文件
        print(f"正在保存结果到: {output_file}")
        daily_scores.to_csv(output_file, index=False)

        # 9. 打印统计信息
        print("\n生成的 daily_stress_scores.csv 统计信息:")
        print(f"总天数: {len(daily_scores)}")
        print(f"stress_score 范围: [{daily_scores['stress_score'].min():.4f}, {daily_scores['stress_score'].max():.4f}]")
        print(f"平均 stress_score: {daily_scores['stress_score'].mean():.4f}")

        # 10. 显示前几行数据
        print("\n前5天的 stress_score:")
        print(daily_scores.head())

        # 11. 保存包含所有列的 scored_df（可选）
        scored_output_file = 'scored_' + input_file
        scored_df.to_csv(scored_output_file, index=False)
        print(f"\n包含所有列的 scored 数据已保存到: {scored_output_file}")

        print(f"\n成功生成 {output_file}!")
        return True

    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
        return False
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def generate_with_custom_config(input_file='daily.csv', output_file='daily_stress_scores.csv'):
    """
    使用自定义配置生成 stress scores
    """
    try:
        # 读取数据
        df = pd.read_csv(input_file)

        # 计算 step_in_day
        if 't' in df.columns:
            df['step_in_day'] = df['t'] % 96

        # 自定义配置
        custom_config = StressConfig(
            steps_per_day=96,
            min_group_size=4,  # 更小的组大小，适用于数据量较小的情况
            active_threshold=1.2,  # 调整活跃阈值
            deviation_clip=6.0,  # 调整偏差裁剪
            variable_weights={
                "price": 0.4,    # 调整权重
                "load": 0.3,
                "lambda": 0.2,
                "joint": 0.1
            }
        )

        scorer = StressScorer(config=custom_config)
        scored_df = scorer.fit_transform(df)
        daily_scores = scored_df.groupby('day_id').first()[['stress_score']].reset_index()
        daily_scores.to_csv(output_file, index=False)

        print(f"使用自定义配置生成 {output_file} 成功!")
        return daily_scores

    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def analyze_daily_data(input_file='daily.csv'):
    """
    分析 daily.csv 数据，帮助理解数据结构
    """
    try:
        df = pd.read_csv(input_file)

        print("="*60)
        print("数据结构分析")
        print("="*60)

        print(f"\n数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")

        print("\n前5行数据:")
        print(df.head())

        print("\n基本统计信息:")
        print(df.describe())

        print("\n按 day_id 分组的统计:")
        daily_stats = df.groupby('day_id').agg({
            'price': ['count', 'mean', 'std'],
            'load': ['count', 'mean', 'std'],
            'lambda': ['count', 'mean', 'std']
        }).round(4)
        print(daily_stats.head(10))

        print(f"\n总天数: {df['day_id'].nunique()}")
        print(f"每天的时间步数: {df.groupby('day_id').size().mean():.1f}")

        # 检查 step_in_day 的计算
        if 't' in df.columns:
            df['step_in_day'] = df['t'] % 96
            print(f"\nstep_in_day 统计:")
            print(f"  范围: {df['step_in_day'].min()} 到 {df['step_in_day'].max()}")
            print(f"  唯一值数量: {df['step_in_day'].nunique()}")

            # 检查每天的 step_in_day 是否完整
            steps_per_day = df.groupby('day_id')['step_in_day'].nunique()
            print(f"\n每天的唯一时间步数:")
            print(f"  平均: {steps_per_day.mean():.1f}")
            print(f"  最小: {steps_per_day.min()}")
            print(f"  最大: {steps_per_day.max()}")

            if steps_per_day.min() < 96:
                print("  警告: 某些天的时间步数不足96个")

        print("\n" + "="*60)

    except Exception as e:
        print(f"分析错误: {str(e)}")
        import traceback
        traceback.print_exc()

# 主程序
if __name__ == "__main__":
    # 首先分析数据
    print("正在分析数据...")
    analyze_daily_data('daily.csv')

    print("\n" + "="*60)
    print("开始生成 stress scores...")
    print("="*60 + "\n")

    # 方法1: 使用默认配置生成
    success = generate_daily_stress_scores('daily.csv', 'daily_stress_scores.csv')

    if success:
        print("\n" + "="*50)
        print("生成完成!")

        # 方法2: 如果需要自定义配置，可以取消下面的注释
        # print("\n使用自定义配置重新生成...")
        # custom_scores = generate_with_custom_config('daily.csv', 'daily_stress_scores_custom.csv')
    else:
        print("\n生成失败，请检查错误信息和数据格式。")
