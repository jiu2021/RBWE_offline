# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np

# # 读取CSV数据
# data = {
#     "Model": ["behavior policy", "Schaferct_model", "gmm_4"],
#     "MSE": [3.3630893, 2.638837, 3.4943924],
#     "Error Rate": [0.25288624, 0.30425996, 0.27412158],
#     "Over-estimated Rate": [0.49099693, 0.477991, 0.306818],
#     "Under-estimated Rate": [0.19904372, 0.17131838, 0.23570395],
#     'MSE std': [2.8816879, 2.6431928, 2.6504238],
#     'Error Rate std': [0.17556052, 0.20140631, 0.18276401],
#     'Over-estimated Rate std': [0.48865032, 0.6435503, 0.43556786],
#     'Under-estimated Rate std': [0.1738832, 0.15239434, 0.17602435]
# }
# df = pd.DataFrame(data)

# # 指标名称（不包括模型名称）
# metrics = df.columns[1:]
# n_metrics = len(metrics)
# n_models = len(df)

# # 设置柱状图参数
# fig, ax1 = plt.subplots(figsize=(12, 7), dpi=300)  # 提高分辨率
# ax2 = ax1.twinx()  # 创建次坐标轴

# bar_width = 0.2  # 柱宽
# indices = np.arange(n_metrics)  # x轴位置，每个指标为一簇

# # 定义颜色，每种方法一种颜色
# colors = {"behavior policy": "blue", "Schaferct_model": "green", "gmm_4": "red"}

# def get_color(model):
#     return colors.get(model, "gray")

# # 绘制 MSE（次坐标轴）
# for i, model in enumerate(df['Model']):
#     ax2.bar(indices[0] + i * bar_width, df['MSE'][i], bar_width,
#             label=model if i == 0 else "", color=get_color(model), alpha=0.7,
#             edgecolor='black' if model == "gmm_4" else 'none', linewidth=1.5)
# ax2.set_ylabel("MSE", color='black', fontsize=16)
# ax2.tick_params(axis='y', labelcolor='black', labelsize=14)

# # 绘制其他指标（主坐标轴）
# for i, model in enumerate(df['Model']):
#     for j, metric in enumerate(metrics[1:]):  # 跳过 MSE
#         ax1.bar(indices[j + 1] + i * bar_width, df[metric][i], bar_width,
#                 label=model if j == 0 else "", color=get_color(model), alpha=0.7,
#                 edgecolor='black' if model == "gmm_4" else 'none', linewidth=1.5)

# ax1.set_ylabel("Rate Values", color='black', fontsize=16)
# ax1.tick_params(axis='y', labelcolor='black', labelsize=14)
# ax1.set_xticks(indices + bar_width)
# ax1.set_xticklabels(metrics, rotation=0, ha='center', fontsize=16)
# ax1.grid(axis='y', linestyle='--', alpha=0.5)

# # 添加图例
# handles, labels = ax1.get_legend_handles_labels()
# ax1.legend(handles, labels, loc='upper right', fontsize=14)

# # plt.title("Comparison of Models on Different Metrics", fontsize=18, fontweight='bold')
# plt.tight_layout()
# # plt.show()
# plt.savefig("offline_metrics.png", dpi=300)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取CSV数据
data = {
    "Model": ["behavior policy", "Schaferct_model", "gmm_4"],
    "MSE": [3.3630893, 2.638837, 3.4943924],
    "Error Rate": [0.25288624, 0.30425996, 0.27412158],
    "Over-estimated Rate": [0.49099693, 0.477991, 0.306818],
    "Under-estimated Rate": [0.19904372, 0.17131838, 0.23570395],
    'MSE std': [2.8816879, 2.6431928, 2.6504238],
    'Error Rate std': [0.17556052, 0.20140631, 0.18276401],
    'Over-estimated Rate std': [0.48865032, 0.6435503, 0.43556786],
    'Under-estimated Rate std': [0.1738832, 0.15239434, 0.17602435]
}
df = pd.DataFrame(data)

# 指标名称（不包括模型名称）
metrics = df.columns[1:5]  # 只选择 MSE 和 Rate 指标
std_metrics = df.columns[5:]  # 对应的标准差列
n_metrics = len(metrics)
n_models = len(df)

# 设置柱状图参数
fig, ax1 = plt.subplots(figsize=(12, 7), dpi=300)  # 提高分辨率
ax2 = ax1.twinx()  # 创建次坐标轴

bar_width = 0.2  # 柱宽
indices = np.arange(n_metrics)  # x轴位置，每个指标为一簇

# 定义颜色，每种方法一种颜色
colors = {"behavior policy": "blue", "Schaferct_model": "green", "gmm_4": "red"}

def get_color(model):
    return colors.get(model, "gray")

# 绘制 MSE（次坐标轴）及其标准差
for i, model in enumerate(df['Model']):
    ax2.bar(indices[0] + i * bar_width, df['MSE'][i], bar_width,
            label=model if i == 0 else "", color=get_color(model), alpha=0.7,
            edgecolor='black' if model == "gmm_4" else 'none', linewidth=1.5)
    ax2.errorbar(indices[0] + i * bar_width, df['MSE'][i], yerr=df['MSE std'][i],
                 fmt='none', ecolor='black', capsize=5, capthick=1.5)

ax2.set_ylabel("MSE", color='black', fontsize=16)
ax2.tick_params(axis='y', labelcolor='black', labelsize=14)

# 绘制其他指标（主坐标轴）及其标准差
for i, model in enumerate(df['Model']):
    for j, metric in enumerate(metrics[1:]):  # 跳过 MSE
        ax1.bar(indices[j + 1] + i * bar_width, df[metric][i], bar_width,
                label=model if j == 0 else "", color=get_color(model), alpha=0.7,
                edgecolor='black' if model == "gmm_4" else 'none', linewidth=1.5)
        ax1.errorbar(indices[j + 1] + i * bar_width, df[metric][i], yerr=df[std_metrics[j + 1]][i],
                     fmt='none', ecolor='black', capsize=5, capthick=1.5)

ax1.set_ylabel("Rate Values", color='black', fontsize=16)
ax1.tick_params(axis='y', labelcolor='black', labelsize=14)
ax1.set_xticks(indices + bar_width)
ax1.set_xticklabels(metrics, rotation=0, ha='center', fontsize=16)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# 添加图例
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels, loc='upper right', fontsize=14)

# 保存图表
plt.tight_layout()
plt.savefig("offline_metrics_with_std.png", dpi=300)
