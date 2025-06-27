import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Use English font (default is usually fine for English)
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False  # Ensure minus sign is shown correctly

# Evaluation results data
accuracy_data = {
    'Overall Accuracy': 0.7377,
    'Fake News Accuracy': 0.7778,
    'True News Accuracy': 0.6989
}
# 新增 F1-score 数据
f1_data = {
    'Fake News F1-score': 0.7447,
    'True News F1-score': 0.7303
}
sample_data = {
    'Fake News Samples': 180,
    'True News Samples': 186
}

# --- Visualization ---

# 设置 seaborn 风格，让图表更好看
sns.set_theme(style="whitegrid")

# 创建一个 1x2 的子图布局，指定图片大小
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Model Performance Visualization', fontsize=22, fontweight='bold')

# --- 1. Accuracy & F1-score Bar Chart ---

# 合并准确率和F1-score数据
labels = list(accuracy_data.keys()) + list(f1_data.keys())
values = list(accuracy_data.values()) + list(f1_data.values())

# 使用 seaborn 的调色板，颜色更多
colors = sns.color_palette('pastel', len(labels))

# 绘制条形图，并设置透明度
bars = ax1.bar(labels, values, color=colors, alpha=0.8, edgecolor='grey')

# 添加标题和标签
ax1.set_title('Model Evaluation Accuracy & F1-score', fontsize=16, fontweight='bold', pad=20)
ax1.set_ylabel('Score', fontsize=12)
ax1.set_ylim(0, 1.15) # 设置y轴范围，给顶部标签留出空间

# 在条形图上添加数值标签
for bar in bars:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.4f}', 
             ha='center', va='bottom', fontsize=11, fontweight='bold')

ax1.tick_params(axis='x', labelsize=11)
plt.setp(ax1.get_xticklabels(), rotation=20, ha='right')  # 旋转标签，右对齐


# --- 2. Sample Distribution Pie Chart ---

# 准备数据
sample_labels = list(sample_data.keys())
sample_sizes = list(sample_data.values())
# 'Paired' 调色板适合类别对比
sample_colors = sns.color_palette('Paired', len(sample_labels))
# 突出真新闻准确率较低的部分，在饼图中也突出显示
explode = (0, 0.05) 

# 绘制饼图
wedges, texts, autotexts = ax2.pie(sample_sizes, 
                                   explode=explode, 
                                   labels=sample_labels, 
                                   colors=sample_colors, 
                                   autopct=lambda p: '{:.1f}%\n({:.0f})'.format(p, p * sum(sample_sizes) / 100),
                                   shadow=True, 
                                   startangle=140,
                                   wedgeprops={'alpha': 0.9, 'edgecolor': 'grey'},
                                   textprops={'fontsize': 12})

# 美化 autopct 文本
plt.setp(autotexts, size=11, weight="bold", color="white")

# 添加标题
ax2.set_title('Training Sample Distribution', fontsize=16, fontweight='bold', pad=20)
ax2.axis('equal')  # 保证饼图是圆的

# 调整整体布局，防止标签重叠
plt.tight_layout(rect=[0, 0, 1, 0.96]) # 调整布局为总标题留出空间

# 显示图片
plt.show()

# --- 保存图片 ---
# 如果需要保存图片，可以取消下面这行代码的注释
# fig.savefig('evaluation_results_en.png', dpi=300, bbox_inches='tight')
