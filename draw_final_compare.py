import matplotlib.pyplot as plt
import numpy as np

# Metrics
metrics = ['Accuracy', 'F1-score']

# Untrained Model (假设为 data_final_combine)
untrained = [0.6284, (0.5641 + 0.6762) / 2]
# Trained Model (假设为 data_final_combine2)
trained = [0.8000, 0.7111]

x = np.arange(len(metrics))  # the label locations
width = 0.3  # the width of the bars

plt.figure(figsize=(8, 5))
bar1 = plt.bar(x - width/2, untrained, width, label='Untrained Model', color='#ADD8E6', alpha=0.8)  # 淡蓝色
bar2 = plt.bar(x + width/2, trained, width, label='Trained Model', color='#FFB6C1', alpha=0.8)    # 淡粉色

plt.xticks(x, metrics, fontsize=12)
plt.title('Model Results Comparison', fontsize=16)
plt.ylabel('Score', fontsize=13)
plt.ylim(0, 1)
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# 在柱子上方标注数值
for bars in [bar1, bar2]:
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f'{height:.3f}',
            ha='center',
            va='bottom',
            fontsize=11
        )

plt.tight_layout()
plt.savefig('result_compare_bar.png', dpi=300)
plt.show()

