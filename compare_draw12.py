import matplotlib.pyplot as plt
import numpy as np

# draw.py 数据
accuracy_data1 = {
    'Overall Accuracy': 0.6311,
    'Fake News Accuracy': 0.9778,
    'True News Accuracy': 0.2957
}
f1_data1 = {
    'Fake News F1-score': 0.7228,
    'True News F1-score': 0.4490
}
sample_data1 = {
    'Fake News Samples': 180,
    'True News Samples': 186
}

# draw2.py 数据
accuracy_data2 = {
    'Overall Accuracy': 0.7377,
    'Fake News Accuracy': 0.7778,
    'True News Accuracy': 0.6989
}
f1_data2 = {
    'Fake News F1-score': 0.7447,
    'True News F1-score': 0.7303
}
sample_data2 = {
    'Fake News Samples': 180,
    'True News Samples': 186
}

# 可视化 Accuracy 对比
labels = list(accuracy_data1.keys())
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
ax.plot(x, list(accuracy_data1.values()), marker='o', label='original')
ax.plot(x, list(accuracy_data2.values()), marker='s', color='red', label='improved')

ax.set_ylabel('Accuracy')
ax.set_title('Accuracy comparison')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15)
ax.legend()
plt.tight_layout()
plt.show()

# 可视化 F1-score 对比
labels_f1 = list(f1_data1.keys())
x_f1 = np.arange(len(labels_f1))

fig, ax = plt.subplots()
ax.plot(x_f1, list(f1_data1.values()), marker='o', label='original')
ax.plot(x_f1, list(f1_data2.values()), marker='s', color='red', label='improved')

ax.set_ylabel('F1-score')
ax.set_title('F1-score comparison')
ax.set_xticks(x_f1)
ax.set_xticklabels(labels_f1, rotation=15)
ax.legend()
plt.tight_layout()
plt.show()
