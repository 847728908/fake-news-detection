import matplotlib.pyplot as plt
import numpy as np
# -------- 饼状图部分 --------
# 数据集划分
sizes = [100, 20, 20]
labels = ['Train', 'Validation', 'Test']
colors = ['#90CAF9', '#FFD1DC', '#B39DDB']  # 淡蓝、淡粉、淡紫

plt.figure(figsize=(6, 6))
plt.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    textprops={'fontsize': 12}
)
plt.title('Dataset Split', fontsize=16)
plt.tight_layout()
plt.savefig('dataset_split_pie.png', dpi=300)
plt.show()