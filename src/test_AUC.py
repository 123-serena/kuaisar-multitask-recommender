import torch
import numpy as np
processor = torch.load("processor.pt")
data = processor.val_df
for col in ["click", "like", "follow"]:
    pos_rate = data[col].mean()
    print(f"{col}: 正样本比例 = {pos_rate:.4f}, 负样本比例 = {1-pos_rate:.4f}, 下限 = {-(pos_rate * np.log(pos_rate) + (1-pos_rate) * np.log(1-pos_rate)):.4f}")
