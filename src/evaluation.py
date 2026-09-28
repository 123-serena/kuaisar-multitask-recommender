import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from Kuaimodel import compute_task_losses
TASK_TYPES = {"click": "binary", "like": "binary", "follow": "binary", "watch_ratio": "regression"}


def safe_auc(y_true, y_score):
    """正负样本比例极端不均衡时（比如follow任务某个batch全是0），AUC算不出来，返回NaN而不是报错。"""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def compute_gauc(y_true, y_score, user_ids, min_pos=1, min_neg=1):
    """按用户分组分别算AUC，再按该用户的样本数加权平均，过滤掉组内只有单一类别的用户。"""
    df = pd.DataFrame({"user": user_ids, "y": y_true, "score": y_score})
    total_weight, weighted_auc = 0.0, 0.0
    for _, g in df.groupby("user"):
        pos, neg = (g["y"] == 1).sum(), (g["y"] == 0).sum()
        if pos < min_pos or neg < min_neg:
            continue
        auc = roc_auc_score(g["y"], g["score"])
        w = len(g)
        weighted_auc += auc * w
        total_weight += w
    return weighted_auc / total_weight if total_weight > 0 else float("nan")


def evaluate(model, data_loader, device, user_id_col_idx=0):
    """user_id_col_idx: user_id在你cat tensor里的列号，按你encoder的顺序(user_id/item_id/author_id/item_type)应该是0，
    如果你搭Dataset时列顺序不一样，记得改这个参数。"""
    model.eval()
    task_names = list(TASK_TYPES.keys())
    all_preds = {name: [] for name in task_names}
    all_labels = {name: [] for name in task_names}
    all_users = []
    n_batches = 0
    total_val_loss = 0
    click_losses = []
    like_losses = []
    follow_losses = []
    watch_ratio_losses = []
    click_loss = 0
    like_loss = 0
    follow_loss = 0
    watch_ratio_loss = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            cat = batch["cat"]
            num = batch["num"]
            outputs,_ = model(cat, num,return_gate_probs = False)   # {task: 预测值}，来自改好的forward
            val_loss = compute_task_losses(outputs, batch)
            click_loss += val_loss["click"].item()
            like_loss += val_loss["like"].item()
            follow_loss += val_loss["follow"].item()
            watch_ratio_loss += val_loss["watch_ratio"].item()
            total_val_loss += sum(val_loss.values()).item()
            n_batches += 1
            for name in task_names:
                pred = outputs[name]
                if TASK_TYPES[name] == "binary":
                    pred = torch.sigmoid(pred)   # 模型出的是raw logit，算AUC前要转成概率
                all_preds[name].append(pred.cpu().numpy())
                all_labels[name].append(batch[name].cpu().numpy())
            all_users.append(cat[:, user_id_col_idx].cpu().numpy())

    preds = {k: np.concatenate(v) for k, v in all_preds.items()}
    labels = {k: np.concatenate(v) for k, v in all_labels.items()}
    users = np.concatenate(all_users)
    
    results = {}
    for task in ["click", "like", "follow"]:
        results[f"{task}_AUC"] = safe_auc(labels[task], preds[task])
        results[f"{task}_GAUC"] = compute_gauc(labels[task], preds[task], users)
    results["watch_ratio_MAE"] = float(np.mean(np.abs(labels["watch_ratio"] - preds["watch_ratio"])))
    results["watch_ratio_RMSE"] = float(np.sqrt(np.mean((labels["watch_ratio"] - preds["watch_ratio"]) ** 2)))
    results["click_loss"] = click_loss / n_batches
    results["like_loss"] = like_loss / n_batches
    results["follow_loss"] = follow_loss / n_batches
    results["watch_ratio_loss"] = watch_ratio_loss / n_batches
    results["total_loss"] = total_val_loss / n_batches
    return results