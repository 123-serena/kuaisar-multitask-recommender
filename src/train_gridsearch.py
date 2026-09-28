import Kuaidataset
import Kuaiprocessor
from Kuaimodel import Lossmodule,PLEMultiLayerModel,PLEmodel,MMOE
from evaluation import evaluate, compute_task_losses
import torch
import os
from torch.utils.data import DataLoader,Subset
from tqdm import tqdm
import json
import numpy as np
import torch.nn.functional as F

import re
import time
from collections import defaultdict




# 1. 固定eval子集（保证跨run可比）
def get_fixed_eval_batch(eval_dataset, n_samples=4096, seed=2026):
    """用固定seed选一批固定的样本索引，而不是每次shuffle随机采样。
    只要eval_dataset本身没变，不同run里拿到的这批样本永远一样。"""
    g = torch.Generator().manual_seed(seed)
    n = len(eval_dataset)
    idx = torch.randperm(n, generator=g)[:min(n_samples, n)].tolist()
    subset = Subset(eval_dataset, idx)
    loader = DataLoader(subset, batch_size=len(idx), shuffle=False)
    return next(iter(loader))


# 2. 按"专家/gate编号"自动分组参数
def _group_key(name, prefix):
    """从参数名里把prefix后面的整数下标依次抠出来当分组key。
    比如 shared_experts.0.1.0.weight (prefix="shared_experts") -> (0, 1)  # (layer, expert)
        experts.2.0.bias           (prefix="experts")        -> (2,)    # (expert,)
    这样不用管n_layers/n_experts具体是多少，调架构参数时这个脚本不用改。"""
    rest = name[len(prefix):].lstrip(".")
    nums = re.findall(r"\d+", rest)
    return tuple(int(x) for x in nums[:2])  # 最多取前两级下标


def get_grouped_params(model, prefix):
    groups = defaultdict(list)
    for n, p in model.named_parameters():
        if n.startswith(prefix) and p.requires_grad:
            key = _group_key(n, prefix)
            groups[key].append((n, p))
    return groups


# 3. 核心诊断：每个任务单独backward，按组统计梯度范数
def diagnose(model, batch, task_types, expert_prefix="shared_experts", gate_prefix="shared_gates"):
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}
    cat_inputs, num_inputs = batch["cat"], batch["num"]

    expert_groups = get_grouped_params(model, expert_prefix)
    gate_groups = get_grouped_params(model, gate_prefix) if gate_prefix else {}
    task_names = list(task_types.keys())

    result = {"per_task": {}, "gate_grad_norm": {}}

    for name in task_names:
        model.zero_grad(set_to_none=True)
        outputs,_ = model(cat_inputs, num_inputs,return_gate_probs = False)
        pred, y = outputs[name], batch[name]
        loss = (F.binary_cross_entropy_with_logits(pred, y) if task_types[name] == "binary"
                else F.mse_loss(pred, y))
        loss.backward()

        expert_norms = {}
        for key, params in expert_groups.items():
            sq = sum((p.grad.detach() ** 2).sum().item() for _, p in params if p.grad is not None)
            expert_norms[str(key)] = sq ** 0.5
        total = sum(expert_norms.values()) or 1e-12
        expert_share = {k: v / total for k, v in expert_norms.items()}

        result["per_task"][name] = {
            "expert_grad_norm": expert_norms,
            "expert_grad_share": expert_share,      # 占比，一眼看出坍塌到哪个专家
            "dominant_expert": max(expert_norms, key=expert_norms.get) if expert_norms else None,
            "n_zero_experts": sum(1 for v in expert_norms.values() if v < 1e-8),  # 完全没梯度的专家数
        }

        # 四个任务各自的task_gate，分开存，一眼看出谁均衡了、谁还没
                    
       

    # gate梯度：所有任务loss合在一起算一次（看gate整体是不是饱和，不用拆任务）
    if gate_groups:
        model.zero_grad(set_to_none=True)
        outputs,_ = model(cat_inputs, num_inputs,return_gate_probs=False)
        total_loss = sum(
            (F.binary_cross_entropy_with_logits(outputs[n], batch[n]) if task_types[n] == "binary"
             else F.mse_loss(outputs[n], batch[n]))
            for n in task_names
        )
        total_loss.backward()
        for key, params in gate_groups.items():
            sq = sum((p.grad.detach() ** 2).sum().item() for _, p in params if p.grad is not None)
            result["gate_grad_norm"][str(key)] = sq ** 0.5

    model.zero_grad(set_to_none=True)
    return result


# 4. 存档：带时间戳+配置标签，方便攒着做对比表
def save_diagnosis(result, config_tag, out_dir="."):
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"{out_dir}/diag_{config_tag}_{ts}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump({"config_tag": config_tag, "timestamp": ts, "result": result}, f,
                   ensure_ascii=False, indent=2)
    print(f"已保存诊断结果: {fname}")
    return fname

def load_balance_loss(gate_probs, eps=1e-9):
    """Switch Transformer风格：num_experts * sum_i(f_i * P_i)，
    各专家使用越均匀这个值越接近1(最小值)，越集中越大。"""
    num_experts = gate_probs.shape[-1]
    with torch.no_grad():
        top1 = gate_probs.argmax(dim=-1)
        f = torch.stack([(top1 == i).float().mean() for i in range(num_experts)])
    P = gate_probs.mean(dim=0)
    return num_experts * (f * P).sum()


def total_gate_balance_loss(gate_probs_log):
    """把forward返回的gate_probs_log里，所有task_gates + shared_gates的
    负载均衡损失取平均，避免gate数量多导致这一项量级失控。"""
    total, count = 0.0, 0
    for probs_list in gate_probs_log["task_gates"].values():
        for probs in probs_list:
            total = total + load_balance_loss(probs)
            count += 1
    for probs in gate_probs_log["shared_gates"]:
        total = total + load_balance_loss(probs)
        count += 1
    return total / max(count, 1)

# if os.path.exists("processor.pt"):
#     processor = torch.load("processor.pt")
TASK_TYPES = {"click": "binary", "like": "binary", "follow": "binary", "watch_ratio": "regression"}
def better(cur, best):
    # 顺序: click_AUC, like_AUC, follow_AUC, watch_ratio_MAE, watch_ratio_RMSE
    higher_is_better = [True, True, True, False, False]
    votes = sum(
        1 for c, b, hib in zip(cur, best, higher_is_better)
        if (hib and c > b) or (not hib and c < b)
    )
    return votes >= 3

REBUILD = False  # 改过特征列表/processor.py/Kuaidataset.py之后，务必设成True重新生成一次

if REBUILD:
    print("[rebuild] 重新生成 processor / train_dataset / eval_dataset ...")
    processor = Kuaiprocessor.processor()
    td = Kuaidataset.KuaiSARTorchDataset(processor.train_df)
    ed = Kuaidataset.KuaiSARTorchDataset(processor.val_df)
    torch.save(processor, "processor.pt")
    torch.save(td, "train_dataset.pt")
    torch.save(ed, "eval_dataset.pt")
else:
    print("[cache] 直接加载缓存")
    processor = torch.load("processor.pt")
    td = torch.load("train_dataset.pt")
    ed = torch.load("eval_dataset.pt")
# processor = torch.load("processor.pt")
# td = Kuaidataset. KuaiSARTorchDataset(processor.train_df)
# ed = Kuaidataset. KuaiSARTorchDataset(processor.val_df)
# torch.save(td,"train_dataset.pt")
# torch.save(ed,"eval_dataset.pt")

el = DataLoader(ed, batch_size=8192, shuffle=False, num_workers=10,pin_memory=True)
dl = DataLoader(td, batch_size=8192, shuffle=True, num_workers=10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fixed_diag_batch = get_fixed_eval_batch(ed, n_samples=8192, seed=2026)
weighting = "uncertainty"

# ============================================================
# 把你原来"建模型->训练->eval/diag->存best/存final_results"这一整块
# 抠成一个函数，逻辑跟你原来train.py里一模一样，一个字没改，只是
# n_shared_experts / task_experts_cfg / config_tag 从外面传进来
# ============================================================
import sys

NUM_MACHINES = 4  # 你实际开几台实例就改成几
# 优先读环境变量MACHINE_ID，没有的话读命令行第一个参数，默认0
MACHINE_ID = int(os.environ.get("MACHINE_ID", sys.argv[1] if len(sys.argv) > 1 else 0))


def train_one_config(n_shared_experts, task_experts_cfg, config_tag, out_dir,
                      n_epochs=10, balance_loss_coef=0.02, gate_noise_std=0.8):
    print(f"\n{'='*70}\n[机器{MACHINE_ID}] 配置: {config_tag}  shared={n_shared_experts}  task_experts={task_experts_cfg}\n{'='*70}")

    os.makedirs(out_dir, exist_ok=True)
    result_path = f"{out_dir}/final_results_{config_tag}.json"

    model = PLEMultiLayerModel(
        num_categorical_features=processor.n_cat_feature,
        n_num_features=processor.n_num_feature,
        embedding_dim=8,
        n_task_experts=task_experts_cfg,
        n_shared_experts=n_shared_experts,
    ).to(device)
    model.gate_noise_std = gate_noise_std

    if weighting == "uncertainty":
        loss_module = Lossmodule().to(device)
        optimizer = torch.optim.Adam(list(model.parameters()) + list(loss_module.parameters()), lr=1e-3)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    final_results = {}
    best_results = [0, 0, 0, float("inf"), float("inf")]
    patience = 0

    for epoch in range(n_epochs):
        model.train()
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs, gate_probs_log = model(batch["cat"], batch["num"], return_gate_probs=True)
            balance_loss = total_gate_balance_loss(gate_probs_log)

            if weighting == "uncertainty":
                total_loss = loss_module(outputs, batch) + balance_loss_coef * balance_loss
            else:
                losses = compute_task_losses(outputs, batch)
                total_loss = sum(losses.values()) + balance_loss_coef * balance_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        if epoch % 2 == 0:
            model.eval()
            results = evaluate(model, el, device=device)

            diag_result = diagnose(model, fixed_diag_batch, TASK_TYPES,
                                    expert_prefix="shared_experts", gate_prefix="shared_gates")
            save_diagnosis(diag_result, config_tag=f"{config_tag}_e{epoch}", out_dir=out_dir)

            with torch.no_grad():
                _, eval_gate_probs = model(fixed_diag_batch["cat"].to(device),
                                            fixed_diag_batch["num"].to(device),
                                            return_gate_probs=True)
                for name in TASK_TYPES.keys():
                    gate_probs_list = eval_gate_probs["task_gates"][name]
                    results[f"{name}_gate_balance_loss"] = sum(
                        load_balance_loss(p).item() for p in gate_probs_list
                    ) / len(gate_probs_list)
                shared_list = eval_gate_probs["shared_gates"]
                if shared_list:
                    results["shared_gate_balance_loss"] = sum(
                        load_balance_loss(p).item() for p in shared_list
                    ) / len(shared_list)
                results["gate_balance_loss_overall"] = total_gate_balance_loss(eval_gate_probs).item()

            final_results[epoch] = results
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(final_results, f, ensure_ascii=False, indent=2)

            cur_result = [results["click_AUC"], results["like_AUC"], results["follow_AUC"],
                          results["watch_ratio_MAE"], results["watch_ratio_RMSE"]]

            if better(cur_result, best_results):
                best_results = cur_result
                patience = 0
                torch.save(model.state_dict(), f"{out_dir}/best_model_{config_tag}.pt")
            else:
                patience += 1

            print(f"[机器{MACHINE_ID}][{config_tag}] epoch{epoch}: click_AUC={results['click_AUC']:.4f}  "
                  f"like_AUC={results['like_AUC']:.4f}  follow_AUC={results['follow_AUC']:.4f}  "
                  f"watch_ratio_MAE={results['watch_ratio_MAE']:.4f}")

            if patience >= 3:
                print(f"[{config_tag}] 早停于 epoch {epoch}")
                break

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    return final_results





# def make_config(dim, value):
#     cfg = dict(BASELINE)
#     cfg[dim] = value
#     return cfg


def config_to_call(cfg):
    task_experts_cfg = {
        "click": cfg["click"], "like": cfg["like"],
        "follow": cfg["follow"], "watch_ratio": cfg["watch_ratio"],
    }
    tag = f"grid_s{cfg['shared']}_c{cfg['click']}_l{cfg['like']}_f{cfg['follow']}_w{cfg['watch_ratio']}"
    return cfg["shared"], task_experts_cfg, tag


if __name__ == "__main__":
    combined_task_experts = {"click": 2, "like": 1, "follow": 2, "watch_ratio": 2}
    combined_tag = "grid_s4_c2_l1_f2_w2"

    train_one_config(
        n_shared_experts=4,
        task_experts_cfg=combined_task_experts,
        config_tag=combined_tag,
        n_epochs=10,
        out_dir="grid_search_results",     # 存到同一个目录,方便结果直接并进grid_all.json
    )