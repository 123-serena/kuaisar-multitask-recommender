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

        
model_name = "PLE_multilayer"
weighting = "uncertainty"
print(f"=== 训练 {model_name} + {weighting} + click task focal loss ===")
print("n_cat_feature:", processor.n_cat_feature)
print("每列实际最大编码id:", processor.train_df[processor.cat_features].max().to_dict())
model = PLEMultiLayerModel(num_categorical_features = processor.n_cat_feature,  n_num_features = processor.n_num_feature, embedding_dim=8).to(device)
if weighting == "uncertainty":
    loss_module = Lossmodule().to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(loss_module.parameters()), lr=1e-3
    )
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

final_results = {}
best_mean_auc = 0
patience = 0
best_results = [0,0,0,float("inf"),float("inf")]
best_click_auc = 0
click_patience = 0
click_frozen = False
best_click_state = None
BALANCE_LOSS_COEF = 0.05
CONFIG_TAG = "nosy1.5"  # 跟你现在save模型/json用的后缀保持一致，方便对应

fixed_diag_batch = get_fixed_eval_batch(ed, n_samples=8192, seed=2026) 
for epoch in range(100):
    model.train()
    for batch in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs, gate_probs_log = model(batch["cat"], batch["num"], return_gate_probs=True)
        balance_loss = total_gate_balance_loss(gate_probs_log)

        if weighting == "uncertainty":
            total_loss = loss_module(outputs, batch) + BALANCE_LOSS_COEF * balance_loss
        else:
            losses = compute_task_losses(outputs, batch)
            total_loss = sum(losses.values()) + BALANCE_LOSS_COEF * balance_loss

       

        optimizer.zero_grad()
        total_loss.backward()
        

        optimizer.step()

        
    if epoch % 2 == 0:
        model.eval()
        
        results = evaluate(model, el, device=device)
        
        diag_result = diagnose(model, fixed_diag_batch, TASK_TYPES,
                                expert_prefix="shared_experts", gate_prefix="shared_gates")
        for task, info in diag_result["per_task"].items():
            print(f"[诊断][{task}] dominant_expert={info['dominant_expert']}  "
                f"零梯度专家数={info['n_zero_experts']}")
        if diag_result["gate_grad_norm"]:
            print(f"[诊断][shared_gates] grad_norm={diag_result['gate_grad_norm']}")
        save_diagnosis(diag_result, config_tag=f"{CONFIG_TAG}_e{epoch}")
        with torch.no_grad():
            _, eval_gate_probs = model(fixed_diag_batch["cat"].to(device),
                                            fixed_diag_batch["num"].to(device),
                                            return_gate_probs=True)

            # 四个任务各自的task_gate，分开存，一眼看出谁均衡了、谁还没
            for name in TASK_TYPES.keys():
                gate_probs_list = eval_gate_probs["task_gates"][name]
                results[f"{name}_gate_balance_loss"] = sum(
                    load_balance_loss(p).item() for p in gate_probs_list
                ) / len(gate_probs_list)

            # shared_gates单独一份(层间路由那个独立的gate)
            shared_list = eval_gate_probs["shared_gates"]
            if shared_list:
                results["shared_gate_balance_loss"] = sum(
                    load_balance_loss(p).item() for p in shared_list
                ) / len(shared_list)

            results["gate_balance_loss_overall"] = total_gate_balance_loss(eval_gate_probs).item()

            print(f"[gate均衡] click={results['click_gate_balance_loss']:.3f}  "
                    f"like={results['like_gate_balance_loss']:.3f}  "
                    f"follow={results['follow_gate_balance_loss']:.3f}  "
                    f"watch_ratio={results['watch_ratio_gate_balance_loss']:.3f}  "
                    f"shared={results.get('shared_gate_balance_loss', float('nan')):.3f}")
        # if not click_frozen:
        #     if results["click_AUC"] > best_click_auc:
        #         best_click_auc = results["click_AUC"]
        #         click_patience = 0
        #         best_click_state = {
        #             "task_experts": {k: v.clone() for k, v in model.task_experts["click"].state_dict().items()},
        #             "task_gates": {k: v.clone() for k, v in model.task_gates["click"].state_dict().items()},
        #             "tower": {k: v.clone() for k, v in model.cattask_towers[0].state_dict().items()},
        #         }
        #     else:
        #         click_patience += 1

        #     if click_patience >= 3:
        #         # 回滚到click自己最好的那次快照
        #         model.task_experts["click"].load_state_dict(best_click_state["task_experts"])
        #         model.task_gates["click"].load_state_dict(best_click_state["task_gates"])
        #         model.cattask_towers[0].load_state_dict(best_click_state["tower"])
        #         # 冻结,之后不再更新这几块参数
        #         for p in model.task_experts["click"].parameters():
        #             p.requires_grad = False
        #         for p in model.task_gates["click"].parameters():
        #             p.requires_grad = False
        #         for p in model.cattask_towers[0].parameters():
        #             p.requires_grad = False
        #         click_frozen = True
        #         print(f"click在epoch{epoch}早停,已回滚到它自己的最优状态并冻结,like/follow/watch_ratio继续训练")
        final_results[epoch] = results
        ccur_result = [results["click_AUC"], results["like_AUC"], results["follow_AUC"], results["watch_ratio_MAE"], results["watch_ratio_RMSE"]]
        



        if better(ccur_result,best_results):
            best_results = ccur_result
            patience = 0
            torch.save(model.state_dict(), f"best_model_负载均衡损失+nosy1.5.pt")
        else:
            patience += 1
        print(f"Epoch {epoch}: click_AUC: {results['click_AUC']:.4f}, \nlike_AUC: {results['like_AUC']:.4f}, \nfollow_AUC: {results['follow_AUC']:.4f}, \nwatch_ratio_MAE: {results['watch_ratio_MAE']:.4f}, \nwatch_ratio_RMSE: {results['watch_ratio_RMSE']:.4f}")

        
        if patience >= 3:
            print(f"早停于 epoch {epoch}")
            break

with open(f"final_results_负载均衡损失+nosy1.5.json", "w") as f:
    json.dump(final_results, f)