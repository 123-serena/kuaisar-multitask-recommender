from torch import nn
from torch.nn import functional as F
import torch    
import torch.nn.functional as F

TASK_TYPES = {"click": "binary", "like": "binary", "follow": "binary", "watch_ratio": "regression"}
# numeric_cols 顺序对应的下标(从0开始),按任务分组
TASK_PRIVATE_NUM_IDX = {
    "click": [3, 7, 9, 10, 12, 15, 17, 19],
    # user_hist_click_rate_by_cat1, user_hist_click_cnt_by_author, user_hist_click_rate_by_hour,
    # user_hist_click_rate_overall, item_hist_click_rate, author_hist_click_rate,
    # user_recent20_click_rate, cross_user_interest_x_item_hotness
    "like": [4, 8, 13],
    # user_hist_like_rate_by_cat1, user_hist_like_cnt_by_author, item_hist_like_rate
    "follow": [5],
    # user_hist_follow_rate_by_cat1
    "watch_ratio": [6, 14, 16, 18],
    # user_hist_avg_watch_ratio_by_cat1, item_hist_avg_watch_ratio,
    # author_hist_avg_watch_ratio, user_recent10_avg_watch_ratio_by_cat1

}
SHARED_NUM_IDX = [0, 1, 2, 11]
# 剩下没出现在任何一个列表里的(log_duration_ms=0, hour_of_day=1, day_of_week=2,
# user_hist_interaction_cnt=11) 就是"公用"特征——它们照常走embedding+专家混合那条路,
# 不额外直连到某个tower。
# 在TASK_TYPES定义下面加一行，只给follow(和可选的like)设权重，
# 其他任务(click)不设，权重设为None时等价于原来的无权重BCE
POS_WEIGHTS = {
    "follow": None,   # 先试10，看效果再往上调（理论值≈207，别一步到位）
    "like": None,     # 先不动，等follow这边看出效果了再决定要不要也加
}



def compute_task_losses(outputs, batch):
    losses = {}
    for name, ttype in TASK_TYPES.items():
        pred = outputs[name]
        y = batch[name]
        if ttype == "binary":
            w = POS_WEIGHTS.get(name)
            pos_weight = torch.tensor([w], device=pred.device) if w is not None else None
            losses[name] = F.binary_cross_entropy_with_logits(pred, y, pos_weight=pos_weight)
        else:  # regression
            losses[name] = F.mse_loss(pred, y)
    return losses

class MMOE(nn.Module):
    def __init__(self,num_categorical_features, n_experts = 4, embedding_dim=8):
        super(MMOE, self).__init__()

        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(num_categories, embedding_dim) for num_categories in num_categorical_features]
        )
        self.numeric_embeddings = nn.ModuleList([nn.Linear(1, embedding_dim) for _ in range(3)])
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim * 7, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, embedding_dim)
            ) for _ in range(n_experts)
        ])

        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim * 7, 64),
                nn.ReLU(),
                nn.Linear(64, n_experts),
                nn.Softmax(dim=-1)
            ) for _ in range(4)
        ]) 
        self.cattask_towers = nn.ModuleList(
            [nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
             
        )for _ in range(3)]
        )

        self.numtask_tower = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
            
        )
    def forward(self, cat_inputs, num_inputs):
        cat_embedded = [emb(cat_inputs[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        num_embedded = [emb(num_inputs[:, i:i+1]) for i, emb in enumerate(self.numeric_embeddings)]
        embedded = torch.cat(cat_embedded + num_embedded, dim=1)

        expert_output = torch.stack([expert(embedded) for expert in self.experts], dim=1)  # (B, n_experts, embedding_dim)

        task_names = ["click", "like", "follow", "watch_ratio"]
        outputs = {}
        for i, name in enumerate(task_names):
            gate_weights = self.gates[i](embedded)               # 每个任务自己的gate
            task_repr = torch.sum(gate_weights.unsqueeze(-1) * expert_output, dim=1)  # (B, embedding_dim)
            if i < 3:
                outputs[name] = self.cattask_towers[i](task_repr).squeeze(-1)
            else:
                outputs[name] = self.numtask_tower(task_repr).squeeze(-1)
        return outputs

class Lossmodule(nn.Module):
    def __init__(self):
        super(Lossmodule, self).__init__()
        self.ks = nn.Parameter(torch.zeros(4))  # 初始化每个任务的log(sigma^2)为0

    def forward(self, outputs, batch):
        losses = compute_task_losses(outputs, batch)
        total_loss = sum([torch.exp(-k) * l + k for k, l in zip(self.ks, losses.values())])
        return total_loss

class PLEmodel(nn.Module):
    def __init__(self,num_categorical_features, n_experts = 4, embedding_dim=8):
        super(PLEmodel, self).__init__()

        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(num_categories, embedding_dim) for num_categories in num_categorical_features]
        )
        self.numeric_embeddings = nn.ModuleList([nn.Linear(1, embedding_dim) for _ in range(3)])
        
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim * 7, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, embedding_dim)
            ) for _ in range(2)
        ])

        self.click_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim * 7, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, embedding_dim)
            ) for _ in range(2)
        ])
        self.like_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim * 7, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, embedding_dim)
            ) for _ in range(2)
        ])

        self.follow_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim * 7, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, embedding_dim)
            ) for _ in range(2)
        ])

        self.watch_ratio_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim * 7, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, embedding_dim)
            ) for _ in range(2)
        ])

        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim * 7, 64),
                nn.ReLU(),
                nn.Linear(64, 4),
                nn.Softmax(dim=-1)
            ) for _ in range(4)
        ]) 
        self.cattask_towers = nn.ModuleList(
            [nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
             
        )for _ in range(3)]
        )

        self.numtask_tower = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()
            
        )
    def forward(self, cat_inputs, num_inputs):
        cat_embedded = [emb(cat_inputs[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        num_embedded = [emb(num_inputs[:, i:i+1]) for i, emb in enumerate(self.numeric_embeddings)]
        embedded = torch.cat(cat_embedded + num_embedded, dim=1)

        click_expert_output = torch.stack([expert(embedded) for expert in (self.click_experts + self.shared_experts)], dim=1)  # (B, n_experts, embedding_dim)
        like_expert_output = torch.stack([expert(embedded) for expert in (self.like_experts + self.shared_experts)], dim=1)
        follow_expert_output = torch.stack([expert(embedded) for expert in (self.follow_experts + self.shared_experts)], dim=1)
        watch_ratio_expert_output = torch.stack([expert(embedded) for expert in (self.watch_ratio_experts + self.shared_experts)], dim=1)

        expert_outputs = [click_expert_output, like_expert_output, follow_expert_output, watch_ratio_expert_output]
        task_names = ["click", "like", "follow", "watch_ratio"]
        outputs = {}
        for i, name in enumerate(task_names):
            gate_weights = self.gates[i](embedded)               # 每个任务自己的gate
            task_repr = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs[i], dim=1)  # (B, embedding_dim)
            if i < 3:
                outputs[name] = self.cattask_towers[i](task_repr).squeeze(-1)
            else:
                outputs[name] = self.numtask_tower(task_repr).squeeze(-1)
        return outputs

class PLEMultiLayerModel(nn.Module):
    def __init__(self, num_categorical_features, n_num_features, embedding_dim=8,
                 n_layers=2, n_task_experts=1, n_shared_experts=2):
        super(PLEMultiLayerModel, self).__init__()
        self.n_layers = n_layers
        self.task_names = ["click", "like", "follow", "watch_ratio"]
        self.gate_noise_std = 2.0
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(num_categories, embedding_dim) for num_categories in num_categorical_features]
        )
        self.numeric_embeddings = nn.ModuleList([nn.Linear(1, embedding_dim) for _ in range(n_num_features)])

        # 所有任务分支、共享分支都用同一份输入:全部类别特征 + 全部数值特征拼接
        self.input_dim = embedding_dim * (len(num_categorical_features) + n_num_features)
        
        def make_expert(in_dim):
            return nn.Sequential(
                nn.Linear(in_dim, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, embedding_dim),
            )

        def layer_in_dim(l):
            return self.input_dim if l == 0 else embedding_dim

       

        if isinstance(n_task_experts, dict):
            self.n_task_experts = {name: n_task_experts[name] for name in self.task_names}
        else:
            self.n_task_experts = {name: n_task_experts for name in self.task_names}
        self.n_shared_experts = n_shared_experts

        self.shared_experts = nn.ModuleList([
            nn.ModuleList([make_expert(layer_in_dim(l)) for _ in range(n_shared_experts)])
            for l in range(n_layers)
        ])

        self.task_experts = nn.ModuleDict({
            name: nn.ModuleList([
                nn.ModuleList([make_expert(layer_in_dim(l)) for _ in range(self.n_task_experts[name])])
                for l in range(n_layers)
            ])
            for name in self.task_names
        })

        self.task_gates = nn.ModuleDict({
            name: nn.ModuleList([
                nn.Sequential(
                    nn.Linear(layer_in_dim(l), 64), nn.ReLU(),
                    nn.Linear(64, self.n_task_experts[name] + n_shared_experts),
                    nn.Softmax(dim=-1),
                )
                for l in range(n_layers)
            ])
            for name in self.task_names
        })

        n_shared_gate_out = sum(self.n_task_experts.values()) + n_shared_experts
        self.shared_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(layer_in_dim(l), 64), nn.ReLU(),
                nn.Linear(64, n_shared_gate_out),
                nn.Softmax(dim=-1),
            )
            for l in range(n_layers - 1)
        ])

      
        self.cattask_towers = nn.ModuleList([
            nn.Sequential(nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Linear(64, 1))
            for _ in range(3)
        ])
        self.numtask_tower = nn.Sequential(
            nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Softplus()
        )
    def _gate_forward(self, gate_module, x):
        """gate_module是Sequential(Linear, ReLU, Linear, Softmax)，
        这里手动跳过最后的Softmax，在softmax前插入noisy gating的噪声。"""
        logits = gate_module[:-1](x)  # 除了最后一层Softmax之外都跑一遍，拿到softmax前的logits
        if self.training and self.gate_noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.gate_noise_std
        return F.softmax(logits, dim=-1)
    def forward(self, cat_inputs, num_inputs,return_gate_probs):
        cat_embedded = [emb(cat_inputs[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        num_embedded = [emb(num_inputs[:, i:i+1]) for i, emb in enumerate(self.numeric_embeddings)]
        embedded = torch.cat(cat_embedded + num_embedded, dim=1)

        # 所有任务分支、共享分支第0层输入都是同一个embedded
        task_inputs = {name: embedded for name in self.task_names}
        shared_input = embedded
        gate_probs_log = {"task_gates": {name: [] for name in self.task_names}, "shared_gates": []}
        for l in range(self.n_layers):
            is_last_layer = (l == self.n_layers - 1)

            task_expert_outputs = {
                name: [expert(task_inputs[name]) for expert in self.task_experts[name][l]]
                for name in self.task_names
            }
            shared_expert_outputs = [expert(shared_input) for expert in self.shared_experts[l]]
            new_task_inputs = {}
            for name in self.task_names:
                combined = torch.stack(task_expert_outputs[name] + shared_expert_outputs, dim=1)
                gate_weights = self._gate_forward(self.task_gates[name][l], task_inputs[name])  # 改了这行
                if return_gate_probs:
                    gate_probs_log["task_gates"][name].append(gate_weights)
                new_task_inputs[name] = torch.sum(gate_weights.unsqueeze(-1) * combined, dim=1)
        

            if not is_last_layer:
                all_experts = []
                for name in self.task_names:
                    all_experts += task_expert_outputs[name]
                all_experts += shared_expert_outputs
                combined_all = torch.stack(all_experts, dim=1)
                shared_gate_weights = self._gate_forward(self.shared_gates[l], shared_input)  # 改了这行
                if return_gate_probs:
                    gate_probs_log["shared_gates"].append(shared_gate_weights)
                shared_input = torch.sum(shared_gate_weights.unsqueeze(-1) * combined_all, dim=1)

            task_inputs = new_task_inputs

        outputs = {}
        for i, name in enumerate(self.task_names):
            if i < 3:
                outputs[name] = self.cattask_towers[i](task_inputs[name]).squeeze(-1)
            else:
                outputs[name] = self.numtask_tower(task_inputs[name]).squeeze(-1)
        return outputs,gate_probs_log