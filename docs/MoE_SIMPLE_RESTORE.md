# MoE 陽春版還原指南

## 目標檔案

```
rl_games-1.6.1/rl_games/algos_torch/network_builder.py
```

---

## 需要修改的三個位置

### 位置 1：Expert / Gate 建構（~line 277）

找到這段（ELU 版）：
```python
if self.moe_num_actors > 1:
    expert_hidden = out_size // 2
    gate_hidden   = max(out_size // 4, self.moe_num_actors * 4)
    self.mu_experts = nn.ModuleList([
        nn.Sequential(
            nn.Linear(out_size, expert_hidden),
            nn.ELU(),
            nn.Linear(expert_hidden, actions_num),
        )
        for _ in range(self.moe_num_actors)
    ])
    self.mu_gate = nn.Sequential(
        nn.Linear(out_size, gate_hidden),
        nn.ELU(),
        nn.Linear(gate_hidden, self.moe_num_actors),
    )
    print(f"[MoE] {self.moe_num_actors} experts | "
          f"expert: {out_size}→{expert_hidden}→{actions_num} | "
          f"gate: {out_size}→{gate_hidden}→{self.moe_num_actors}")
```

**換成（陽春版）：**
```python
if self.moe_num_actors > 1:
    self.mu_experts = nn.ModuleList([
        nn.Linear(out_size, actions_num)
        for _ in range(self.moe_num_actors)
    ])
    self.mu_gate = nn.Linear(out_size, self.moe_num_actors)
    print(f"[MoE] {self.moe_num_actors} experts | "
          f"expert: {out_size}→{actions_num} | "
          f"gate: {out_size}→{self.moe_num_actors}")
```

---

### 位置 2：Weight init（~line 319）

找到這段（ELU 版）：
```python
if self.moe_num_actors > 1:
    for expert in self.mu_experts:
        mu_init(expert[-1].weight)
```

**換成（陽春版）：**
```python
if self.moe_num_actors > 1:
    for expert in self.mu_experts:
        mu_init(expert.weight)
```

---

### 位置 3：`_compute_mu` 方法（~line 479）

這段**不需要修改**，ELU 版和陽春版都一樣：
```python
def _compute_mu(self, actor_out, obs=None):
    if self.moe_num_actors > 1:
        expert_mus = torch.stack(
            [expert(actor_out) for expert in self.mu_experts],
            dim=1,
        )
        gate_weights = torch.softmax(self.mu_gate(actor_out), dim=-1)
        return torch.sum(expert_mus * gate_weights.unsqueeze(-1), dim=1)
    return self.mu(actor_out)
```

---

### 位置 4：params 讀取（~line 568）

這段**保留不動**，只是讀參數不影響架構：
```python
self.moe_num_actors    = params.get('moe_num_actors', 1)
self.gate_temperature  = params.get('gate_temperature', 0.3)
self.gate_phase_idx    = params.get('gate_phase_idx', -1)
```

---

## 陽春版架構總覽

```
Shared Backbone  obs(72) → 512 → 256 → 128  (ELU，來自 yaml units)

Expert heads × N
  Linear(128 → 9)           ← 直接一層，無隱藏層，無 ELU

Gate
  Linear(128 → N) → Softmax ← 直接一層，無隱藏層

Forward
  expert_mus  = stack([expert_i(backbone_out) for i in N])  # [B, N, 9]
  gate_weights = softmax(gate(backbone_out))                 # [B, N]
  mu = sum(expert_mus * gate_weights.unsqueeze(-1), dim=1)  # [B, 9]
```

---

## 啟動指令（N=2 experts）

```bash
python train.py task=DiabloBalanceGrasp train=DiabloBalanceGraspPPO \
    moe_num_actors=2 headless=True
```

啟動時應看到：
```
[MoE] 2 experts | expert: 128→9 | gate: 128→2
```

---

## 注意

- 陽春版訓練的 checkpoint **無法**用 ELU 版載入（key 名稱不同）
- ELU 版 checkpoint key：`mu_experts.0.0.weight`（Sequential[0]）
- 陽春版 checkpoint key：`mu_experts.0.weight`（直接 Linear）
- 目前 repo 的 ELU 版 checkpoint：`runs/DiabloBalanceGrasp_moe_0.2_10-01-53-46/`
