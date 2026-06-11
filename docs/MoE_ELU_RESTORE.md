# MoE ELU 版還原指南（現行架構，對應 checkpoint: DiabloBalanceGrasp_moe_0.2）

## 目標檔案

```
rl_games-1.6.1/rl_games/algos_torch/network_builder.py
```

---

## 需要修改的三個位置

### 位置 1：Expert / Gate 建構（~line 277）

找到這段（陽春版）：
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

**換成（ELU 版）：**
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

---

### 位置 2：Weight init（~line 319）

找到這段（陽春版）：
```python
if self.moe_num_actors > 1:
    for expert in self.mu_experts:
        mu_init(expert.weight)
```

**換成（ELU 版）：**
```python
if self.moe_num_actors > 1:
    for expert in self.mu_experts:
        mu_init(expert[-1].weight)
```

---

### 位置 3：`_compute_mu` 方法（~line 479）

**不需要修改**，兩個版本相同：
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

## ELU 版架構總覽

```
Shared Backbone  obs(72) → 512 → 256 → 128  (ELU，來自 yaml units)

Expert heads × N
  Linear(128 → 64) → ELU → Linear(64 → 9)
  expert_hidden = out_size // 2 = 64

Gate
  Linear(128 → 32) → ELU → Linear(32 → N) → Softmax
  gate_hidden = max(out_size // 4, N*4) = max(32, 8) = 32  (N=2 時)

Forward
  expert_mus   = stack([expert_i(backbone_out) for i in N])  # [B, N, 9]
  gate_weights = softmax(gate(backbone_out))                  # [B, N]
  mu = sum(expert_mus * gate_weights.unsqueeze(-1), dim=1)   # [B, 9]
```

---

## Checkpoint key 名稱（ELU 版）

```
a2c_network.mu_experts.0.0.weight   # expert 0, Sequential[0] = Linear(128→64)
a2c_network.mu_experts.0.0.bias
a2c_network.mu_experts.0.2.weight   # expert 0, Sequential[2] = Linear(64→9)
a2c_network.mu_experts.0.2.bias
a2c_network.mu_experts.1.0.weight   # expert 1
a2c_network.mu_experts.1.0.bias
a2c_network.mu_experts.1.2.weight
a2c_network.mu_experts.1.2.bias
a2c_network.mu_gate.0.weight        # gate, Sequential[0] = Linear(128→32)
a2c_network.mu_gate.0.bias
a2c_network.mu_gate.2.weight        # gate, Sequential[2] = Linear(32→2)
a2c_network.mu_gate.2.bias
```

---

## 對應 Checkpoint

```
runs/DiabloBalanceGrasp_moe_0.2_10-01-53-46/nn/DiabloBalanceGrasp_moe_0.2.pth
```

## 啟動確認

載入 checkpoint 時應看到：
```
[MoE] 2 experts | expert: 128→64→9 | gate: 128→32→2
```

---

## 注意

- ELU 版 checkpoint **無法**用陽春版架構載入（key 名稱不同）
- 陽春版 checkpoint key：`mu_experts.0.weight`（直接 Linear）
- 陽春版還原請見：`MoE_SIMPLE_RESTORE.md`
