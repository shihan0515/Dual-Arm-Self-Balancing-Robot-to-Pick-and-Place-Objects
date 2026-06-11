# 評價與作圖流程

## 前置條件

```bash
pip install openpyxl matplotlib pandas tensorboard
```

---

## 訓練 / 評估模式切換

**每次跑評估前，兩個任務檔都要先切換到 EVAL MODE：**

| 檔案 | 位置 |
|------|------|
| `tasks/diablo_balance_grasp.py` | 搜尋 `_EVAL_MODE` |
| `tasks/diablo_balance_grasp_claude.py` | 搜尋 `_EVAL_MODE` |

```python
# 訓練時（安靜，不印成功率）
_EVAL_MODE = False

# 評估時（印出成功率、存到 extras）
_EVAL_MODE = True
```

> **訓練完記得改回 `False` 再繼續訓練**

---

## Step 0：確認 PYTHONPATH

每次開新終端機，跑評估前先確認：

```bash
export PYTHONPATH=/home/neo/Repositories/IsaacGymEnvs:$PYTHONPATH
```

（`shell_eval_ppo.sh` 和 `shell_eval_claude.sh` 已內建此行，不用手動設）

---

## Step 1：切換到評估模式

在 `diablo_balance_grasp.py` 和 `diablo_balance_grasp_claude.py` 中找到：

```python
_EVAL_MODE = False
```

改成：

```python
_EVAL_MODE = True
```

---

## Step 2：跑評估腳本

```bash
bash shell_eval_ppo.sh
```

跑完後產生資料夾，例如：
```
eval_ppo_20260609_130000/
  mug_seed42.log    mug_seed7.log    mug_seed123.log
  drill_seed42.log  drill_seed7.log  drill_seed123.log
  dumbbell_seed42.log  dumbbell_seed7.log  dumbbell_seed123.log
  summary.txt
```

設定說明（`shell_eval_ppo.sh` 內）：

| 參數 | 說明 |
|------|------|
| `CKPT` | checkpoint 路徑 |
| `NUM_ENVS` | 平行環境數（預設 500） |
| `SEEDS` | 評估用的隨機種子（預設 42 / 7 / 123） |
| `OBJECTS` | 評估物件（mug / drill / dumbbell） |

統計說明：每次跑會跳過前 `num_envs` 個 episode（warmup），從第二輪開始計數，避免冷啟動偏差。

---

## Step 3：解析 log → Excel

```bash
python parse_eval_to_excel.py eval_ppo_20260609_130000 eval_results.xlsx
```

**輸出：** `eval_results.xlsx`，包含 4 個 sheet：

| Sheet | 條件 | 難易 |
|-------|------|------|
| Phase2 Entry Rate | 成功拿起物體進入搬運 | 易 |
| Tight Contact Rate | XY < 5 cm，\|ΔZ\| < 2 cm | 中 |
| Partial Success Rate | XY < 5 cm，ΔZ ∈ [−2, +1] cm | 難 |
| Summary | 三個指標彙總 | — |

每個 sheet 欄位：

| 欄位 | 說明 |
|------|------|
| Object | mug / drill / dumbbell / Average |
| seed_42 / seed_7 / seed_123 | 各 seed 結果（%） |
| Mean (%) | 三個 seed 平均 |
| Std (%) | 三個 seed 標準差 |

---

## Step 4：畫評估結果圖

```bash
python plot_eval_results.py eval_results.xlsx thesis_fig
```

**輸出：**

| 檔案 | 說明 |
|------|------|
| `thesis_fig_by_metric.png` | 三個指標並排，X 軸為物件，附誤差棒 |
| `thesis_fig_by_object.png` | 三個物件並排，X 軸為指標（易 → 難） |

---

## Step 5：畫訓練獎勵曲線

### 5a. 單一模型（指定 label）

```bash
# PPO
python plot_reward_curves.py runs/<PPO_run_dir> thesis_ppo_reward --labels "PPO"

# MoE-PPO
python plot_reward_curves.py runs/<MoE_run_dir> thesis_moe_reward --labels "MoE-PPO"
```

### 5b. 兩個模型對比（PPO vs MoE-PPO）

```bash
python plot_reward_curves.py \
    runs/<PPO_run_dir> \
    runs/<MoE_run_dir> \
    thesis_compare_reward
# 第一個 run dir → PPO（藍色），第二個 → MoE-PPO（橘色）
```

**`--labels` 選項（可選）：**
```bash
python plot_reward_curves.py runs/<dir1> runs/<dir2> <prefix> --labels "PPO,MoE-PPO"
```

**輸出（每次執行都會產生三張）：**

| 檔案 | 說明 |
|------|------|
| `<prefix>_main.png` | 訓練總獎勵曲線（EMA 平滑，陰影為原始值） |
| `<prefix>_metrics.png` | Phase1/2/3 進入率 + Fall Rate |
| `<prefix>_grasp.png` | Grasp Reward 個別曲線 |
| `<prefix>_lift.png` | Lift Reward 個別曲線 |
| `<prefix>_height.png` | Height Reward 個別曲線 |
| `<prefix>_placement.png` | Placement Reward 個別曲線 |
| `<prefix>_transport.png` | Transport Reward 個別曲線 |
| `<prefix>_balance.png` | Balance Reward 個別曲線 |

> **注意：** 陰影是原始未平滑值，**不是** ±1 std；這是單一 run，不是多 seed 平均。

---

## Step 6：畫評估結果圖（指定模型名稱）

```bash
# PPO
python plot_eval_results.py eval_ppo_<時間戳>.xlsx thesis_ppo_fig "PPO"

# MoE-PPO
python plot_eval_results.py eval_moe_<時間戳>.xlsx thesis_moe_fig "MoE-PPO"
```

**輸出：**

| 檔案 | 說明 |
|------|------|
| `<prefix>_by_metric.png` | 三個 metric 並排，X 軸為物件 |
| `<prefix>_by_object.png` | 三個物件並排，X 軸為 metric（易→難） |

---

## 完整執行流程

```bash
# ① 切換評估模式（diablo_balance_grasp.py 搜尋 _EVAL_MODE 改 True）

# ② 跑評估（PPO + MoE-PPO 各自）
bash shell_eval_ppo.sh        # → eval_ppo_<ts>/
bash shell_eval_claude.sh     # → eval_claude_<ts>/
# 或一次跑兩個：bash run_all_eval.sh

# ③ 解析 log → Excel
python parse_eval_to_excel.py eval_ppo_<ts>    eval_ppo_<ts>.xlsx
python parse_eval_to_excel.py eval_claude_<ts> eval_moe_<ts>.xlsx

# ④ 畫評估長條圖
python plot_eval_results.py eval_ppo_<ts>.xlsx thesis_ppo_fig "PPO"
python plot_eval_results.py eval_moe_<ts>.xlsx thesis_moe_fig "MoE-PPO"

# ⑤ 畫訓練曲線（各獎勵分量 + 總獎勵 + phase metrics）
python plot_reward_curves.py runs/<PPO_run_dir> thesis_ppo_reward --labels "PPO"
python plot_reward_curves.py runs/<MoE_run_dir> thesis_moe_reward --labels "MoE-PPO"

# ⑥ 改回訓練模式（_EVAL_MODE 改 False）
```

---

## 現有 Checkpoint

| 模型 | 路徑 |
|------|------|
| PPO | `runs/DiabloBalanceGrasp_PPO_6000_09-21-50-29/nn/DiabloBalanceGrasp_PPO_6000.pth` |
| MoE-PPO | `runs/DiabloBalanceGrasp_moe_0.2_10-01-53-46/nn/DiabloBalanceGrasp_moe_0.2.pth` |

---

## 檔案清單

| 檔案 | 用途 |
|------|------|
| `shell_eval_ppo.sh` | PPO 評估腳本 |
| `shell_eval_claude.sh` | MoE-PPO 評估腳本 |
| `run_all_eval.sh` | PPO + MoE-PPO 一次跑完，自動 parse + plot |
| `parse_eval_to_excel.py` | log → Excel |
| `plot_eval_results.py` | 評估結果長條圖（支援第三參數指定模型名） |
| `plot_reward_curves.py` | 訓練曲線（總獎勵 + phase metrics + 各分量獨立圖） |
| `tasks/diablo_balance_grasp.py` | PPO / MoE-PPO 任務（搜尋 `_EVAL_MODE` 切換） |
