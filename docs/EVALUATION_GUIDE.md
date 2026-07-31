# 評價與作圖流程

## 前置條件

```bash
pip install openpyxl matplotlib pandas tensorboard
```

---

## 訓練 / 評估模式（2026-06-12 起改為 cfg 開關，不再改程式碼）

評估統計由 cfg 的 `task.env.eval_mode` 控制，**預設 `False`（訓練模式）**，
兩個評估腳本已內建 `task.env.eval_mode=True`，跑評估不需要再手動改任何檔案。

```bash
# 手動跑單趟評估時，自行加上這個 override：
python train.py task=DiabloBalanceGrasp test=True checkpoint=<ckpt> \
    task.env.eval_mode=True task.env.eval_object_name="mug" ...
```

| 開關位置 | 說明 |
|------|------|
| `cfg/task/DiabloBalanceGrasp.yaml` → `env.eval_mode` | 預設 False |
| `cfg/task/DiabloBalanceGraspClaude.yaml` → `env.eval_mode` | 預設 False |
| `env.success_hold_steps` | success 條件須連續 N 步（預設 1＝瞬間判定；腳本評估用 10） |
| `env.latch_hold_steps` | psr/tcr/p2r 統計 latch 須連續 N 步（同上） |

> **嚴格判定（2026-06-12 起）**：瞬間判定會把碰撞彈跳、擦過容差區的瞬間達標
> 記成達成，統計遠高於肉眼觀感。腳本現在統一用 HOLD=10（0.2 秒持續成立）。
> p2r 定義為「持續攜帶」：EEF 距 handle < 5cm ＋ 物體離開原位 > 10cm ＋ z > 0.45m
> 連續 10 步（高度差判定對傾斜搬運失效，姿態對齊判定對非標準抓姿失效，皆已棄用）。

> 舊版的 `_EVAL_MODE` 手動開關已移除。訓練指令不帶 override 就是訓練模式，
> 不會再有「忘記改回 False」或「忘記開 True 導致統計全空」的問題。
> 若腳本抓不到統計輸出，summary 會印 `WARNING`（不再靜默留空）。

---

## Step 0：確認 PYTHONPATH

每次開新終端機，跑評估前先確認：

```bash
export PYTHONPATH=${IGE_ROOT}:$PYTHONPATH
```

（`shell_eval_ppo.sh` 和 `shell_eval_claude.sh` 已內建此行，不用手動設）

---

## Step 1：跑評估腳本

```bash
bash shell_eval_ppo.sh        # PPO → eval_ppo_hold10_<時間戳>/

# MoE：bash shell_eval_claude.sh [checkpoint] [標籤] [expert數] [變體 simple|elu]
bash shell_eval_claude.sh runs/.../DiabloBalanceGrasp_moe_2021.pth moe2021 2 simple
bash shell_eval_claude.sh runs/.../DiabloBalanceGrasp_moe_t3.pth   moet3   3 simple
bash shell_eval_claude.sh runs/.../DiabloBalanceGrasp_moe_0.2.pth  moe0.2  2 elu
```

> MoE 變體自 2026-06-13 起由 cfg 控制（`moe_expert_hidden` / `moe_gate_hidden`，
> 0=simple 單層、64/32=ELU 兩層），**不再需要手動切換 network_builder.py**，
> 兩份 MoE_*_RESTORE.md 已過時。變體須與 checkpoint 訓練時一致，否則載入報
> state_dict key 錯誤。

跑完後產生資料夾，例如：
```
eval_ppo_20260612_154305/
  mug_seed42.log    mug_seed7.log    mug_seed123.log
  drill_seed42.log  drill_seed7.log  drill_seed123.log
  dumbbell_seed42.log  dumbbell_seed7.log  dumbbell_seed123.log
  summary.txt
```

`summary.txt` 內容：每個物體列出各 seed 的四個指標，並自動附上
`mean±std (n=3)` 彙總行（樣本標準差，n−1）：

```
[ Object: mug ]
    seed=42  sr=0.0008  psr=0.8603  p2r=0.9645  tcr=0.9154
    ...
    mean±std (n=3): sr=0.0018±0.0011  psr=0.8694±0.0079  p2r=0.9744±0.0103  tcr=0.9279±0.0115
```

設定說明（腳本內變數）：

| 參數 | 說明 |
|------|------|
| `CKPT` | checkpoint 路徑 |
| `MOE_ACTORS` | expert 數（PPO=1，MoE=2） |
| `NUM_ENVS` | 平行環境數（預設 500） |
| `SEEDS` | 評估用的隨機種子（預設 42 / 7 / 123） |
| `OBJECTS` | 評估物件（mug / drill / dumbbell） |

統計說明：

- 每趟會跳過前 `num_envs` 個 episode（warmup），從第二輪開始計數，避免冷啟動偏差。
- 兩個腳本跑的是**同一個任務檔**（`task=DiabloBalanceGrasp`），環境與成功判定完全相同，
  唯一變因是 checkpoint 與 `moe_num_actors`，比較公平。
- **注意**：任務程式碼改動後，舊的評估結果不可與新結果混用
  （例：6/10 與 6/12 同一 checkpoint 的 psr 相差約 0.4）。
  論文表格中所有模型都要用同一版程式碼重跑。

---

## Step 2：解析 log → Excel

```bash
python parse_eval_to_excel.py eval_ppo_20260612_154305 eval_results.xlsx
```

解析的是各 seed 的 `.log` 檔（取最後一行累計統計），與 summary.txt 格式無關。

**輸出：** `eval_results.xlsx`，包含 4 個 sheet（**不含最終成功率 sr**，見下方說明）：

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

> **關於最終成功率（sr）**：目前**刻意不畫** sr 的比較圖。
> 現有 checkpoint 的 sr 接近 0（瓶頸在 release + retreat 最後一關），
> 畫成圖表沒有鑑別度；sr 數字仍保留在 summary.txt 與 log 中，
> 待模型改善後再決定是否納入圖表。

---

## Step 3：畫評估結果圖

```bash
python plot_eval_results.py eval_results.xlsx thesis_fig
# 或指定模型名稱：
python plot_eval_results.py eval_ppo_<ts>.xlsx thesis_ppo_fig "PPO"
python plot_eval_results.py eval_moe_<ts>.xlsx thesis_moe_fig "MoE-PPO"
```

**輸出：**

| 檔案 | 說明 |
|------|------|
| `<prefix>_by_metric.png` | 三個指標並排，X 軸為物件，附誤差棒 |
| `<prefix>_by_object.png` | 三個物件並排，X 軸為指標（易 → 難） |

---

## Step 4：畫訓練獎勵曲線

### 4a. 單一模型（指定 label）

```bash
# PPO
python plot_reward_curves.py runs/<PPO_run_dir> thesis_ppo_reward --labels "PPO"

# MoE-PPO
python plot_reward_curves.py runs/<MoE_run_dir> thesis_moe_reward --labels "MoE-PPO"
```

### 4b. 兩個模型對比（PPO vs MoE-PPO）

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

**輸出（每次執行都會產生）：**

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
> 論文若要畫多 seed 平均 ± std 的訓練曲線，需要多個訓練 seed 的 run（目前各模型只有單一訓練 run）。

---

## 完整執行流程

```bash
# ① 跑評估（PPO + MoE-PPO 各自；評估模式由腳本自動開啟，不用改程式碼）
bash shell_eval_ppo.sh        # → eval_ppo_<ts>/
bash shell_eval_claude.sh     # → eval_claude_<ts>/
# 或一次跑兩個：bash run_all_eval.sh

# ② 解析 log → Excel
python parse_eval_to_excel.py eval_ppo_<ts>    eval_ppo_<ts>.xlsx
python parse_eval_to_excel.py eval_claude_<ts> eval_moe_<ts>.xlsx

# ③ 畫評估長條圖（p2r / tcr / psr 三指標，不含 sr）
python plot_eval_results.py eval_ppo_<ts>.xlsx thesis_ppo_fig "PPO"
python plot_eval_results.py eval_moe_<ts>.xlsx thesis_moe_fig "MoE-PPO"

# ④ 畫訓練曲線（各獎勵分量 + 總獎勵 + phase metrics）
python plot_reward_curves.py runs/<PPO_run_dir> thesis_ppo_reward --labels "PPO"
python plot_reward_curves.py runs/<MoE_run_dir> thesis_moe_reward --labels "MoE-PPO"
```

---

## 現有 Checkpoint 與最新評估結果

| 模型 | Checkpoint |
|------|------|
| PPO | `runs/DiabloBalanceGrasp_PPO_6000_09-21-50-29/nn/DiabloBalanceGrasp_PPO_6000.pth` |
| MoE-PPO | `runs/DiabloBalanceGrasp_moe_0.2_10-01-53-46/nn/DiabloBalanceGrasp_moe_0.2.pth` |

**正式評估結果（2026-06-12，嚴格判定 HOLD=10 + carry 版 p2r）**——六模型 sr/psr：

| 模型（變體/訓練seed） | mug sr | mug psr | drill psr | dumbbell sr | dumbbell psr |
|---|---|---|---|---|---|
| PPO_6000 | 0.000 | 0.826 | 0.912 | 0.000 | 0.974 |
| moe_t (simple K2/42) | 0.000 | 0.679 | 0.752 | 0.000 | 0.731 |
| moe_2021 (simple K2/2021) | 0.000 | 0.865 | 0.983 | 0.000 | 0.944 |
| moe_t3 (simple K3/42) | **0.250** | 0.914 | 0.982 | 0.015 | 0.885 |
| moe_0.2 (ELU K2/42) | **0.296** | 0.935 | 0.923 | **0.131** | 0.945 |
| moe_4242 (ELU K2/4242) | 0.190 | 0.937 | 0.989 | 0.002 | 0.937 |

正式結果資料夾（drill sr 全模型為 0）：
`eval_ppo_hold10_20260612_230202/`、`eval_moe2021_final_hold10_*`、`eval_moet_final_hold10_*`、
`eval_moet3_final_hold10_*`、`eval_moe0.2_final_hold10_*`、`eval_moe4242_final_hold10_*`
（舊的瞬間判定結果 `eval_*_20260612_1[5-7]*` 僅供新舊對照，論文用嚴格版。）

---

## 檔案清單

| 檔案 | 用途 |
|------|------|
| `shell_eval_ppo.sh` | PPO 評估腳本（自動開 eval_mode、算 mean±std） |
| `shell_eval_claude.sh` | MoE-PPO 評估腳本（同上） |
| `run_all_eval.sh` | PPO + MoE-PPO 一次跑完，自動 parse + plot |
| `parse_eval_to_excel.py` | log → Excel（p2r/tcr/psr，不含 sr） |
| `plot_eval_results.py` | 評估結果長條圖（支援第三參數指定模型名） |
| `plot_reward_curves.py` | 訓練曲線（總獎勵 + phase metrics + 各分量獨立圖） |
| `cfg/task/DiabloBalanceGrasp*.yaml` | `env.eval_mode` 開關（預設 False） |
| `tasks/diablo_balance_grasp.py` | PPO / MoE-PPO 任務（讀 cfg 的 eval_mode） |
