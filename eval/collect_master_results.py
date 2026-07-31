#!/usr/bin/env python3
"""
collect_master_results.py
從所有 eval 目錄（舊的 1337/2021 + 新的 extra seeds）提取：
  PSR | fall_rate | drop_rate | timeout_rate | minL2
並依訓練 reward 排序輸出論文用表格。

Usage: python collect_master_results.py [--dir BASEDIR]
"""
import os, re, sys, glob, argparse
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))

# ─── 舊 eval 目錄的 hardcode 映射 ─────────────────────────────────────────
EXISTING = {
    ("PPO",     1337): "eval_ppo_1337_hold10_20260618_131528",
    ("PPO",     2021): "eval_ppo_2021_hold10_20260618_131057",
    ("ReLU_k2", 1337): "eval_moek2_1337_hold10_20260618_132427",
    ("ReLU_k2", 2021): "eval_moek2_2021_hold10_20260618_132001",
    ("ReLU_k3", 1337): "eval_moek3_1337_hold10_20260618_133339",
    ("ReLU_k3", 2021): "eval_moek3_2021_hold10_20260618_132902",
    ("ELU_k2",  1337): "eval_ELU_k2_seed1337_20260623_135159",
    ("ELU_k2",  2021): "eval_ELU_k2_seed2021_20260623_135159",
    ("ELU_k3",  1337): "eval_ELU_k3_seed1337_20260623_135159",
    ("ELU_k3",  2021): "eval_ELU_k3_seed2021_20260623_135159",
}

# ─── 訓練 reward 排序 (降序，來自訓練日誌) ────────────────────────────────
TRAIN_REWARD = {
    ("ReLU_k2", 1337): 76873,
    ("ELU_k3",  789):  70569,
    ("ReLU_k2", 789):  69089,
    ("ELU_k2",  123):  68963,
    ("ReLU_k3", 2021): 68952,
    ("ELU_k2",  777):  68589,
    ("ELU_k2",  999):  65621,
    ("ELU_k3",  2021): 65449,
    ("ELU_k2",  789):  65388,
    ("ELU_k2",  42):   64847,
    ("ReLU_k3", 1337): 64045,
    ("ELU_k2",  456):  63653,
    ("ELU_k3",  123):  61994,
    ("PPO",     42):   61853,
    ("PPO",     789):  61335,
    ("PPO",     456):  60300,
    ("ReLU_k3", 42):   57915,
    ("ReLU_k2", 123):  55935,
    ("PPO",     2021): 55750,
    ("ReLU_k2", 2021): 53306,
    ("ELU_k3",  777):  51900,
    ("ReLU_k3", 456):  51615,
    ("ELU_k3",  999):  51099,
    ("PPO",     123):  50351,
    ("ELU_k2",  1337): 50156,
    ("ELU_k3",  1337): 49685,
    ("PPO",     1337): 45835,
    ("ELU_k3",  42):   41904,
    ("ReLU_k2", 42):   32648,
    ("ReLU_k3", 789):  18961,
    ("ELU_k3",  456):  17627,
    ("ReLU_k3", 123):  17514,
    ("ELU_k2",  2021): 12166,
    ("ReLU_k2", 456):   7801,
}

CONVERGENCE = {
    ("ReLU_k2", 456):  "✗中斷",
    ("ReLU_k3", 789):  "✗未收斂",
    ("ReLU_k3", 123):  "✗未收斂",
    ("ELU_k2",  2021): "✗未收斂",
    ("ELU_k3",  456):  "✗未收斂",
    ("ReLU_k2", 42):   "△部分",
}

OBJECTS   = ["mug", "drill", "dumbbell"]
EVAL_SEED_PATTERN = re.compile(r"_seed(\d+)\.log$")


def parse_log(filepath):
    """從單一 eval log 提取最終指標。"""
    metrics = {"psr": None, "fall": None, "drop": None, "timeout": None, "minL2": None}
    if not os.path.exists(filepath):
        return metrics
    with open(filepath) as f:
        lines = f.readlines()
    # 從後往前找最後一筆 place_L2 行（含 fall/drop/timeout）
    for line in reversed(lines):
        if "final place_L2" in line:
            m = re.search(r"final fall_rate:\s*([0-9.]+)", line)
            if m: metrics["fall"] = float(m.group(1))
            m = re.search(r"final drop_rate:\s*([0-9.]+)", line)
            if m: metrics["drop"] = float(m.group(1))
            m = re.search(r"final timeout_rate:\s*([0-9.]+)", line)
            if m: metrics["timeout"] = float(m.group(1))
            m = re.search(r"min_L2\(mm\):\s*([0-9.]+)", line)
            if m: metrics["minL2"] = float(m.group(1))
            break
    for line in reversed(lines):
        if "final partial_success_rate:" in line:
            m = re.search(r"final partial_success_rate:\s*([0-9.]+)", line)
            if m: metrics["psr"] = float(m.group(1))
            break
    return metrics


def collect_eval_dir(eval_dir):
    """讀取一個 eval 目錄，回傳 {obj: {eval_seed: metrics}} 。"""
    result = {}
    for obj in OBJECTS:
        result[obj] = {}
        for log_file in glob.glob(os.path.join(eval_dir, f"{obj}_seed*.log")):
            m = EVAL_SEED_PATTERN.search(log_file)
            if not m:
                continue
            eseed = int(m.group(1))
            result[obj][eseed] = parse_log(log_file)
    return result


def avg(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def fmt(v, decimals=4):
    return f"{v:.{decimals}f}" if v is not None else "N/A"


def find_new_eval_dir(basedir, model, seed):
    """找 eval_extra_seeds.sh 產生的目錄。"""
    pattern = os.path.join(basedir, f"eval_{model}_seed{seed}_*")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None  # 取最新的


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=BASE, help="isaacgymenvs 根目錄")
    args = parser.parse_args()
    basedir = args.dir

    print("=" * 100)
    print(f"{'Rank':>4}  {'Model':<12} {'Seed':>5}  {'TrainRew':>8}  "
          f"{'PSR':>6}  {'fall':>6}  {'drop':>6}  {'timeout':>7}  {'minL2':>7}  Status")
    print("=" * 100)

    rank = 0
    for (model, seed), train_rew in sorted(TRAIN_REWARD.items(), key=lambda x: -x[1]):
        rank += 1
        status = CONVERGENCE.get((model, seed), "✓")

        # 找 eval 目錄
        if (model, seed) in EXISTING:
            eval_dir = os.path.join(basedir, EXISTING[(model, seed)])
        else:
            eval_dir = find_new_eval_dir(basedir, model, seed)

        if eval_dir is None or not os.path.isdir(eval_dir):
            print(f"{rank:>4}  {model:<12} {seed:>5}  {train_rew:>8,}  "
                  f"{'—':>6}  {'—':>6}  {'—':>6}  {'—':>7}  {'—':>7}  {status}  [eval 未找到: {eval_dir}]")
            continue

        data = collect_eval_dir(eval_dir)

        # 計算跨 object 和 eval seed 的平均
        all_psrs, all_falls, all_drops, all_touts, all_ml2s = [], [], [], [], []
        for obj in OBJECTS:
            for eseed, m in data[obj].items():
                all_psrs.append(m["psr"])
                all_falls.append(m["fall"])
                all_drops.append(m["drop"])
                all_touts.append(m["timeout"])
                all_ml2s.append(m["minL2"])

        psr     = avg(all_psrs)
        fall    = avg(all_falls)
        drop    = avg(all_drops)
        timeout = avg(all_touts)
        minl2   = avg(all_ml2s)

        print(f"{rank:>4}  {model:<12} {seed:>5}  {train_rew:>8,}  "
              f"{fmt(psr):>6}  {fmt(fall):>6}  {fmt(drop):>6}  {fmt(timeout):>7}  {fmt(minl2,1):>7}  {status}")

    print("=" * 100)
    print("\nPSR = phase success rate; fall/drop/timeout = failure reason rates; minL2 = 最近距離平均(mm)")
    print("所有指標為 3 eval seeds × 3 objects 平均")


if __name__ == "__main__":
    main()
