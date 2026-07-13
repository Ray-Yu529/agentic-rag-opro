"""
plot.py — D5: 對比曲線 + 多目標 Pareto 圖

左圖: best-objective-so-far vs 評估次數 (random / OPRO / hybrid)
      —— Demo 高潮: 推理型最佳化用更少評估逼近好配置。
右圖: 正確率 vs 幻覺率 散點 (所有試過的配置) + Pareto 前緣
      —— 展示多目標權衡: 哪些配置在「不增幻覺」下最準。
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Windows 內建中文字型，避免中文標籤變方框
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from cache import DEFAULT_DATASET, ResultCache
from memory.trajectory import Trajectory
from pareto import pareto_front


def _last_meta() -> dict:
    """run.py/sweep.py 記在 meta.json 的 dataset key / seeds
    (自訂資料集與多 seed 才能畫對圖)。"""
    meta = RESULTS / "meta.json"
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8"))
    return {}

RESULTS = Path(__file__).parent / "results"
# 與 Web UI 同一套系列色 (CVD/對比已驗證):
# opro=blue, hybrid=aqua, bandit=violet, random=中性灰
STRATEGIES = {"random.jsonl": ("Random", "#898781", "o"),
              "bandit.jsonl": ("Bandit", "#4a3aa7", "D"),
              "opro.jsonl": ("OPRO", "#2a78d6", "s"),
              "hybrid.jsonl": ("Hybrid", "#1baf7a", "^")}


def _load_curves(base: str, seeds: list[int]) -> tuple[list[list[float]], float | None]:
    """讀一個策略的 best-so-far 曲線: 單 seed 讀 {base}.jsonl，
    多 seed 讀 {base}_s{seed}.jsonl。回傳 (曲線列表, λ)。"""
    paths = ([RESULTS / f"{base}.jsonl"] if len(seeds) == 1
             else [RESULTS / f"{base}_s{s}.jsonl" for s in seeds])
    curves, lam = [], None
    for p in paths:
        if not p.exists():
            continue
        traj = Trajectory.load(p)
        if traj.trials:
            curves.append(traj.best_curve())
            lam = traj.trials[-1].lam
    return curves, lam


def main() -> None:
    meta = _last_meta()
    seeds = meta.get("seeds", [42])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- 左: best-so-far 曲線 (多 seed 時畫 mean 實線 + min–max 帶狀) ---
    for fname, (label, color, marker) in STRATEGIES.items():
        curves, lam = _load_curves(fname.removesuffix(".jsonl"), seeds)
        if not curves:
            continue
        if lam is not None:
            label = f"{label} (λ={lam})"
        m = min(len(c) for c in curves)
        arr = np.array([c[:m] for c in curves])
        xs = range(1, m + 1)
        mean = arr.mean(axis=0)
        if len(curves) > 1:
            label = f"{label}, {len(curves)} seeds"
            ax1.fill_between(xs, arr.min(axis=0), arr.max(axis=0),
                             color=color, alpha=0.15)
        ax1.plot(xs, mean, marker + "-", label=label, color=color)
    ax1.set_xlabel("評估次數 (configs evaluated)")
    ax1.set_ylabel("best objective so far\n(正確率 - λ×幻覺率 - μ×成本)")
    ax1.set_title("最佳化效率: 越快越高越好")
    ax1.legend(); ax1.grid(alpha=0.3)

    # --- 右: Pareto，三目標視覺化 (第三軸 = 成本，用點大小編碼) ---
    cache = ResultCache(dataset=meta.get("dataset", DEFAULT_DATASET))
    records = cache.all_records()
    pts = [(1 - r["score"]["faithfulness"], r["score"]["correctness"])
           for r in records]
    if pts:
        toks = np.array([r["score"].get("avg_tokens", 0) for r in records], dtype=float)
        span = toks.max() - toks.min()
        sizes = 25 + 155 * (toks - toks.min()) / (span if span > 0 else 1)
        xs, ys = zip(*pts)
        ax2.scatter(xs, ys, s=sizes, alpha=0.45, color="#898781",
                    label="all configs (點大小=每題 tokens)")
        front = pareto_front(pts)
        fx, fy = zip(*front)
        ax2.plot(fx, fy, "-o", color="#2a78d6", label="Pareto front")
    ax2.set_xlabel("幻覺率 (越低越好) →")
    ax2.set_ylabel("正確率 (越高越好) →")
    ax2.set_title("三目標權衡: 正確率 vs 幻覺率 (大小=成本)")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = RESULTS / "comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"已存圖 -> {out}")


if __name__ == "__main__":
    main()
