"""
plot.py — D5: 對比曲線 + 多目標 Pareto 圖

左圖: best-objective-so-far vs 評估次數 (random / OPRO / hybrid)
      —— Demo 高潮: 推理型最佳化用更少評估逼近好配置。
右圖: 正確率 vs 幻覺率 散點 (所有試過的配置) + Pareto 前緣
      —— 展示多目標權衡: 哪些配置在「不增幻覺」下最準。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

# Windows 內建中文字型，避免中文標籤變方框
plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from cache import ResultCache
from memory.trajectory import Trajectory

RESULTS = Path(__file__).parent / "results"
STRATEGIES = {"random.jsonl": ("Random", "#888", "o"),
              "opro.jsonl": ("OPRO", "#d1495b", "s"),
              "hybrid.jsonl": ("Hybrid", "#2e7d32", "^")}


def pareto_front(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """points = (hallucination, correctness)。要 hallucination 低、correctness 高。"""
    pts = sorted(points, key=lambda p: (p[0], -p[1]))
    front, best_corr = [], -1.0
    for hall, corr in pts:
        if corr > best_corr:
            front.append((hall, corr))
            best_corr = corr
    return front


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- 左: best-so-far 曲線 ---
    for fname, (label, color, marker) in STRATEGIES.items():
        path = RESULTS / fname
        if not path.exists():
            continue
        traj = Trajectory.load(path)
        if not traj.trials:
            continue
        curve = traj.best_curve()
        ax1.plot(range(1, len(curve) + 1), curve, marker + "-",
                 label=label, color=color)
    ax1.set_xlabel("評估次數 (configs evaluated)")
    ax1.set_ylabel("best objective so far\n(正確率 - 0.5×幻覺率)")
    ax1.set_title("最佳化效率: 越快越高越好")
    ax1.legend(); ax1.grid(alpha=0.3)

    # --- 右: Pareto (用 cache 裡所有試過的配置) ---
    cache = ResultCache()
    pts = [(1 - r["score"]["faithfulness"], r["score"]["correctness"])
           for r in cache.all_records()]
    if pts:
        xs, ys = zip(*pts)
        ax2.scatter(xs, ys, alpha=0.4, color="#888", label="all configs")
        front = pareto_front(pts)
        fx, fy = zip(*front)
        ax2.plot(fx, fy, "r-o", label="Pareto front")
    ax2.set_xlabel("幻覺率 (越低越好) →")
    ax2.set_ylabel("正確率 (越高越好) →")
    ax2.set_title("多目標權衡: 正確率 vs 幻覺率")
    ax2.legend(); ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = RESULTS / "comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"已存圖 -> {out}")


if __name__ == "__main__":
    main()
