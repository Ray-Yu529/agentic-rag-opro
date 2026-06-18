"""
plot.py — D5: 評估次數 vs best-so-far F1 對比曲線

讀 results/random.jsonl 與 results/opro.jsonl，畫出兩條 best-objective-so-far 曲線，
再加一條 oracle 天花板。這張圖就是 Demo 的高潮:
「OPRO 用更少評估次數逼近最佳配置」。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from optimizer import load_oracle
from memory.trajectory import Trajectory

RESULTS = Path(__file__).parent / "results"


def main() -> None:
    rand = Trajectory.load(RESULTS / "random.jsonl")
    opro = Trajectory.load(RESULTS / "opro.jsonl")
    if not rand.trials or not opro.trials:
        raise SystemExit("找不到軌跡，請先跑 run.py。")

    oracle = load_oracle()
    oracle_best = max(r["objective"] for r in oracle.values())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    xr = range(1, len(rand.best_curve()) + 1)
    xo = range(1, len(opro.best_curve()) + 1)
    ax.plot(xr, rand.best_curve(), "o-", label="Random search", color="#888")
    ax.plot(xo, opro.best_curve(), "s-", label="OPRO (ours)", color="#d1495b")
    ax.axhline(oracle_best, ls="--", color="#2e7d32",
               label=f"Oracle best ({oracle_best:.3f})")

    ax.set_xlabel("Number of configurations evaluated")
    ax.set_ylabel("Best F1 so far")
    ax.set_title("Agentic RAG tuning: OPRO vs Random")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = RESULTS / "comparison.png"
    fig.savefig(out, dpi=150)
    print(f"已存圖 -> {out}")


if __name__ == "__main__":
    main()
