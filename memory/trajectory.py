"""
memory/trajectory.py — 經驗軌跡庫 (D3)

一次最佳化 run 的試驗序列，存成 jsonl。每筆 Trial 記:
- config : 試了哪組參數
- score  : em/f1/recall/faithfulness/correctness/abstain_rate/成本
- objective : 拿來最佳化的純量 (正確率 - λ×幻覺率)
- lam    : 當時的 λ (不同權重的軌跡不可混著比)
- failures  : 1~2 個最糟的題 (含檢索預覽)，給 OPRO 做錯誤分析

OPRO 的 meta-agent 就是讀這個庫來推論下一步。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Trial:
    config: dict                      # {chunk_size, top_k, retriever, ...agentic 開關}
    score: dict                       # {em, f1, recall, faithfulness, correctness, ...}
    objective: float                  # 最佳化目標 (正確率 - λ×幻覺率)
    failures: list[dict] = field(default_factory=list)
    source: str = "init"              # "init" | "random" | "opro" | "hybrid"
    lam: float | None = None          # 計算 objective 時用的 λ

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class Trajectory:
    """append-only 試驗序列，同時維護記憶體內 list 與 jsonl 落地。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.trials: list[Trial] = []

    def add(self, trial: Trial) -> None:
        self.trials.append(trial)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(trial.to_json() + "\n")

    def tried_keys(self) -> set[tuple]:
        """已試過的 config key，供去重 (放程式裡，不靠 prompt)。"""
        fields = ("chunk_size", "top_k", "retriever", "chunk_overlap",
                  "rerank", "query_decompose", "verify")
        return {tuple(t.config.get(f) for f in fields) for t in self.trials}

    def best(self) -> Trial | None:
        return max(self.trials, key=lambda t: t.objective) if self.trials else None

    def best_curve(self) -> list[float]:
        """best-objective-so-far 隨評估次數的曲線 (給 plot.py)。"""
        curve, best = [], float("-inf")
        for t in self.trials:
            best = max(best, t.objective)
            curve.append(best)
        return curve

    @classmethod
    def load(cls, path: str | Path) -> "Trajectory":
        traj = cls(path)
        p = Path(path)
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    traj.trials.append(Trial(**json.loads(line)))
        return traj

    def reset(self) -> None:
        """清空 (重跑實驗用)。"""
        self.trials = []
        if self.path.exists():
            self.path.unlink()
