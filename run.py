"""
run.py — 主編排: random vs OPRO vs hybrid 對照 (多目標)

搜索空間 216 組，三種策略各用 BUDGET 次評估去找最佳配置。
分數查共用 cache.json (查過不重跑)，OPRO/hybrid 額外呼叫 meta-LLM 推理。

產出 results/*.jsonl，之後跑 plot.py 出對比曲線 + Pareto 圖。
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from eval import load_hotpot
from cache import ResultCache, dataset_key
from optimizer import random_search, opro_search
from hybrid import hybrid_search

load_dotenv()

RESULTS = Path(__file__).parent / "results"
BUDGET = 10    # 每種策略的評估次數 (216 組裡只試 10 組)
SEED = 42
N_INIT = 3     # 暖身組數 (三種策略同 seed，公平起跑)
N, N_HARD = 30, 15   # 測試集大小 (需與 sweep.py 一致才能共用 cache)


def summarize(name: str, traj) -> None:
    b = traj.best()
    c = b.config
    print(f"\n[{name}] {len(traj.trials)} 次評估 | best obj={b.objective:.3f} "
          f"(正確率={b.score['correctness']:.2f}, 幻覺率={1-b.score['faithfulness']:.2f})")
    print(f"        最佳配置: chunk={c['chunk_size']}, top_k={c['top_k']}, "
          f"ret={c['retriever']}, rerank={c['rerank']}, "
          f"decomp={c['query_decompose']}, verify={c['verify']}")


def main() -> None:
    examples = load_hotpot(n=N, n_hard=N_HARD)
    # 與 sweep 共用; 已掃過的核心網格在這免費重用 (快取鍵含測試集 fingerprint)
    cache = ResultCache(dataset=dataset_key(N, N_HARD, SEED))

    print("=== Random search baseline ===")
    rand = random_search(examples, cache, budget=BUDGET, seed=SEED,
                         traj_path=str(RESULTS / "random.jsonl"))
    summarize("random", rand)

    print("\n=== OPRO meta-optimizer ===")
    opro = opro_search(examples, cache, budget=BUDGET, seed=SEED, n_init=N_INIT,
                       traj_path=str(RESULTS / "opro.jsonl"))
    summarize("opro", opro)

    print("\n=== Hybrid (LLM 縮範圍 + Optuna 收斂) ===")
    hyb = hybrid_search(examples, cache, budget=BUDGET, seed=SEED, n_init=N_INIT,
                        traj_path=str(RESULTS / "hybrid.jsonl"))
    summarize("hybrid", hyb)

    print("\n完成。跑 plot.py 出對比曲線 + Pareto 圖。")


if __name__ == "__main__":
    main()
