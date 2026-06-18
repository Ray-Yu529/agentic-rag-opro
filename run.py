"""
run.py — D5 主編排: random vs OPRO 對照實驗

前提: 已跑過 sweep.py 產生 oracle.json。
本檔不做大量 RAG 評估 (分數查 oracle)，只有 OPRO 會呼叫 meta-LLM 推理。

產出:
  results/random.jsonl  — random search 軌跡
  results/opro.jsonl     — OPRO 軌跡
之後跑 plot.py 出對比曲線。
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from optimizer import load_oracle, random_search, opro_search

load_dotenv()

RESULTS = Path(__file__).parent / "results"
BUDGET = 8     # 兩種策略各跑幾次評估 (27 組裡只試 8 組)
SEED = 42
N_INIT = 3     # OPRO 暖身組數 (與 random 同 seed，公平起跑)


def summarize(name: str, traj) -> None:
    best = traj.best()
    print(f"\n[{name}] {len(traj.trials)} 次評估，best F1 = {best.objective:.3f} "
          f"@ chunk={best.config['chunk_size']}, top_k={best.config['top_k']}, "
          f"retriever={best.config['retriever']}")


def main() -> None:
    oracle = load_oracle()
    oracle_best = max(r["objective"] for r in oracle.values())
    print(f"oracle 全域最佳 F1 = {oracle_best:.3f} (27 組的真最佳，當作天花板)\n")

    print("=== Random search baseline ===")
    rand = random_search(oracle, budget=BUDGET, seed=SEED,
                         traj_path=str(RESULTS / "random.jsonl"))
    summarize("random", rand)

    print("\n=== OPRO meta-optimizer ===")
    opro = opro_search(oracle, budget=BUDGET, seed=SEED, n_init=N_INIT,
                       traj_path=str(RESULTS / "opro.jsonl"))
    summarize("opro", opro)

    print("\n完成。跑 plot.py 出對比曲線。")


if __name__ == "__main__":
    main()
