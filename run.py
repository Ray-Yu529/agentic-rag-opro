"""
run.py — 主編排: random vs OPRO vs hybrid 對照 (多目標)

搜索空間 216 組，三種策略各用 BUDGET 次評估去找最佳配置。
分數查共用 cache.json (查過不重跑)，OPRO/hybrid 額外呼叫 meta-LLM 推理。

資料集二選一:
  預設                      HotpotQA 分層子集 (N/N_HARD)
  --corpus PATH --qa PATH   你自己的文件 + QA (QA 先用 dataset.py 生成或自備)

產出 results/*.jsonl + results/meta.json，之後跑 plot.py 出對比曲線 + Pareto 圖。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from eval import load_hotpot
from cache import ResultCache, dataset_key, custom_dataset_key
from optimizer import random_search, opro_search
from hybrid import hybrid_search

load_dotenv()

RESULTS = Path(__file__).parent / "results"
BUDGET = 10    # 每種策略的評估次數 (216 組裡只試 10 組)
SEED = 42
N_INIT = 3     # 暖身組數 (三種策略同 seed，公平起跑)
N, N_HARD = 30, 15   # HotpotQA 測試集大小 (需與 sweep.py 一致才能共用 cache)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="random vs OPRO vs hybrid 對照")
    ap.add_argument("--corpus", help="自訂語料: .txt/.md 檔或資料夾 (不給則用 HotpotQA)")
    ap.add_argument("--qa", help="自訂 QA jsonl (與 --corpus 併用；"
                                 "先跑 python dataset.py --corpus ... 生成，或自備)")
    ap.add_argument("--n", type=int, default=None,
                    help="題數 (custom 預設用全部 QA；hotpot 固定用 N=30)")
    return ap.parse_args()


def load_examples_and_key(args) -> tuple[list, str]:
    """回傳 (examples, cache 的 dataset key)。custom 與 hotpot 共用同一條下游管線。"""
    if args.corpus:
        if not args.qa:
            raise SystemExit(
                "--corpus 需要搭配 --qa。先生成:\n"
                f"  python dataset.py --corpus {args.corpus} --n 20 --out data/qa.jsonl")
        from dataset import load_corpus, load_qa, build_examples, dataset_fingerprint
        paragraphs = load_corpus(args.corpus)
        qa = load_qa(args.qa)
        examples = build_examples(paragraphs, qa, n=args.n, seed=SEED, log_fn=print)
        key = custom_dataset_key(dataset_fingerprint(paragraphs, qa),
                                 len(examples), SEED)
        print(f"自訂資料集: {len(paragraphs)} 段語料, {len(examples)} 題")
        return examples, key
    return load_hotpot(n=N, n_hard=N_HARD), dataset_key(N, N_HARD, SEED)


def summarize(name: str, traj) -> None:
    b = traj.best()
    c = b.config
    print(f"\n[{name}] {len(traj.trials)} 次評估 | best obj={b.objective:.3f} "
          f"(正確率={b.score['correctness']:.2f}, 幻覺率={1-b.score['faithfulness']:.2f})")
    print(f"        最佳配置: chunk={c['chunk_size']}, top_k={c['top_k']}, "
          f"ret={c['retriever']}, rerank={c['rerank']}, "
          f"decomp={c['query_decompose']}, verify={c['verify']}")


def main() -> None:
    args = parse_args()
    examples, dkey = load_examples_and_key(args)
    # 與 sweep 共用; 已掃過的配置在這免費重用 (快取鍵含測試集+評估指紋)
    cache = ResultCache(dataset=dkey)
    # 記下本次的 dataset key，plot.py 的 Pareto 圖才會取到同一批 record
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "meta.json").write_text(
        json.dumps({"dataset": dkey}, ensure_ascii=False), encoding="utf-8")

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
