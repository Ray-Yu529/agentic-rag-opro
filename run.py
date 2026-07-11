"""
run.py — 主編排: random vs OPRO vs hybrid 對照 (多目標)

搜索空間 216 組，三種策略各用 BUDGET 次評估去找最佳配置。
分數查共用 cache.json (查過不重跑)，OPRO/hybrid 額外呼叫 meta-LLM 推理。

資料集二選一:
  預設                      HotpotQA 分層子集 (N/N_HARD)
  --corpus PATH --qa PATH   你自己的文件 + QA (QA 先用 dataset.py 生成或自備)

其他:
  --seeds K      跑 K 個 seed (plot.py 會畫 mean±range 帶狀，結論才有統計意義；
                 cache 以配置為鍵跨 seed 共用，重疊到的配置免費)
  --mu M         成本懲罰權重: objective -= M × 每題平均 ktok
  --warmstart    OPRO/hybrid 用過去相似資料集的最佳配置暖身
                 (破壞與 random 的公平起跑，對照實驗時別開)

產出 results/*.jsonl + results/meta.json，之後跑 plot.py 出對比曲線 + Pareto 圖。
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from dotenv import load_dotenv

from eval import load_hotpot
from cache import ResultCache, dataset_key, custom_dataset_key
from optimizer import DEFAULT_LAMBDA, random_search, opro_search
from hybrid import hybrid_search
from memory import warmstart

load_dotenv()

RESULTS = Path(__file__).parent / "results"
BUDGET = 10    # 每種策略的評估次數 (216 組裡只試 10 組)
SEED = 42
N_INIT = 3     # 暖身組數 (三種策略同 seed，公平起跑)
N, N_HARD = 30, 15   # HotpotQA 測試集大小 (需與 sweep.py 一致才能共用 cache)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="random vs OPRO vs hybrid 對照")
    ap.add_argument("--corpus", help="自訂語料: .txt/.md/.pdf 檔或資料夾 (不給則用 HotpotQA)")
    ap.add_argument("--qa", help="自訂 QA jsonl (與 --corpus 併用；"
                                 "先跑 python dataset.py --corpus ... 生成，或自備)")
    ap.add_argument("--n", type=int, default=None,
                    help="題數 (custom 預設用全部 QA；hotpot 固定用 N=30)")
    ap.add_argument("--seeds", type=int, default=1,
                    help="跑幾個 seed (>1 時 plot.py 畫 mean±range 帶狀)")
    ap.add_argument("--mu", type=float, default=0.0,
                    help="成本懲罰權重 μ (objective -= μ×每題ktok)")
    ap.add_argument("--warmstart", action="store_true",
                    help="OPRO/hybrid 用過去相似資料集的最佳配置暖身 "
                         "(破壞與 random 的公平起跑)")
    return ap.parse_args()


def load_examples_and_key(args) -> tuple[list, str]:
    """回傳 (examples, cache 的 dataset key)。custom 與 hotpot 共用同一條下游管線。"""
    if args.corpus:
        if not args.qa:
            raise SystemExit(
                "--corpus 需要搭配 --qa。先生成:\n"
                f"  python dataset.py --corpus {args.corpus} --n 20 --out data/qa.jsonl")
        from dataset import load_corpus, load_qa, build_examples, dataset_fingerprint
        paragraphs = load_corpus(args.corpus, log_fn=print)
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
    seeds = [SEED + i for i in range(max(1, args.seeds))]
    # 記下本次的 dataset key / seeds，plot.py 才會取到同一批 record 與正確的軌跡檔
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "meta.json").write_text(
        json.dumps({"dataset": dkey, "seeds": seeds,
                    "lam": DEFAULT_LAMBDA, "mu": args.mu}, ensure_ascii=False),
        encoding="utf-8")

    feats = warmstart.dataset_features(examples)
    warm = None
    if args.warmstart:
        warm = warmstart.suggest(feats, k=2, exclude_dataset=dkey) or None
        print(f"warm-start 建議: {warm or '無 (記憶庫還是空的)'}")

    finals: dict[str, list[float]] = {"random": [], "opro": [], "hybrid": []}
    overall_best = None
    for seed in seeds:
        sfx = "" if len(seeds) == 1 else f"_s{seed}"
        if len(seeds) > 1:
            print(f"\n================ seed {seed} ================")

        print("=== Random search baseline ===")
        rand = random_search(examples, cache, budget=BUDGET, seed=seed,
                             traj_path=str(RESULTS / f"random{sfx}.jsonl"),
                             mu=args.mu)
        summarize("random", rand)

        print("\n=== OPRO meta-optimizer ===")
        opro = opro_search(examples, cache, budget=BUDGET, seed=seed, n_init=N_INIT,
                           traj_path=str(RESULTS / f"opro{sfx}.jsonl"),
                           mu=args.mu, warm_configs=warm)
        summarize("opro", opro)

        print("\n=== Hybrid (LLM 縮範圍 + Optuna 收斂) ===")
        hyb = hybrid_search(examples, cache, budget=BUDGET, seed=seed, n_init=N_INIT,
                            traj_path=str(RESULTS / f"hybrid{sfx}.jsonl"),
                            mu=args.mu, warm_configs=warm)
        summarize("hybrid", hyb)

        for name, traj in (("random", rand), ("opro", opro), ("hybrid", hyb)):
            b = traj.best()
            finals[name].append(b.objective)
            if overall_best is None or b.objective > overall_best.objective:
                overall_best = b

    if len(seeds) > 1:
        print("\n========== 多 seed 匯總 (final best objective) ==========")
        for name, vals in finals.items():
            mean = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(f"  {name:6s}: {mean:.3f} ± {sd:.3f}  ({len(vals)} seeds)")

    # 跨 run 經驗記憶: 記下本資料集的最佳配置 (使用與否由 --warmstart 決定)
    if overall_best is not None:
        warmstart.record(dkey, feats, overall_best.config, overall_best.objective)

    print("\n完成。跑 plot.py 出對比曲線 + Pareto 圖。")


if __name__ == "__main__":
    main()
