"""
sweep.py — D2.5: 基準網格掃描 + go/no-go (cache-backed)

完整空間 216 組太大、不適合全掃。這裡只掃「核心數值網格」(27 組:
chunk_size × top_k × retriever，agentic 開關全關)，目的有二:
  1. go/no-go: 看核心參數對 correctness 是否有區分度。
  2. 順便把這 27 組存進共用 cache.json，之後 random/OPRO/hybrid 重疊到就免費。

agentic 開關 (rerank/decompose/verify) 的效益交給 run.py 的最佳化器去探索。
"""

from __future__ import annotations

from dotenv import load_dotenv
from tqdm import tqdm

from rag import RagConfig
from eval import load_hotpot
from cache import ResultCache, dataset_key
from optimizer import SEARCH_SPACE, objective

load_dotenv()


def base_grid() -> list[RagConfig]:
    return [RagConfig(chunk_size=c, top_k=k, retriever=r)  # agentic 開關用預設 False
            for c in SEARCH_SPACE["chunk_size"]
            for k in SEARCH_SPACE["top_k"]
            for r in SEARCH_SPACE["retriever"]]


def main(n: int = 30, n_hard: int = 15) -> None:
    examples = load_hotpot(n=n, n_hard=n_hard)
    cache = ResultCache(dataset=dataset_key(n, n_hard))
    configs = base_grid()
    print(f"載入 {len(examples)} 題，掃描核心網格 {len(configs)} 組 (agentic 全關)\n")

    records = []
    for cfg in tqdm(configs, desc="sweep"):
        records.append(cache.evaluate_cached(cfg, examples, verbose=False))

    # --- go/no-go: 用 correctness 看區分度 ---
    ranked = sorted(records, key=lambda r: r["score"]["correctness"], reverse=True)
    best, worst = ranked[0], ranked[-1]
    gap = best["score"]["correctness"] - worst["score"]["correctness"]
    print(f"\n已存 cache (本測試集 {len(cache.all_records())} 組)")
    print("\n========== D2.5 go/no-go ==========")
    print(f"最佳 correctness = {best['score']['correctness']:.3f}  @ {best['config']}")
    print(f"最差 correctness = {worst['score']['correctness']:.3f}  @ {worst['config']}")
    print(f"差距 = {gap:.3f}")
    if gap >= 0.10:
        print("✅ GO: 核心參數有明顯區分度，可進 run.py 跑最佳化。")
    elif gap >= 0.05:
        print("⚠️  邊緣: 區分度偏小，建議加大 chunk_size 跨度或挑更難的題。")
    else:
        print("❌ NO-GO: 幾乎沒區分度。先調高 n_hard 或擴大搜索空間。")


if __name__ == "__main__":
    main()
