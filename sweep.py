"""
sweep.py — D2.5: 基準網格掃描 + go/no-go (cache-backed)

完整空間 216 組太大、不適合全掃。這裡只掃「核心數值網格」(27 組:
chunk_size × top_k × retriever，agentic 開關全關)，目的有二:
  1. go/no-go: 看核心參數對 correctness 是否有區分度。
  2. 順便把這 27 組存進共用 cache.json，之後 random/OPRO/hybrid 重疊到就免費。

agentic 開關 (rerank/decompose/verify) 的效益交給 run.py 的最佳化器去探索。
"""

from __future__ import annotations

from math import comb

from dotenv import load_dotenv
from tqdm import tqdm

from rag import RagConfig
from eval import load_hotpot
from cache import ResultCache, dataset_key
from optimizer import SEARCH_SPACE

load_dotenv()


def _sign_test_p(b: int, c: int) -> float:
    """精確二項符號檢定 (McNemar) 雙尾 p 值。
    b/c = 兩配置在「同一題」上一對一錯的不一致題數。"""
    n = b + c
    if n == 0:
        return 1.0
    k = max(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n)


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

    # --- go/no-go: 平均差距 + 成對符號檢定 ---
    # n=30 時 correctness 的標準誤約 0.09，27 組取極值差很容易只是噪音；
    # 改在「同一題集」上逐題成對比較 (符號檢定)，比看平均分敏感得多。
    ranked = sorted(records, key=lambda r: r["score"]["correctness"], reverse=True)
    best, worst = ranked[0], ranked[-1]
    gap = best["score"]["correctness"] - worst["score"]["correctness"]
    print(f"\n已存 cache (本測試集 {len(cache.all_records())} 組)")
    print("\n========== D2.5 go/no-go ==========")
    print(f"最佳 correctness = {best['score']['correctness']:.3f}  @ {best['config']}")
    print(f"最差 correctness = {worst['score']['correctness']:.3f}  @ {worst['config']}")
    print(f"平均差距 = {gap:.3f}")

    pq_best = best.get("per_question_correct") or {}
    pq_worst = worst.get("per_question_correct") or {}
    common = set(pq_best) & set(pq_worst)
    b = sum(1 for q in common if pq_best[q] > pq_worst[q])
    c = sum(1 for q in common if pq_worst[q] > pq_best[q])
    p = _sign_test_p(b, c)
    print(f"成對比較 (同題集逐題): 最佳贏 {b} 題、最差贏 {c} 題、平手 {len(common) - b - c} 題")
    print(f"符號檢定 p ≈ {p:.3f}  (注意: 最佳/最差是事後從 {len(records)} 組挑的，"
          "多重比較未校正，p 值偏樂觀)")

    if p < 0.05:
        print("✅ GO: 核心參數有統計上可信的區分度，可進 run.py 跑最佳化。")
    elif p < 0.20 or gap >= 0.10:
        print("⚠️  邊緣: 區分度可能只是噪音。建議調高 n (題數) 或挑更難的題再確認。")
    else:
        print("❌ NO-GO: 幾乎沒區分度。先調高 n / n_hard 或擴大搜索空間。")


if __name__ == "__main__":
    main()
