"""
sweep.py — D2.5: 全 27 組掃描，產生 oracle.json + go/no-go 報告

把搜索空間每一組都用 API 真實評估一次 (這是整個 Demo 唯一一次大量 API 呼叫)，
結果存成 oracle.json。之後 random / OPRO 都從 oracle 查分數，零額外成本。

go/no-go 檢查: 看最佳與最差配置的 F1 差距。
  差距夠大 -> 參數有區分度，Demo 站得住。
  差距太小 -> 換更難的題或加大參數跨度，否則最佳化看不出差別。
"""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from eval import load_hotpot, evaluate
from optimizer import all_configs, ORACLE_PATH

load_dotenv()


def main(n: int = 30, n_hard: int = 15) -> None:
    examples = load_hotpot(n=n, n_hard=n_hard)
    configs = all_configs()
    print(f"載入 {len(examples)} 題，準備掃描 {len(configs)} 組配置 "
          f"(約 {len(configs) * len(examples)} 次生成呼叫)\n")

    records = []
    for cfg in tqdm(configs, desc="sweep"):
        score = evaluate(cfg, examples, verbose=False)
        records.append({
            "config": {"chunk_size": cfg.chunk_size, "top_k": cfg.top_k,
                       "retriever": cfg.retriever, "chunk_overlap": cfg.chunk_overlap},
            "score": score.as_dict(),
            "objective": score.f1,                       # 最佳化目標 = F1
            "failures": [d.as_dict() for d in score.worst_cases(2)],
        })

    Path(ORACLE_PATH).write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已存 oracle -> {ORACLE_PATH}")

    # --- go/no-go 報告 ---
    ranked = sorted(records, key=lambda r: r["objective"], reverse=True)
    best, worst = ranked[0], ranked[-1]
    gap = best["objective"] - worst["objective"]
    print("\n========== D2.5 go/no-go ==========")
    print(f"最佳 F1 = {best['objective']:.3f}  @ {best['config']}")
    print(f"最差 F1 = {worst['objective']:.3f}  @ {worst['config']}")
    print(f"差距   = {gap:.3f}")
    if gap >= 0.10:
        print("✅ GO: 參數有明顯區分度，可以進 D3/D4 最佳化。")
    elif gap >= 0.05:
        print("⚠️  邊緣: 區分度偏小，Demo 勉強可用，建議加大 chunk_size 跨度或挑更難的題。")
    else:
        print("❌ NO-GO: 參數幾乎沒區分度。先換更難的題 (調高 n_hard) 或擴大搜索空間。")


if __name__ == "__main__":
    main()
