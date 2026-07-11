"""
memory/warmstart.py — 跨 run 經驗記憶 (warm-start)

每次最佳化 run 結束後記下「這個資料集長怎樣 + 哪組配置贏了」；
新資料集進來時，用資料集特徵 (段落數/平均段長/中文比例) 找最相似的
過去 run，把它們的最佳配置當 OPRO/hybrid 的暖身組。

注意: warm-start 會破壞「三策略同 seed 公平起跑」的對照實驗，
所以是 opt-in (--warmstart / UI 勾選)；「記錄」則永遠進行 (不花 API)。
存放於 data/warmstart.jsonl (gitignored，屬使用者資料)。
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

PATH = Path(__file__).parent.parent / "data" / "warmstart.jsonl"
_CJK = re.compile(r"[一-鿿]")


def dataset_features(examples) -> dict:
    """資料集的粗粒度特徵，供相似度比對 (不花 API)。"""
    paras = {p for ex in examples for p in ex.paragraphs}
    sample = " ".join(list(paras)[:200])
    return {
        "n_questions": len(examples),
        "n_paragraphs": len(paras),
        "avg_para_chars": round(sum(map(len, paras)) / max(1, len(paras)), 1),
        "cjk_ratio": round(len(_CJK.findall(sample)) / max(1, len(sample)), 3),
    }


def _load() -> list[dict]:
    if not PATH.exists():
        return []
    return [json.loads(ln) for ln in PATH.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


def record(dataset_key: str, features: dict, best_config: dict,
           objective: float) -> None:
    """記錄一次 run 的最佳配置；同資料集只保留 objective 最高的一筆。"""
    rows = _load()
    old = next((r for r in rows if r["dataset"] == dataset_key), None)
    if old is not None:
        if old["objective"] >= objective:
            return
        rows.remove(old)
    rows.append({"dataset": dataset_key, "features": features,
                 "best_config": best_config, "objective": round(objective, 4)})
    PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PATH.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                   encoding="utf-8")
    tmp.replace(PATH)


def _distance(a: dict, b: dict) -> float:
    """特徵距離: 段長/段數用對數比 (規模差一個量級才算遠)，語言差異加重。"""
    return (abs(math.log((a["avg_para_chars"] + 1) / (b["avg_para_chars"] + 1)))
            + 0.5 * abs(math.log((a["n_paragraphs"] + 1) / (b["n_paragraphs"] + 1)))
            + 2.0 * abs(a["cjk_ratio"] - b["cjk_ratio"]))


def suggest(features: dict, k: int = 2,
            exclude_dataset: str | None = None) -> list[dict]:
    """回傳最多 k 組「相似資料集上贏過的配置」(去重)。記憶庫空時回 []。"""
    rows = [r for r in _load() if r["dataset"] != exclude_dataset]
    rows.sort(key=lambda r: _distance(r["features"], features))
    out, seen = [], set()
    for r in rows:
        key = json.dumps(r["best_config"], sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(r["best_config"])
        if len(out) >= k:
            break
    return out
