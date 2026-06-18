"""
cache.py — 快取式評估

搜索空間加了 agentic 開關後變成 216 組，全掃 oracle 已不可行
(這正是需要最佳化器的理由)。改成「查過的不重跑」:
  evaluate_cached(cfg) -> 命中快取直接回；未命中才呼叫 API 評估並存檔。

random / OPRO / hybrid 共用同一個 cache.json，重疊到的配置零成本。

注意: 快取以 config 為鍵，假設測試集固定 (load_hotpot 的 n/n_hard/seed 不變)。
"""

from __future__ import annotations

import json
from pathlib import Path

from rag import RagConfig

CACHE_PATH = Path(__file__).parent / "cache.json"

_FIELDS = ("chunk_size", "top_k", "retriever", "chunk_overlap",
           "rerank", "query_decompose", "verify")


def config_to_dict(cfg: RagConfig) -> dict:
    return {f: getattr(cfg, f) for f in _FIELDS}


def dict_to_config(d: dict) -> RagConfig:
    return RagConfig(**{f: d[f] for f in _FIELDS if f in d})


def key(cfg_or_dict) -> str:
    d = config_to_dict(cfg_or_dict) if isinstance(cfg_or_dict, RagConfig) else cfg_or_dict
    return "|".join(str(d[f]) for f in _FIELDS)


class ResultCache:
    def __init__(self, path: Path = CACHE_PATH):
        self.path = Path(path)
        self.store: dict[str, dict] = {}
        if self.path.exists():
            self.store = json.loads(self.path.read_text(encoding="utf-8"))

    def get(self, cfg: RagConfig) -> dict | None:
        return self.store.get(key(cfg))

    def _save(self) -> None:
        self.path.write_text(
            json.dumps(self.store, ensure_ascii=False, indent=2), encoding="utf-8")

    def evaluate_cached(self, cfg: RagConfig, examples, verbose: bool = False) -> dict:
        """回傳該配置的完整 record (命中快取則不呼叫 API)。"""
        hit = self.get(cfg)
        if hit is not None:
            return hit
        from eval import evaluate  # 局部 import，避免載入順序問題
        score = evaluate(cfg, examples, verbose=verbose)
        record = {
            "config": config_to_dict(cfg),
            "score": score.as_dict(),
            "failures": [d.as_dict() for d in score.worst_cases(2)],
            "fail_clusters": score.fail_clusters(),
        }
        self.store[key(cfg)] = record
        self._save()
        return record

    def all_records(self) -> list[dict]:
        return list(self.store.values())
