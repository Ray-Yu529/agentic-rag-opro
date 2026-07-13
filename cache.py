"""
cache.py — 快取式評估

搜索空間加了 agentic 開關後變成 216 組，全掃 oracle 已不可行
(這正是需要最佳化器的理由)。改成「查過的不重跑」:
  evaluate_cached(cfg) -> 命中快取直接回；未命中才呼叫 API 評估並存檔。

random / OPRO / hybrid 共用同一個 cache.json，重疊到的配置零成本。

快取鍵 = 測試集 fingerprint + 評估指紋 + config:
- 不同 (n, n_hard, seed) 的測試集分數不可比，各自成一格，
  避免 Web UI 改題數後沿用到舊測試集的分數。
- 評估指紋含模型 id + 生成 prompt + EVAL_VERSION：換模型、改 prompt
  或評估邏輯改版後，舊分數自動失效，不會被無聲沿用。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from rag import RagConfig, GEN_MODEL, EMBED_MODEL, RERANK_MODEL, SYSTEM_PROMPT
from judge import JUDGE_MODEL

# Docker 等場景可用 CACHE_PATH 指到掛載的持久化目錄
CACHE_PATH = Path(os.environ.get("CACHE_PATH",
                                 Path(__file__).parent / "cache.json"))

# 搜索空間的全部維度 (RagConfig.key() / 快取鍵 / 軌跡去重共用這份定義)
CONFIG_FIELDS = ("chunk_size", "top_k", "retriever", "chunk_overlap",
                 "hybrid_alpha", "rerank", "query_decompose", "hyde",
                 "iterative", "compress", "parent_child", "verify")
_FIELDS = CONFIG_FIELDS   # 舊名

# 評估邏輯/指標「語意」改動時手動 +1 (例如改判官 prompt、改 recall 定義)，
# 讓舊 cache 全部失效。v5: compress/parent_child 進搜索空間 + hop 分層。
EVAL_VERSION = 5


def _eval_fingerprint() -> str:
    """模型 + prompt 版本指紋: 換模型或改 prompt 後，舊分數不可沿用。"""
    blob = "|".join([str(EVAL_VERSION), GEN_MODEL, EMBED_MODEL,
                     RERANK_MODEL, JUDGE_MODEL, SYSTEM_PROMPT])
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]


def dataset_key(n: int, n_hard: int, seed: int = 42) -> str:
    """測試集 fingerprint (load_hotpot 的參數) + 評估指紋。"""
    return f"hotpot|n={n}|hard={n_hard}|seed={seed}|ev={_eval_fingerprint()}"


def custom_dataset_key(fingerprint: str, n: int, seed: int = 42) -> str:
    """自訂資料集 (使用者文件+QA) 的快取鍵。
    fingerprint = dataset.dataset_fingerprint(語料, QA)，改內容自動失效。"""
    return f"custom|fp={fingerprint}|n={n}|seed={seed}|ev={_eval_fingerprint()}"


DEFAULT_DATASET = dataset_key(30, 15, 42)   # CLI (sweep/run/plot) 的預設測試集


def config_to_dict(cfg: RagConfig) -> dict:
    return {f: getattr(cfg, f) for f in _FIELDS}


def dict_to_config(d: dict) -> RagConfig:
    return RagConfig(**{f: d[f] for f in _FIELDS if f in d})


def _canonical(d: dict) -> dict:
    """等價配置正規化 (等價的配置共用同一筆快取分數，省真實評估):
    1) hybrid_alpha 只影響 hybrid/graph；bm25/dense 下映射回預設 0.5。
    2) RERANK_MODEL=dense 時，dense 檢索＋無分解的 rerank 是恆等變換，
       映射到 rerank=False。(預設用專用 reranker 模型時兩者真的不同。)"""
    if d.get("retriever") in ("bm25", "dense") and d.get("hybrid_alpha") != 0.5:
        d = dict(d, hybrid_alpha=0.5)
    if (RERANK_MODEL == "dense" and d.get("retriever") == "dense"
            and not d.get("query_decompose") and d.get("rerank")):
        d = dict(d, rerank=False)
    return d


def config_key(cfg_or_dict) -> str:
    d = config_to_dict(cfg_or_dict) if isinstance(cfg_or_dict, RagConfig) else cfg_or_dict
    d = _canonical(d)
    return "|".join(str(d[f]) for f in _FIELDS)


class ResultCache:
    def __init__(self, path: Path = CACHE_PATH, dataset: str = DEFAULT_DATASET):
        self.path = Path(path)
        self.dataset = dataset
        self.store: dict[str, dict] = {}
        if self.path.exists():
            self.store = json.loads(self.path.read_text(encoding="utf-8"))

    def _key(self, cfg: RagConfig) -> str:
        return f"{self.dataset}::{config_key(cfg)}"

    def get(self, cfg: RagConfig) -> dict | None:
        return self.store.get(self._key(cfg))

    def _save(self) -> None:
        # 寫暫存檔再原子替換: 中斷/並行寫入不會留下半截 JSON。
        # (CLI 與 server 同時跑仍可能互相覆蓋「新增的」條目，Demo 等級可接受。)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(self.store, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def evaluate_cached(self, cfg: RagConfig, examples, verbose: bool = False,
                        prune_below: float | None = None,
                        prune_lam: float = 0.5) -> dict:
        """回傳該配置的完整 record (命中快取則不呼叫 API)。
        評估失敗題過多時 evaluate 會拋出 -> 不落 cache (避免壞分數被永久快取)。
        prune_below 啟用 racing: 提早停止的配置回傳帶 pruned 標記的輕量 record，
        同樣不落 cache (沒有完整分數)。"""
        hit = self.get(cfg)
        if hit is not None:
            return hit
        from eval import evaluate, EvalPruned  # 局部 import，避免載入順序問題
        try:
            score = evaluate(cfg, examples, verbose=verbose,
                             prune_below=prune_below, prune_lam=prune_lam)
        except EvalPruned as p:
            return {"config": config_to_dict(cfg), "score": p.partial_score,
                    "pruned": True, "bound": p.bound,
                    "failures": [], "fail_clusters": {},
                    "per_question_correct": {}}
        record = {
            "config": config_to_dict(cfg),
            "score": score.as_dict(),
            "failures": [d.as_dict() for d in score.worst_cases(2)],
            "fail_clusters": score.fail_clusters(),
            # 逐題對錯 (question -> 0/1): 供 sweep.py 做成對顯著性檢定
            "per_question_correct": {d.question: d.correct for d in (score.details or [])},
        }
        self.store[self._key(cfg)] = record
        self._save()
        return record

    def all_records(self) -> list[dict]:
        """只回傳「同一個測試集」的 record — 不同題數/種子的分數不可混著比。"""
        prefix = f"{self.dataset}::"
        return [rec for k, rec in self.store.items() if k.startswith(prefix)]
