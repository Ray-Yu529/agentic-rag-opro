"""
rag.py — 被優化的 Agentic RAG 系統

被 OPRO 優化的配置 = RagConfig:
  數值/離散: chunk_size, top_k, retriever
  agentic 開關:
    rerank          — 檢索後用 dense 相似度對候選重排，再取 top_k
    query_decompose — 多跳問題先拆成子問題各自檢索 (HotpotQA 是多跳)
    verify          — 生成後做 NLI 忠實度自我檢查，不過關就重檢索/棄答 (降幻覺)

所有模型呼叫走 NVIDIA NIM (OpenAI 相容)，本機不需要 GPU。
檢索是 per-question: HotpotQA 每題自帶 context，故可算 retrieval recall。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi

# --- NVIDIA NIM 端點設定 ---------------------------------------------------
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
GEN_MODEL = "meta/llama-3.1-8b-instruct"            # RAG generator (便宜、量大)
EMBED_MODEL = "nvidia/llama-3.2-nv-embedqa-1b-v2"   # dense 檢索 / rerank 用
# 註: model id 請到 build.nvidia.com 對最新清單; 介面是 OpenAI 相容的。

ABSTAIN = "I don't have enough evidence to answer."


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "找不到 NVIDIA_API_KEY。請複製 .env.example 成 .env 並填入金鑰。"
        )
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)


# --- 配置 (OPRO 的搜索空間 θ) ---------------------------------------------
@dataclass(frozen=True)
class RagConfig:
    chunk_size: int = 512
    top_k: int = 5
    retriever: str = "hybrid"          # "bm25" | "dense" | "hybrid"
    chunk_overlap: int = 0
    rerank: bool = False               # agentic: 檢索後重排
    query_decompose: bool = False      # agentic: 多跳拆解
    verify: bool = False               # agentic: NLI 自我檢查守門員

    def key(self) -> tuple:
        return (self.chunk_size, self.top_k, self.retriever, self.chunk_overlap,
                self.rerank, self.query_decompose, self.verify)


# --- 1. Chunking -----------------------------------------------------------
def make_chunks(paragraphs: list[str], cfg: RagConfig) -> list[str]:
    chunks: list[str] = []
    step = max(1, cfg.chunk_size - cfg.chunk_overlap)
    for para in paragraphs:
        text = para.strip()
        if not text:
            continue
        if len(text) <= cfg.chunk_size:
            chunks.append(text)
            continue
        for start in range(0, len(text), step):
            piece = text[start:start + cfg.chunk_size].strip()
            if piece:
                chunks.append(piece)
    return chunks


# --- 2. Retrievers ---------------------------------------------------------
def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def _bm25_scores(chunks: list[str], query: str) -> np.ndarray:
    bm25 = BM25Okapi([_tokenize(c) for c in chunks])
    return np.asarray(bm25.get_scores(_tokenize(query)), dtype=float)


def _embed(texts: list[str], input_type: str) -> np.ndarray:
    """NVIDIA embedding。input_type 必為 query/passage (nv-embedqa 強制)。"""
    client = get_client()
    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        extra_body={"input_type": input_type, "truncate": "END"},
    )
    return np.asarray([d.embedding for d in resp.data], dtype=float)


def _dense_scores(chunks: list[str], query: str) -> np.ndarray:
    doc_vecs = _embed(chunks, "passage")
    q_vec = _embed([query], "query")[0]
    doc_norm = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-8)
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-8)
    return doc_norm @ q_norm


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _retriever_scores(chunks: list[str], query: str, retriever: str) -> np.ndarray:
    if retriever == "bm25":
        return _bm25_scores(chunks, query)
    if retriever == "dense":
        return _dense_scores(chunks, query)
    if retriever == "hybrid":
        return _minmax(_bm25_scores(chunks, query)) + _minmax(_dense_scores(chunks, query))
    raise ValueError(f"未知 retriever: {retriever}")


# --- 3. Query 分解 (agentic) ----------------------------------------------
def decompose_query(question: str) -> tuple[list[str], int]:
    """把多跳問題拆成 1~3 個子問題。回傳 (子問題列表, 用掉的 token)。"""
    client = get_client()
    prompt = (
        "Break the following multi-hop question into the minimal list of simpler "
        "sub-questions needed to answer it. Output each sub-question on its own line, "
        "no numbering. If it is already simple, output it unchanged.\n\n"
        f"Question: {question}"
    )
    resp = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=128,
    )
    tokens = resp.usage.total_tokens if resp.usage else 0
    subs = [ln.strip(" -•\t") for ln in resp.choices[0].message.content.splitlines()]
    subs = [s for s in subs if s]
    return (subs[:3] or [question]), tokens


# --- 4. 檢索 (含多查詢合併 + 可選 rerank) ---------------------------------
def retrieve(chunks: list[str], queries: list[str], cfg: RagConfig) -> list[int]:
    """對 (一或多個) query 檢索，合併候選，可選 rerank，回傳 top_k chunk index。"""
    if not chunks:
        return []
    # 多 query: 每個 idx 取跨子問題的最高分
    best = np.full(len(chunks), -np.inf)
    for q in queries:
        best = np.maximum(best, _minmax(_retriever_scores(chunks, q, cfg.retriever)))

    k = min(cfg.top_k, len(chunks))
    if not cfg.rerank:
        return np.argsort(best)[::-1][:k].tolist()

    # rerank: 先取較寬的候選，再用 dense 對「原始問題」重排取 top_k
    wide = min(max(k * 2, k + 3), len(chunks))
    cand = np.argsort(best)[::-1][:wide].tolist()
    rr = _dense_scores([chunks[i] for i in cand], queries[0])
    order = np.argsort(rr)[::-1][:k]
    return [cand[i] for i in order]


# --- 5. Generation ---------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. "
    "Answer ONLY using the provided context. "
    "Give the shortest exact answer (a name, entity, yes/no, or short phrase). "
    "Do not explain. If the context is insufficient, say you don't have enough evidence."
)


def generate(query: str, retrieved_chunks: list[str]) -> tuple[str, int]:
    client = get_client()
    context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(retrieved_chunks))
    user_msg = f"Context:\n{context}\n\nQuestion: {query}\nShort answer:"
    resp = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=64,
    )
    tokens = resp.usage.total_tokens if resp.usage else 0
    return resp.choices[0].message.content.strip(), tokens


# --- 6. 完整 pipeline ------------------------------------------------------
@dataclass
class RagResult:
    answer: str
    retrieved_idx: list[int] = field(default_factory=list)
    retrieved_chunks: list[str] = field(default_factory=list)
    faithful: float | None = None      # verify 開啟時的 NLI 結果 (1/0)，否則 None
    tokens: int = 0                    # 成本: 累計 token
    latency: float = 0.0               # 成本: 秒


def run_rag(question: str, paragraphs: list[str], cfg: RagConfig) -> RagResult:
    """單題完整流程: (decompose) -> retrieve(+rerank) -> generate -> (verify)。"""
    t0 = time.perf_counter()
    tokens = 0

    queries = [question]
    if cfg.query_decompose:
        queries, dt = decompose_query(question)
        tokens += dt

    chunks = make_chunks(paragraphs, cfg)
    idx = retrieve(chunks, queries, cfg)
    picked = [chunks[i] for i in idx]
    answer, gt = generate(question, picked)
    tokens += gt

    faithful = None
    if cfg.verify:
        # 局部 import 避免 rag <-> judge 循環引用
        from judge import judge_faithfulness
        faithful = judge_faithfulness(answer, picked)
        if faithful < 1.0:
            # 守門員觸發: 拓寬檢索再答一次
            wider = RagConfig(**{**cfg.__dict__, "top_k": min(cfg.top_k + 3, len(chunks)),
                                 "rerank": cfg.rerank, "verify": False})
            idx = retrieve(chunks, queries, wider)
            picked = [chunks[i] for i in idx]
            answer, gt = generate(question, picked)
            tokens += gt
            faithful = judge_faithfulness(answer, picked)
            if faithful < 1.0:
                answer = ABSTAIN  # 仍不忠實 -> 棄答，寧可不答也不要幻覺

    return RagResult(answer=answer, retrieved_idx=idx, retrieved_chunks=picked,
                     faithful=faithful, tokens=tokens,
                     latency=time.perf_counter() - t0)


if __name__ == "__main__":
    demo_paras = [
        "The Eiffel Tower is located in Paris, France. It was completed in 1889.",
        "Mount Everest is the highest mountain on Earth, located in the Himalayas.",
        "The Great Wall of China is over 13,000 miles long.",
    ]
    cfg = RagConfig(chunk_size=256, top_k=2, retriever="hybrid", verify=True)
    res = run_rag("Where is the Eiffel Tower located?", demo_paras, cfg)
    print("檢索 index:", res.retrieved_idx, "| 忠實:", res.faithful,
          "| tokens:", res.tokens, f"| {res.latency:.2f}s")
    print("答案:", res.answer)
