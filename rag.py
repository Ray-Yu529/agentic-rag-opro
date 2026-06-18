"""
rag.py — 被優化的 Agentic RAG 系統 (Demo 版)

設計重點:
- 所有模型呼叫走 NVIDIA NIM (OpenAI 相容)，本機不需要 GPU。
- 檢索是「per-question」: HotpotQA distractor 每題自帶 10 段 context，
  我們在這 10 段上做 chunk + 檢索，這樣才能算 retrieval recall。
- 三種 retriever (bm25 / dense / hybrid) 可由 config 切換，這就是 OPRO 要調的東西。

被 OPRO 優化的配置 = RagConfig (chunk_size, top_k, retriever)。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from openai import OpenAI
from rank_bm25 import BM25Okapi

# --- NVIDIA NIM 端點設定 ---------------------------------------------------
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
GEN_MODEL = "meta/llama-3.1-8b-instruct"            # RAG generator (便宜、量大)
EMBED_MODEL = "nvidia/llama-3.2-nv-embedqa-1b-v2"   # dense 檢索用
# 註: model id 請到 build.nvidia.com 對最新清單; 介面是 OpenAI 相容的。


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """單例 client。金鑰從環境變數 NVIDIA_API_KEY 讀 (見 .env.example)。"""
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "找不到 NVIDIA_API_KEY。請複製 .env.example 成 .env 並填入金鑰，"
            "或在環境變數設定。"
        )
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)


# --- 配置 (這就是 OPRO 的搜索空間 θ) --------------------------------------
@dataclass(frozen=True)
class RagConfig:
    chunk_size: int = 512          # 每個 chunk 的字元數
    top_k: int = 5                 # 檢索回傳的 chunk 數
    retriever: str = "hybrid"      # "bm25" | "dense" | "hybrid"
    chunk_overlap: int = 0         # Demo 先固定 0，行有餘力再開放

    def key(self) -> tuple:
        """用來在軌跡庫去重的唯一鍵。"""
        return (self.chunk_size, self.top_k, self.retriever, self.chunk_overlap)


# --- 1. Chunking -----------------------------------------------------------
def make_chunks(paragraphs: list[str], cfg: RagConfig) -> list[str]:
    """
    把 context 段落切成固定大小的 chunk。
    paragraphs: 該題的 context 段落 (HotpotQA 每段是 title + 句子串起來的文字)。
    回傳: chunk 文字 list。
    """
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
    """
    呼叫 NVIDIA embedding。input_type 必須是 "query" 或 "passage"
    (nv-embedqa 系列強制要求，否則品質會掉)。
    """
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
    # cosine 相似度 (embedding 已大致正規化，仍除一次保險)
    doc_norm = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-8)
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-8)
    return doc_norm @ q_norm


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def retrieve(chunks: list[str], query: str, cfg: RagConfig) -> list[int]:
    """回傳 top_k 個 chunk 的 index (由高分到低分)。"""
    if not chunks:
        return []
    if cfg.retriever == "bm25":
        scores = _bm25_scores(chunks, query)
    elif cfg.retriever == "dense":
        scores = _dense_scores(chunks, query)
    elif cfg.retriever == "hybrid":
        # 兩者 min-max 正規化後相加 (簡單有效，Demo 夠用)
        scores = _minmax(_bm25_scores(chunks, query)) + _minmax(_dense_scores(chunks, query))
    else:
        raise ValueError(f"未知 retriever: {cfg.retriever}")
    k = min(cfg.top_k, len(chunks))
    return np.argsort(scores)[::-1][:k].tolist()


# --- 3. Generation ---------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a precise question-answering assistant. "
    "Answer ONLY using the provided context. "
    "Give the shortest exact answer (a name, entity, yes/no, or short phrase). "
    "Do not explain. If the context is insufficient, answer with your best guess from the context."
)


def generate(query: str, retrieved_chunks: list[str]) -> str:
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
    return resp.choices[0].message.content.strip()


# --- 4. 完整 pipeline ------------------------------------------------------
@dataclass
class RagResult:
    answer: str
    retrieved_idx: list[int] = field(default_factory=list)
    retrieved_chunks: list[str] = field(default_factory=list)


def run_rag(question: str, paragraphs: list[str], cfg: RagConfig) -> RagResult:
    """單題跑完整流程: chunk -> retrieve -> generate。"""
    chunks = make_chunks(paragraphs, cfg)
    idx = retrieve(chunks, question, cfg)
    picked = [chunks[i] for i in idx]
    answer = generate(question, picked)
    return RagResult(answer=answer, retrieved_idx=idx, retrieved_chunks=picked)


if __name__ == "__main__":
    # 煙霧測試: 不連 HotpotQA，先確認 API 通不通。
    demo_paras = [
        "The Eiffel Tower is located in Paris, France. It was completed in 1889.",
        "Mount Everest is the highest mountain on Earth, located in the Himalayas.",
        "The Great Wall of China is over 13,000 miles long.",
    ]
    cfg = RagConfig(chunk_size=256, top_k=2, retriever="hybrid")
    res = run_rag("Where is the Eiffel Tower located?", demo_paras, cfg)
    print("檢索到的 chunk index:", res.retrieved_idx)
    print("答案:", res.answer)
