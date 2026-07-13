"""
rag.py — 被優化的 Agentic RAG 系統

被 OPRO 優化的配置 = RagConfig:
  數值/離散: chunk_size, chunk_overlap, top_k,
             retriever (bm25/dense/hybrid/graph), hybrid_alpha (dense 權重)
  agentic 開關:
    rerank          — 檢索後用專用 reranker 模型對候選重排，再取 top_k
    query_decompose — 多跳問題先拆成子問題各自檢索 (HotpotQA 是多跳)
    hyde            — 生成一句假設答案當額外查詢 (HyDE)
    iterative       — 看第一輪 context 缺什麼、追加查詢再檢索一輪
    verify          — 生成後做 NLI 忠實度自我檢查，不過關就重檢索/棄答 (降幻覺)
  graph 檢索 = graph-lite GraphRAG: 實體共現圖 + 種子鄰居擴展 (零建圖 API 成本)

所有模型呼叫走 NVIDIA NIM (OpenAI 相容)，本機不需要 GPU。
檢索是 per-question: HotpotQA 每題自帶 context，故可算 retrieval recall。
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np
import requests
from dotenv import load_dotenv
from openai import (APIConnectionError, APITimeoutError, InternalServerError,
                    OpenAI, RateLimitError)
from rank_bm25 import BM25Okapi

load_dotenv()  # 先讀 .env，下面模型 id 的環境變數覆寫才吃得到

# --- NVIDIA NIM 端點設定 ---------------------------------------------------
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
# 模型 id 可用環境變數覆寫 (端點 404/不穩時換模型，或換私有 OpenAI 相容端點)
GEN_MODEL = os.environ.get("GEN_MODEL", "meta/llama-3.1-8b-instruct")     # generator (便宜、量大)
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nvidia/llama-3.2-nv-embedqa-1b-v2")  # dense 檢索
# rerank 用專用 reranker 模型 (dense-cosine 重排在 dense 檢索下是恆等變換，
# rerank 開關會學不到東西)。設 RERANK_MODEL=dense 可退回舊行為；端點失效自動 fallback。
RERANK_MODEL = os.environ.get("RERANK_MODEL", "nvidia/llama-3.2-nv-rerankqa-1b-v2")
RERANK_URL = os.environ.get(
    "RERANK_URL",
    "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking")
# 註: model id 請到 build.nvidia.com 對最新清單; 介面是 OpenAI 相容的。

ABSTAIN = "I don't have enough evidence to answer."

# 生成器在 context 不足時會自行棄答 (SYSTEM_PROMPT 有指示)。不論 verify 開關，
# 棄答式回答都必須按同一規則計分 (見 eval.py)，否則 objective 會隱性補貼 verify。
# 中文語料下生成器常用中文棄答，pattern 必須雙語都涵蓋 (繁/簡)。
_ABSTAIN_PAT = re.compile(
    r"(don'?t|do not|doesn'?t) have enough (evidence|information)"
    r"|not enough (evidence|information|context)"
    r"|insufficient (evidence|information|context)"
    r"|cannot (answer|determine)|unable to (answer|determine)"
    r"|無法(回答|判斷|確定)|无法(回答|判断|确定)"
    r"|(沒有|没有)(足夠|足够)"
    r"|(資訊|資料|證據|信息|资料|证据)不足"
    r"|不足以(回答|判斷|判断)",
    re.IGNORECASE,
)


def is_abstain(answer: str) -> bool:
    """答案是否為棄答式回答 (verify 守門員的 ABSTAIN 或生成器自行棄答)。"""
    return answer.strip() == ABSTAIN or bool(_ABSTAIN_PAT.search(answer))


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "找不到 NVIDIA_API_KEY。請複製 .env.example 成 .env 並填入金鑰。"
        )
    # openai client 預設逾時 600s；搭配 api_call 的 5 次重試，一個真正掛掉/
    # 不回應的端點最壞情況要等將近 1 小時才報錯。降到 30s，讓我們自己的
    # 指數退避負責重試節奏，而不是卡在單次請求裡。
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key, timeout=30.0)


# NVIDIA 免費 tier 常回 429，暫時性錯誤一律指數退避重試
_RETRIABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)


def api_call(fn, *args, retries: int = 5, base_delay: float = 2.0, **kwargs):
    """呼叫 API；429 / 連線逾時 / 5xx 時指數退避重試，重試耗盡才拋出。"""
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except _RETRIABLE:
            if attempt == retries - 1:
                raise
            time.sleep(base_delay * (2 ** attempt))


# --- 配置 (OPRO 的搜索空間 θ) ---------------------------------------------
@dataclass(frozen=True)
class RagConfig:
    chunk_size: int = 512
    top_k: int = 5
    retriever: str = "hybrid"          # "bm25" | "dense" | "hybrid" | "graph"
    chunk_overlap: int = 0
    hybrid_alpha: float = 0.5          # hybrid/graph 的 dense 權重 (bm25 = 1-α)
    rerank: bool = False               # agentic: 檢索後重排
    query_decompose: bool = False      # agentic: 多跳拆解
    hyde: bool = False                 # agentic: 生成假設答案當額外查詢 (HyDE)
    iterative: bool = False            # agentic: 看缺什麼、再檢索一輪
    compress: bool = False             # agentic: 檢索後壓縮成與問題相關的句子
    parent_child: bool = False         # 小 chunk 檢索命中，餵生成器時換完整父段落
    verify: bool = False               # agentic: NLI 自我檢查守門員

    def key(self) -> tuple:
        return (self.chunk_size, self.top_k, self.retriever, self.chunk_overlap,
                self.hybrid_alpha, self.rerank, self.query_decompose,
                self.hyde, self.iterative, self.compress, self.parent_child,
                self.verify)


# --- 1. Chunking -----------------------------------------------------------
def make_chunks_with_parents(paragraphs: list[str],
                             cfg: RagConfig) -> tuple[list[str], list[int]]:
    """切 chunk 並記住每個 chunk 的父段落 index (parent_child 檢索用)。"""
    chunks: list[str] = []
    parents: list[int] = []
    step = max(1, cfg.chunk_size - cfg.chunk_overlap)
    for pi, para in enumerate(paragraphs):
        text = para.strip()
        if not text:
            continue
        if len(text) <= cfg.chunk_size:
            chunks.append(text)
            parents.append(pi)
            continue
        for start in range(0, len(text), step):
            piece = text[start:start + cfg.chunk_size].strip()
            if piece:
                chunks.append(piece)
                parents.append(pi)
    return chunks, parents


def make_chunks(paragraphs: list[str], cfg: RagConfig) -> list[str]:
    return make_chunks_with_parents(paragraphs, cfg)[0]


# --- 2. Retrievers ---------------------------------------------------------
def _tokenize(text: str) -> list[str]:
    return text.lower().split()


_LATIN_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\-]{3,}")
_CJK_RUN = re.compile(r"[一-鿿]{2,}")


def key_tokens(text: str) -> set[str]:
    """「稀有 token 候選」: 拉丁字 (≥4 字元) + CJK 連續片段的 2-gram。
    graph 檢索的實體共現邊與 dataset.py 的多跳 QA 配對共用這個定義。"""
    toks = {t.lower() for t in _LATIN_TOKEN.findall(text)}
    for run in _CJK_RUN.findall(text):
        toks.update(run[i:i + 2] for i in range(len(run) - 1))
    return toks


# 進程級 embedding 快取: chunk 只由 chunk_size/overlap 決定，同 chunk_size 的
# 所有配置 chunk 文字完全相同 —— 不快取的話每個配置都重打一輪 embedding API。
# 命中時把「首次呼叫按字元比例攤提」的 token 成本記回，讓 avg_tokens 仍反映
# 該配置獨立執行的成本，不因評估順序而失真。
_EMBED_BATCH = 32   # NIM embedding 端點有單次 batch 上限，分批送
_embed_cache: dict[str, tuple[np.ndarray, int]] = {}   # key -> (向量, 攤提 token)


def _embed_key(text: str, input_type: str) -> str:
    return input_type + ":" + hashlib.sha1(text.encode("utf-8")).hexdigest()


def _embed(texts: list[str], input_type: str) -> tuple[np.ndarray, int]:
    """NVIDIA embedding。input_type 必為 query/passage (nv-embedqa 強制)。
    回傳 (向量, 攤提的 token 成本)；快取命中不打 API。"""
    keys = [_embed_key(t, input_type) for t in texts]
    todo: dict[str, str] = {}   # 未快取的 key -> 文字 (順便去重)
    for k, t in zip(keys, texts):
        if k not in _embed_cache and k not in todo:
            todo[k] = t
    if todo:
        client = get_client()
        todo_keys = list(todo)
        for s in range(0, len(todo_keys), _EMBED_BATCH):
            batch_keys = todo_keys[s:s + _EMBED_BATCH]
            batch_texts = [todo[k] for k in batch_keys]
            resp = api_call(
                client.embeddings.create,
                model=EMBED_MODEL,
                input=batch_texts,
                extra_body={"input_type": input_type, "truncate": "END"},
            )
            tok = resp.usage.total_tokens if getattr(resp, "usage", None) else 0
            total_chars = sum(len(t) for t in batch_texts) or 1
            for k, t, d in zip(batch_keys, batch_texts, resp.data):
                vec = np.asarray(d.embedding, dtype=np.float32)
                _embed_cache[k] = (vec, round(tok * len(t) / total_chars))
    vecs = np.stack([_embed_cache[k][0] for k in keys]).astype(float)
    tokens = sum(_embed_cache[k][1] for k in keys)
    return vecs, tokens


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


class ChunkIndex:
    """單題內重用的檢索索引。

    BM25 index 與 chunk embeddings 各只建一次；query_decompose 的多個子問題、
    verify 守門員的二次檢索、rerank 都重用同一份。chunk 向量以「索引」為單位
    memo：bm25+rerank 只需 embed 候選子集，不再為重排少數候選 embed 全部 chunk。
    (跨題/跨配置的相同文字另有進程級 _embed_cache，不重打 API。)
    """

    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self._bm25: BM25Okapi | None = None
        self._doc_norm: dict[int, np.ndarray] = {}    # idx -> 正規化 chunk 向量
        self._q_cache: dict[str, np.ndarray] = {}     # 正規化後的 query 向量
        self._adj: dict[int, set[int]] | None = None  # graph-lite 實體共現邊
        self.embed_tokens = 0                          # 成本: embedding token 累計
        self.extra_tokens = 0                          # 成本: reranker token (估算)

    def bm25_scores(self, query: str) -> np.ndarray:
        if self._bm25 is None:
            self._bm25 = BM25Okapi([_tokenize(c) for c in self.chunks])
        return np.asarray(self._bm25.get_scores(_tokenize(query)), dtype=float)

    def _doc_matrix(self, indices: list[int]) -> np.ndarray:
        missing = [i for i in indices if i not in self._doc_norm]
        if missing:
            vecs, tok = _embed([self.chunks[i] for i in missing], "passage")
            self.embed_tokens += tok
            for i, v in zip(missing, vecs):
                self._doc_norm[i] = v / (np.linalg.norm(v) + 1e-8)
        return np.stack([self._doc_norm[i] for i in indices])

    def _query_vector(self, query: str) -> np.ndarray:
        if query not in self._q_cache:
            vec, tok = _embed([query], "query")
            self.embed_tokens += tok
            v = vec[0]
            self._q_cache[query] = v / (np.linalg.norm(v) + 1e-8)
        return self._q_cache[query]

    def dense_scores(self, query: str, subset: list[int] | None = None) -> np.ndarray:
        indices = list(subset) if subset is not None else list(range(len(self.chunks)))
        return self._doc_matrix(indices) @ self._query_vector(query)

    def _fused(self, query: str, alpha: float) -> np.ndarray:
        """hybrid: (1-α)·bm25 + α·dense。α=0.5 等價舊版等權重 (外層會再 minmax)。
        α 極值時短路，省掉不需要的另一路 (α=0 連 embedding 都不用打)。"""
        if alpha <= 0.0:
            return self.bm25_scores(query)
        if alpha >= 1.0:
            return self.dense_scores(query)
        return ((1 - alpha) * _minmax(self.bm25_scores(query))
                + alpha * _minmax(self.dense_scores(query)))

    def _graph(self) -> dict[int, set[int]]:
        """graph-lite 實體共現圖: 兩 chunk 共享「稀有 token」(只出現在少數
        chunk 的實體/詞) 就連邊。純字面統計，不花任何 LLM/embedding 呼叫。"""
        if self._adj is None:
            tok_chunks: dict[str, list[int]] = {}
            for i, c in enumerate(self.chunks):
                for t in key_tokens(c):
                    tok_chunks.setdefault(t, []).append(i)
            cap = max(3, len(self.chunks) // 10)   # 高頻詞不當實體 (hub 邊爆炸)
            adj: dict[int, set[int]] = {i: set() for i in range(len(self.chunks))}
            for idxs in tok_chunks.values():
                if 2 <= len(idxs) <= cap:
                    for a in idxs:
                        adj[a].update(b for b in idxs if b != a)
            self._adj = adj
        return self._adj

    def graph_scores(self, query: str, alpha: float, seed_k: int,
                     gamma: float = 0.5) -> np.ndarray:
        """graph 檢索: hybrid 分數選種子，沿實體共現邊把「橋接 chunk」
        (與高分種子共享實體、但自身字面/語意分數低) 拉上來。
        多跳題的第二跳段落常是這種 —— 這正是 GraphRAG 的核心紅利。"""
        base = _minmax(self._fused(query, alpha))
        adj = self._graph()
        boosted = base.copy()
        for s in np.argsort(base)[::-1][:max(1, seed_k)]:
            for nb in adj.get(int(s), ()):
                boosted[nb] = max(boosted[nb], base[nb] + gamma * base[int(s)])
        return boosted

    def scores(self, query: str, retriever: str, alpha: float = 0.5,
               seed_k: int = 5) -> np.ndarray:
        if retriever == "bm25":
            return self.bm25_scores(query)
        if retriever == "dense":
            return self.dense_scores(query)
        if retriever == "hybrid":
            return self._fused(query, alpha)
        if retriever == "graph":
            return self.graph_scores(query, alpha, seed_k)
        raise ValueError(f"未知 retriever: {retriever}")


# --- 3. Query 分解 (agentic) ----------------------------------------------
def _clean_subquestion(line: str) -> str:
    """剝掉 bullet 與 '1.' / '2)' 這類編號 (模型不一定聽話)。"""
    return re.sub(r"^\s*(?:\d+\s*[.)]|[-•*])\s*", "", line).strip()


def decompose_query(question: str) -> tuple[list[str], int]:
    """把多跳問題拆成 1~3 個子問題。回傳 (子問題列表, 用掉的 token)。"""
    client = get_client()
    prompt = (
        "Break the following multi-hop question into the minimal list of simpler "
        "sub-questions needed to answer it. Output each sub-question on its own line, "
        "no numbering. If it is already simple, output it unchanged.\n\n"
        f"Question: {question}"
    )
    resp = api_call(
        client.chat.completions.create,
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=128,
    )
    tokens = resp.usage.total_tokens if resp.usage else 0
    subs = [_clean_subquestion(ln) for ln in resp.choices[0].message.content.splitlines()]
    subs = [s for s in subs if s]
    return (subs[:3] or [question]), tokens


def hyde_query(question: str) -> tuple[str, int]:
    """HyDE (agentic): 生成一句「假設答案」當額外查詢。
    假設答案的用詞比問題更貼近文件語言，dense 檢索特別受益。"""
    client = get_client()
    prompt = (
        "Write ONE short hypothetical sentence that would plausibly answer the "
        "question, phrased as if quoted from a reference document. Output only "
        "that sentence, in the same language as the question.\n\n"
        f"Question: {question}"
    )
    resp = api_call(
        client.chat.completions.create,
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=60,
    )
    tokens = resp.usage.total_tokens if resp.usage else 0
    return resp.choices[0].message.content.strip(), tokens


def gap_query(question: str, chunks: list[str]) -> tuple[str | None, int]:
    """iterative (agentic): 看第一輪 context 還缺什麼資訊。
    回傳追加查詢；context 已足夠時回 None (不做第二輪)。"""
    client = get_client()
    context = "\n".join(f"- {c[:300]}" for c in chunks)
    prompt = (
        "You will answer the QUESTION using the CONTEXT. If the context already "
        "contains all information needed, reply exactly NONE. Otherwise reply "
        "with ONE short search query (same language as the question) that would "
        "find the missing information.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\nReply:"
    )
    resp = api_call(
        client.chat.completions.create,
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=48,
    )
    tokens = resp.usage.total_tokens if resp.usage else 0
    text = resp.choices[0].message.content.strip()
    if not text or text.upper().startswith("NONE"):
        return None, tokens
    return text.splitlines()[0].strip(), tokens


# --- 4. Rerank (專用 reranker 模型，失效退回 dense) --------------------------
_rerank_disabled = False   # 端點失效時本進程退回 dense rerank (只警告一次)


def _rerank_api(query: str, passages: list[str]) -> list[float] | None:
    """NIM reranker 端點 (非 OpenAI 介面)。回傳各 passage 分數；
    不可用時回 None，呼叫端退回 dense-cosine 重排。429/5xx 指數退避。"""
    global _rerank_disabled
    if _rerank_disabled or RERANK_MODEL == "dense":
        return None
    headers = {"Authorization": f"Bearer {os.environ.get('NVIDIA_API_KEY', '')}",
               "Accept": "application/json"}
    body = {"model": RERANK_MODEL, "query": {"text": query},
            "passages": [{"text": p} for p in passages], "truncate": "END"}
    for attempt in range(4):
        try:
            r = requests.post(RERANK_URL, headers=headers, json=body, timeout=60)
            if r.status_code in (404, 410):
                # 永久性錯誤 (端點/模型不存在或已除役) -> 立刻停損，不浪費重試
                print(f"[warn] reranker 端點 HTTP {r.status_code} (永久性)，"
                     "本進程退回 dense rerank")
                _rerank_disabled = True
                return None
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2.0 * (2 ** attempt))
                continue
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            scores = [0.0] * len(passages)
            for item in r.json()["rankings"]:
                scores[item["index"]] = float(item["logit"])
            return scores
        except Exception as e:  # noqa: BLE001 — reranker 掛掉不該讓整輪評估死掉
            if attempt == 3:
                print(f"[warn] reranker 端點不可用，本進程退回 dense rerank: {e}")
                _rerank_disabled = True
                return None
    return None


# --- 5. 檢索 (含多查詢合併 + 可選 rerank) ---------------------------------
def retrieve(index: ChunkIndex, queries: list[str], cfg: RagConfig) -> list[int]:
    """對 (一或多個) query 檢索，合併候選，可選 rerank，回傳 top_k chunk index。"""
    chunks = index.chunks
    if not chunks:
        return []
    # 多 query: 每個 idx 取跨子問題的最高分
    best = np.full(len(chunks), -np.inf)
    for q in queries:
        best = np.maximum(best, _minmax(index.scores(
            q, cfg.retriever, alpha=cfg.hybrid_alpha, seed_k=cfg.top_k)))

    k = min(cfg.top_k, len(chunks))
    if not cfg.rerank:
        return np.argsort(best)[::-1][:k].tolist()

    use_api = RERANK_MODEL != "dense" and not _rerank_disabled
    # dense 檢索 + 單 query 時，dense-cosine 重排與不重排「完全等價」(同一分數
    # 函數重新排序)；沒有 reranker 模型可用時直接跳過，省一次假動作
    if not use_api and cfg.retriever == "dense" and len(queries) == 1:
        return np.argsort(best)[::-1][:k].tolist()

    # 先取較寬的候選，再對「原始問題」重排取 top_k
    wide = min(max(k * 2, k + 3), len(chunks))
    cand = np.argsort(best)[::-1][:wide].tolist()
    cand_texts = [chunks[i] for i in cand]
    scores = _rerank_api(queries[0], cand_texts) if use_api else None
    if scores is not None:
        # reranker 端點不回 usage，token 成本以 chars/4 估算
        index.extra_tokens += sum(len(t) for t in cand_texts + [queries[0]]) // 4
        rr = np.asarray(scores, dtype=float)
    else:
        rr = index.dense_scores(queries[0], subset=cand)
    order = np.argsort(rr)[::-1][:k]
    return [cand[i] for i in order]


# --- 6. Generation ---------------------------------------------------------
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
    resp = api_call(
        client.chat.completions.create,
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


_CITE_SUFFIX = (" After the answer, cite the numbers of the context passages "
                "that support it, in square brackets, e.g. [1][3].")


def generate_cited(query: str, retrieved_chunks: list[str]) -> tuple[str, list[int], int]:
    """帶引用標註的生成 (Playground 用；評估管線不用，以免污染 EM/F1)。
    回傳 (答案含 [n] 標記, 有效引用的 chunk 編號, tokens)。"""
    client = get_client()
    context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(retrieved_chunks))
    user_msg = f"Context:\n{context}\n\nQuestion: {query}\nShort answer:"
    resp = api_call(
        client.chat.completions.create,
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT + _CITE_SUFFIX},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=96,
    )
    tokens = resp.usage.total_tokens if resp.usage else 0
    answer = resp.choices[0].message.content.strip()
    cites = sorted({int(m) for m in re.findall(r"\[(\d+)\]", answer)
                    if 1 <= int(m) <= len(retrieved_chunks)})
    return answer, cites, tokens


def strip_citations(answer: str) -> str:
    """拿掉 [n] 引用標記 (餵 NLI 判官 / 棄答偵測前用，避免干擾判定)。"""
    return re.sub(r"\s*\[\d+\]", "", answer).strip()


def compress_chunks(question: str, chunks: list[str]) -> tuple[list[str], int]:
    """compress (agentic): 把檢索到的 chunk 壓成「與問題相關的原句」再餵生成器
    —— 降 token 成本、升信噪比 (與 μ 成本懲罰互補: 最佳化器可學會用
    compress 換更大的 top_k)。壓到全空時退回原文，別讓生成器沒 context。"""
    client = get_client()
    context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(chunks))
    prompt = (
        "From the CONTEXT passages, copy VERBATIM only the sentences that are "
        "relevant to answering the QUESTION. Do not paraphrase, do not add "
        "anything, omit passages with nothing relevant.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\nRelevant sentences:"
    )
    resp = api_call(
        client.chat.completions.create,
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=512,
    )
    tokens = resp.usage.total_tokens if resp.usage else 0
    text = resp.choices[0].message.content.strip()
    return ([text] if text else chunks), tokens


# --- 7. 完整 pipeline ------------------------------------------------------
@dataclass
class RagResult:
    answer: str
    retrieved_idx: list[int] = field(default_factory=list)
    retrieved_chunks: list[str] = field(default_factory=list)
    faithful: float | None = None      # verify 開啟時的 NLI 結果 (1/0)，否則 None
    abstained: bool = False            # verify 守門員最終棄答
    citations: list[int] = field(default_factory=list)   # cite=True 時的引用編號
    tokens: int = 0                    # 成本: 累計 token (生成+分解+embedding+守門員判官)
    latency: float = 0.0               # 成本: 秒


def run_rag(question: str, paragraphs: list[str], cfg: RagConfig,
            cite: bool = False) -> RagResult:
    """單題完整流程:
    (decompose)(hyde) -> retrieve(+rerank) -> (iterative 缺口再檢索)
    -> (parent_child 換父段落) -> (compress) -> generate -> (verify)。
    cite=True 時生成器附 [n] 引用 (Playground 用；評估管線一律 False)。"""
    t0 = time.perf_counter()
    tokens = 0

    queries = [question]
    if cfg.query_decompose:
        queries, dt = decompose_query(question)
        tokens += dt
    if cfg.hyde:
        hq, ht = hyde_query(question)
        tokens += ht
        if hq:
            queries = queries + [hq]   # 與原問題/子問題 max-融合

    chunks, parents = make_chunks_with_parents(paragraphs, cfg)
    index = ChunkIndex(chunks)

    def to_context(sel: list[int]) -> list[str]:
        """parent_child: 小 chunk 檢索命中，餵生成器/判官時換成完整父段落 (去重)
        —— 檢索精準命中、生成拿到完整上下文，兩頭都要。"""
        if not cfg.parent_child:
            return [chunks[i] for i in sel]
        seen, out = set(), []
        for i in sel:
            p = parents[i]
            if p not in seen:
                seen.add(p)
                out.append(paragraphs[p].strip())
        return out

    def make_answer(context: list[str]) -> tuple[str, list[int]]:
        nonlocal tokens
        gen_ctx = context
        if cfg.compress and context:
            gen_ctx, ct = compress_chunks(question, context)
            tokens += ct
        if cite:
            ans, cites, gt = generate_cited(question, gen_ctx)
            if cfg.compress:
                cites = []   # 壓縮後編號對不回原 chunk，不提供錯誤的高亮
        else:
            ans, gt = generate(question, gen_ctx)
            cites = []
        tokens += gt
        return ans, cites

    idx = retrieve(index, queries, cfg)

    if cfg.iterative and idx:
        # 第一輪 context 缺什麼 -> 追加查詢再檢索，合併 (上限與 verify 拓寬一致)
        gq, gt0 = gap_query(question, to_context(idx))
        tokens += gt0
        if gq:
            merged = list(idx)
            merged += [i for i in retrieve(index, [gq], cfg) if i not in merged]
            idx = merged[:min(cfg.top_k + 3, len(chunks))]

    picked = to_context(idx)
    answer, citations = make_answer(picked)

    faithful = None
    abstained = False
    if cfg.verify:
        # 局部 import 避免 rag <-> judge 循環引用
        from judge import judge_faithfulness
        faithful, jt = judge_faithfulness(strip_citations(answer), picked)
        tokens += jt
        if faithful < 1.0:
            # 守門員觸發: 拓寬檢索再答一次
            wider = replace(cfg, top_k=min(cfg.top_k + 3, len(chunks)), verify=False)
            idx = retrieve(index, queries, wider)
            picked = to_context(idx)
            answer, citations = make_answer(picked)
            faithful, jt = judge_faithfulness(strip_citations(answer), picked)
            tokens += jt
            if faithful < 1.0:
                # 仍不忠實 -> 棄答，寧可不答也不要幻覺。
                # 棄答沒有捏造內容，faithful 記 1.0 (答錯由 correctness 懲罰即可)，
                # 否則 objective 會雙重懲罰、反而教壞 meta-optimizer「別開 verify」。
                answer = ABSTAIN
                abstained = True
                faithful = 1.0
                citations = []

    tokens += index.embed_tokens + index.extra_tokens
    return RagResult(answer=answer, retrieved_idx=idx, retrieved_chunks=picked,
                     faithful=faithful, abstained=abstained, citations=citations,
                     tokens=tokens, latency=time.perf_counter() - t0)


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
