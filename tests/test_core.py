"""核心指標/快取/最佳化邏輯的離線測試 (不打 API)。
這個專案的價值就在指標語意的正確性 —— 改動時這些測試必須維持綠燈。"""

import numpy as np
import pytest

import rag
from rag import ABSTAIN, RagConfig, is_abstain, make_chunks

# 完整 10 維配置 dict (SEARCH_SPACE 全欄位)，供提案/快取鍵測試用
FULL_CFG = {"chunk_size": 512, "top_k": 5, "retriever": "hybrid",
            "chunk_overlap": 0, "hybrid_alpha": 0.5, "rerank": True,
            "query_decompose": False, "hyde": False, "iterative": False,
            "verify": False}


# --- 棄答偵測 (雙語) ----------------------------------------------------------
@pytest.mark.parametrize("ans", [
    ABSTAIN,
    "I don't have enough evidence to answer.",
    "I do not have enough information to answer this question.",
    "There is insufficient evidence in the context.",
    "I cannot determine the answer from the given context.",
    "根據提供的內容，我無法回答這個問題。",
    "資訊不足，無法確定答案。",
    "沒有足夠的證據支持任何答案。",
    "根据上下文，无法判断。",
    "信息不足。",
])
def test_is_abstain_true(ans):
    assert is_abstain(ans)


@pytest.mark.parametrize("ans", [
    "Paris", "no", "Yes, the Eiffel Tower was completed in 1889.",
    "巴黎鐵塔", "1889 年", "答案是聖母院。",
])
def test_is_abstain_false(ans):
    assert not is_abstain(ans)


# --- chunking -----------------------------------------------------------------
def test_make_chunks():
    cfg = RagConfig(chunk_size=10, chunk_overlap=0)
    assert make_chunks(["abcdefghijklmnop"], cfg) == ["abcdefghij", "klmnop"]


# --- EM / F1 (含 CJK) ----------------------------------------------------------
def test_cjk_metrics():
    from eval import exact_match, f1_score, normalize_answer
    # CJK 去空白: "1889 年" == "1889年"
    assert exact_match("1889 年", "1889年") == 1.0
    # CJK 標點被正規化掉
    assert normalize_answer("巴黎，鐵塔。") == "巴黎鐵塔"
    # 字元級 F1: 部分重疊要有非零分數 (空白斷詞下整句中文會變 0/1)
    f1 = f1_score("法國巴黎鐵塔", "巴黎鐵塔")
    assert 0.0 < f1 < 1.0
    # 英文行為不變
    assert f1_score("the Eiffel Tower", "Eiffel Tower") == 1.0


# --- retrieval recall (覆蓋率) --------------------------------------------------
def test_retrieval_recall_coverage():
    from eval import retrieval_recall
    gold = "x" * 1000
    assert retrieval_recall([gold[:256]], [gold]) == 0.0   # 皮毛不算命中
    assert retrieval_recall([gold[:600]], [gold]) == 1.0   # 覆蓋 60% 算命中
    assert retrieval_recall([gold], [gold]) == 1.0
    assert retrieval_recall([], [gold]) == 0.0
    assert retrieval_recall([gold], []) == 0.0


# --- 失敗分群 -------------------------------------------------------------------
def test_classify():
    from eval import _classify
    assert _classify(1.0, 0.0, True) == "abstain"
    assert _classify(0.5, 0.0, False) == "retrieval"
    assert _classify(1.0, 0.0, False) == "generation"
    assert _classify(0.0, 1.0, False) == "ok"


# --- embedding 快取 / 分批 / 子集 -------------------------------------------------
class _FakeUsage:
    total_tokens = 100


class _FakeData:
    def __init__(self):
        self.embedding = [1.0, 2.0, 3.0, 4.0]


class _FakeResp:
    def __init__(self, n):
        self.data = [_FakeData() for _ in range(n)]
        self.usage = _FakeUsage()


def test_embed_batch_cache_dedupe(monkeypatch):
    calls = []
    monkeypatch.setattr(rag, "api_call",
                        lambda fn, *a, **kw: (calls.append(len(kw["input"])),
                                              _FakeResp(len(kw["input"])))[1])
    monkeypatch.setattr(rag, "get_client", lambda: type("C", (), {
        "embeddings": type("E", (), {"create": staticmethod(lambda **kw: None)})})())
    rag._embed_cache.clear()

    texts = [f"text-{i}" for i in range(70)]
    vecs, tok = rag._embed(texts, "passage")
    assert vecs.shape == (70, 4)
    assert calls == [32, 32, 6]              # 分批
    _, tok2 = rag._embed(texts, "passage")
    assert calls == [32, 32, 6]              # 快取命中不再打 API
    assert tok2 == tok and tok > 0           # 攤提成本不因命中而失真

    rag._embed_cache.clear()
    calls.clear()
    v3, _ = rag._embed(["same", "same", "other"], "query")
    assert calls == [2] and v3.shape == (3, 4)   # 批內去重


def test_chunkindex_subset_embedding(monkeypatch):
    embed_calls = []

    def fake_embed(texts, input_type):
        embed_calls.append((len(texts), input_type))
        n = len(texts)
        return np.arange(n * 3, dtype=float).reshape(n, 3) + 1.0, 10 * n

    monkeypatch.setattr(rag, "_embed", fake_embed)
    idx = rag.ChunkIndex(["a", "b", "c", "d"])
    idx.dense_scores("q", subset=[1, 3])
    assert embed_calls == [(2, "passage"), (1, "query")]   # 只 embed 候選子集
    idx.dense_scores("q")
    assert embed_calls[-1] == (2, "passage")               # 全量只補缺的


# --- 等價配置正規化 (快取鍵) --------------------------------------------------
def test_rerank_dense_noop_key(monkeypatch):
    import cache as cachemod
    monkeypatch.setattr(cachemod, "RERANK_MODEL", "dense")
    a = dict(FULL_CFG, retriever="dense")
    b = dict(a, rerank=False)
    assert cachemod.config_key(a) == cachemod.config_key(b)   # 等價 -> 同一筆快取
    c = dict(a, retriever="bm25")
    assert cachemod.config_key(c) != cachemod.config_key(b)
    # 預設 (真 reranker) 不做正規化
    monkeypatch.setattr(cachemod, "RERANK_MODEL", "some-reranker")
    assert cachemod.config_key(a) != cachemod.config_key(b)


def test_alpha_canonical_key():
    import cache as cachemod
    # α 只影響 hybrid/graph；bm25/dense 下不同 α 是等價配置 -> 共用快取鍵
    a = dict(FULL_CFG, retriever="bm25", hybrid_alpha=0.3, rerank=False)
    b = dict(a, hybrid_alpha=0.7)
    assert cachemod.config_key(a) == cachemod.config_key(b)
    # hybrid 下 α 真的有差 -> 不正規化
    h3 = dict(FULL_CFG, hybrid_alpha=0.3)
    h7 = dict(FULL_CFG, hybrid_alpha=0.7)
    assert cachemod.config_key(h3) != cachemod.config_key(h7)


# --- racing 提早停止 --------------------------------------------------------------
def _fake_examples(n):
    from eval import Example
    return [Example(question=f"q{i}", answer="gold", paragraphs=["ctx " * 20],
                    gold_paragraphs=["ctx " * 20], level="custom")
            for i in range(n)]


def test_racing_prunes(monkeypatch):
    import eval as ev
    import judge
    calls = {"n": 0}

    def fake_run_rag(q, paras, cfg):
        calls["n"] += 1
        return rag.RagResult(answer="wrong", retrieved_idx=[0],
                             retrieved_chunks=["ctx"], faithful=1.0,
                             tokens=10, latency=0.0)

    monkeypatch.setattr(ev, "run_rag", fake_run_rag)
    monkeypatch.setattr(judge, "judge_correctness", lambda *a: (0.0, 0))
    monkeypatch.setattr(judge, "judge_faithfulness", lambda *a: (1.0, 0))

    examples = _fake_examples(12)
    with pytest.raises(ev.EvalPruned) as exc:
        ev.evaluate(RagConfig(), examples, verbose=False,
                    prune_below=0.9, prune_lam=0.5)
    # 全錯: 第 5 題後上界 (0+7)/12 ≈ 0.583 < 0.9 -> 提早停止
    assert calls["n"] == 5
    assert exc.value.partial_score["pruned"] is True
    assert exc.value.bound < 0.9

    # 不給 prune_below -> 跑好跑滿
    calls["n"] = 0
    score = ev.evaluate(RagConfig(), examples, verbose=False)
    assert calls["n"] == 12 and score.n == 12


# --- objective (λ/μ) + 邊際統計 + 提案清洗 -----------------------------------------
def test_objective_and_proposals(tmp_path):
    from optimizer import (DEFAULT_LAMBDA, _marginal_lines, _valid_untried,
                           all_configs, is_valid, objective)
    s = {"correctness": 0.8, "faithfulness": 0.9, "avg_tokens": 2000}
    assert objective(s, 0.5, 0.0) == pytest.approx(0.75)
    assert objective(s, 0.5, 0.1) == pytest.approx(0.55)   # μ×2ktok = 0.2
    assert DEFAULT_LAMBDA == 0.5
    # 3 chunk × 3 top_k × 4 retriever × 2 overlap × 3 α × 2^5 開關 = 6912
    assert len(all_configs()) == 6912

    assert is_valid(FULL_CFG)
    assert not is_valid(dict(FULL_CFG, chunk_size=999))
    assert not is_valid({k: v for k, v in FULL_CFG.items() if k != "hyde"})  # 缺欄位
    assert len(_valid_untried([FULL_CFG, dict(FULL_CFG),
                               dict(FULL_CFG, chunk_size=999)], set())) == 1

    from memory.trajectory import Trajectory, Trial
    traj = Trajectory(tmp_path / "t.jsonl")
    traj.add(Trial(config=dict(FULL_CFG),
                   score={"correctness": 0.8, "faithfulness": 1.0},
                   objective=0.8, lam=0.5))
    lines = _marginal_lines(traj)
    assert "512" in lines and "未試" in lines


# --- pareto / 符號檢定 --------------------------------------------------------------
def test_pareto_and_sign_test():
    from pareto import pareto_front
    from sweep import _sign_test_p
    assert pareto_front([(0.1, 0.5), (0.2, 0.6), (0.05, 0.4), (0.3, 0.6)]) == \
        [(0.05, 0.4), (0.1, 0.5), (0.2, 0.6)]
    assert _sign_test_p(0, 0) == 1.0
    assert _sign_test_p(5, 5) == 1.0
    assert _sign_test_p(9, 1) == pytest.approx(22 / 1024)
    assert _sign_test_p(1, 9) == _sign_test_p(9, 1)


# --- cache: 指紋 / record / 原子寫檔 / racing 穿透 -----------------------------------
def test_cache_record_and_pruned(tmp_path, monkeypatch):
    import cache as cachemod
    import eval as ev
    from eval import EvalScore, QuestionDetail

    d = QuestionDetail(question="q1", gold="g", pred=ABSTAIN, em=0, f1=0,
                       recall=1.0, faithful=1.0, correct=0.0,
                       fail_type="abstain", retrieved_preview="")
    es = EvalScore(em=0, f1=0, recall=1, faithfulness=1, correctness=0,
                   abstain_rate=1, avg_latency=0, avg_tokens=0, n=1,
                   n_errors=0, details=[d])
    monkeypatch.setattr(ev, "evaluate", lambda *a, **k: es)

    p = tmp_path / "c.json"
    rc = cachemod.ResultCache(path=p, dataset="test")
    rec = rc.evaluate_cached(RagConfig(), [None])
    assert rec["per_question_correct"] == {"q1": 0.0}
    assert rec["fail_clusters"]["abstain"] == 1
    assert p.exists()
    assert cachemod.ResultCache(path=p, dataset="test").get(RagConfig()) is not None

    # racing 穿透: EvalPruned -> 輕量 record 且不落 cache
    def raise_pruned(*a, **k):
        raise ev.EvalPruned(5, 0.4, {"correctness": 0.1, "faithfulness": 1.0,
                                     "recall": 0, "em": 0, "f1": 0,
                                     "abstain_rate": 0, "avg_latency": 0,
                                     "avg_tokens": 0, "n": 5, "n_errors": 0,
                                     "pruned": True})
    monkeypatch.setattr(ev, "evaluate", raise_pruned)
    cfg2 = RagConfig(chunk_size=1024)
    rec2 = rc.evaluate_cached(cfg2, [None], prune_below=0.9)
    assert rec2["pruned"] and rc.get(cfg2) is None

    assert "ev=" in cachemod.dataset_key(30, 15, 42)
    assert cachemod.custom_dataset_key("abc123", 10).startswith("custom|fp=abc123")


# --- warm-start ---------------------------------------------------------------------
def test_warmstart_roundtrip(tmp_path, monkeypatch):
    from memory import warmstart as ws
    monkeypatch.setattr(ws, "PATH", tmp_path / "warmstart.jsonl")
    feats_zh = {"n_questions": 20, "n_paragraphs": 50,
                "avg_para_chars": 120.0, "cjk_ratio": 0.9}
    feats_en = {"n_questions": 30, "n_paragraphs": 300,
                "avg_para_chars": 600.0, "cjk_ratio": 0.0}
    cfg_zh = dict(FULL_CFG, chunk_size=256, verify=True)
    cfg_en = dict(FULL_CFG, chunk_size=1024, top_k=8, retriever="dense",
                  rerank=False, query_decompose=True)
    ws.record("ds-zh", feats_zh, cfg_zh, 0.7)
    ws.record("ds-en", feats_en, cfg_en, 0.6)
    ws.record("ds-zh", feats_zh, dict(cfg_zh, top_k=3), 0.5)   # 較差 -> 不覆蓋
    got = ws.suggest({"n_questions": 20, "n_paragraphs": 60,
                      "avg_para_chars": 100.0, "cjk_ratio": 0.85}, k=1)
    assert got == [cfg_zh]                    # 最相似 (中文/短段) 的贏家先出
    assert ws.suggest(feats_zh, k=5, exclude_dataset="ds-zh") == [cfg_en]

    from optimizer import _warm_init, all_configs
    grid = all_configs()
    init, n_warm = _warm_init(grid, [cfg_zh, cfg_zh, {"bad": 1}], 3)
    assert n_warm == 1 and len(init) == 3     # 去重+非法過濾，隨機補滿
    init2, n0 = _warm_init(grid, None, 3)
    assert n0 == 0 and len(init2) == 3


# --- graph-lite 檢索: 橋接 chunk 被實體共現邊拉上來 ---------------------------------
def test_graph_pulls_bridge_chunk():
    # A 與 query 高度相關 (種子)；B 與 A 共享稀有實體 zephyrium (橋接) 但與
    # query 無字面重疊；C 無關。α=0 -> 純 bm25，不打任何 API。
    chunks = [
        "The zephyrium reactor was designed by Doctor Marlow in Geneva.",
        "Zephyrium alloys are produced only at the Kestrel facility in Norway.",
        "Bananas are rich in potassium and grow in tropical climates.",
    ]
    query = "Who designed the reactor built by Doctor Marlow?"
    idx = rag.ChunkIndex(chunks)
    base = idx.scores(query, "hybrid", alpha=0.0)
    graph = idx.scores(query, "graph", alpha=0.0, seed_k=1)
    assert graph[1] > base[1]                       # 橋接 chunk 被加分
    assert list(np.argsort(graph)[::-1][:2]) == [0, 1]   # 種子+橋接進前二
    assert np.argsort(graph)[::-1][2] == 2          # 無關 chunk 還是最後


def test_hybrid_alpha_weighting(monkeypatch):
    # fake embedding: 讓 dense 偏愛 chunk1 (與 query 向量同向)，bm25 偏愛 chunk0
    def fake_embed(texts, input_type):
        vecs = []
        for t in texts:
            if input_type == "query" or "quantum" in t:
                vecs.append([0.0, 1.0])     # query 與 chunk1 同向
            else:
                vecs.append([1.0, 0.0])
        return np.array(vecs, dtype=float), 10

    calls = []
    monkeypatch.setattr(rag, "_embed",
                        lambda t, it: (calls.append(1), fake_embed(t, it))[1])
    chunks = ["apple pie recipe with apples", "quantum entanglement basics"]
    q = "apple pie"

    idx = rag.ChunkIndex(chunks)
    assert int(np.argmax(idx.scores(q, "hybrid", alpha=0.0))) == 0   # 純字面
    assert not calls                     # α=0 短路，完全不碰 embedding
    assert int(np.argmax(idx.scores(q, "hybrid", alpha=1.0))) == 1   # 純語意
    assert calls                         # α=1 才需要 embedding


# --- iterative + HyDE 管線 (全 mock，不打 API) --------------------------------------
def test_run_rag_iterative_and_hyde(monkeypatch):
    monkeypatch.setattr(rag, "generate", lambda q, ch: ("Paris", 5))
    monkeypatch.setattr(rag, "hyde_query", lambda q: ("The capital city is Paris.", 3))
    gap_calls = []

    def fake_gap(q, ch):
        gap_calls.append(len(ch))
        return "second hop query about towers", 4

    monkeypatch.setattr(rag, "gap_query", fake_gap)
    paras = [f"passage number {i} about towers and capitals " * 3 for i in range(6)]
    cfg = RagConfig(chunk_size=512, top_k=2, retriever="bm25",
                    hyde=True, iterative=True)
    res = rag.run_rag("What is the capital?", paras, cfg)
    assert res.answer == "Paris"
    assert gap_calls and gap_calls[0] == 2          # 第一輪 top_k=2 進 gap 檢查
    assert len(res.retrieved_chunks) <= cfg.top_k + 3   # 合併上限
    assert res.tokens >= 5 + 3 + 4                  # generate + hyde + gap

    # gap 回 NONE -> 不做第二輪，檢索結果維持 top_k
    monkeypatch.setattr(rag, "gap_query", lambda q, ch: (None, 2))
    res2 = rag.run_rag("What is the capital?", paras, cfg)
    assert len(res2.retrieved_chunks) == 2


# --- 雙判官抽查一致率 ----------------------------------------------------------------
def test_judge_audit_agreement(monkeypatch):
    import eval as ev
    import judge

    monkeypatch.setattr(judge, "AUDIT_RATE", 1.0)   # 全抽
    monkeypatch.setattr(ev, "run_rag", lambda q, p, c: rag.RagResult(
        answer="Paris", retrieved_idx=[0], retrieved_chunks=["ctx"],
        faithful=1.0, tokens=10, latency=0.0))
    monkeypatch.setattr(judge, "judge_correctness", lambda *a: (1.0, 0))
    monkeypatch.setattr(judge, "audit_correctness", lambda *a: (0.0, 0))  # 全不同意

    score = ev.evaluate(RagConfig(), _fake_examples(4), verbose=False)
    assert score.judge_agreement == 0.0
    assert score.as_dict()["judge_agreement"] == 0.0

    # 抽查關閉 -> None，不多花任何呼叫
    monkeypatch.setattr(judge, "AUDIT_RATE", 0.0)
    score2 = ev.evaluate(RagConfig(), _fake_examples(4), verbose=False)
    assert score2.judge_agreement is None
