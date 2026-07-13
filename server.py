"""
server.py — FastAPI 後端: 把 optimizer 包成 API 給 React 前端用

目的: 使用者不用碰 CLI / 改 Python / 讀 JSON，按按鈕就能跑、看圖表。

端點:
  POST /api/run      啟動一次最佳化 (背景執行緒)，body: {strategy, n, n_hard, budget, lam}
  GET  /api/status   即時進度 + OPRO 推理 log
  GET  /api/results  三種策略的曲線 / Pareto / leaderboard / 最佳配置

開發時前端 (Vite, :5173) 透過 proxy 打到這裡 (:8000)。
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import os
import threading
import time
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from eval import load_hotpot
from rag import is_abstain, run_rag, strip_citations
from cache import (DEFAULT_DATASET, ResultCache, dataset_key,
                   custom_dataset_key, dict_to_config)
from dataset import (split_paragraphs, generate_qa, validate_qa, save_qa,
                     load_qa, build_examples, dataset_fingerprint,
                     extract_pdf_text, MAX_PARAGRAPHS)
from optimizer import (DEFAULT_LAMBDA, DEFAULT_MU, SEARCH_SPACE,
                       bandit_search, random_search, opro_search)
from hybrid import hybrid_search
from memory import warmstart
from memory.trajectory import Trajectory
from pareto import pareto_front

load_dotenv()

RESULTS = Path(__file__).parent / "results"
DATA = Path(__file__).parent / "data"      # 自訂語料/生成 QA 的落地目錄
HISTORY = RESULTS / "history.jsonl"        # 每次 run 的摘要 (History 面板/報告用)
app = FastAPI(title="Agentic RAG + OPRO")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# --- 全域任務狀態 (Demo: 一次跑一個) --------------------------------------
STATE: dict = {"running": False, "strategy": None, "done": 0, "total": 0,
               "log": [], "error": None, "dataset": DEFAULT_DATASET,
               "lam": DEFAULT_LAMBDA, "mu": DEFAULT_MU, "run_id": None}
_lock = threading.Lock()


@lru_cache(maxsize=4)
def _load_examples(n: int, n_hard: int) -> list:
    """同參數的測試集只載一次 (load_hotpot 要下載/掃 HotpotQA，很慢)。"""
    return load_hotpot(n=n, n_hard=n_hard)


class RunReq(BaseModel):
    strategy: str = "opro"      # random | bandit | opro | hybrid
    n: int = 20                 # 測試題數 (custom 模式 = 生成/取用的 QA 題數)
    n_hard: int = 10            # hotpot 專用; custom 模式忽略
    budget: int = 8             # 評估次數
    lam: float = 0.5            # 幻覺懲罰權重
    mu: float = 0.0             # 成本懲罰權重 (objective -= μ×每題ktok)
    dataset_mode: str = "hotpot"        # hotpot | custom
    corpus_text: str = ""               # custom: 文件全文 (前端已把多檔串好)
    corpus_pdfs: list[dict] | None = None  # custom: PDF 檔 [{name, b64}]
    qa: list[dict] | None = None        # custom: 自備 QA; None -> 用 LLM 生成 n 題
    multihop: bool = False              # custom+生成: 約一半生成跨段落多跳題
    warmstart: bool = False             # OPRO/hybrid 用過去相似資料集的配置暖身


def _log(msg: str) -> None:
    with _lock:
        STATE["log"].append(msg)
        # done 以軌跡檔行數估算在 status 端做; 這裡僅收集 log


def _prepare_custom(req: RunReq) -> tuple[list, str]:
    """自訂資料集: (PDF 抽取 +) 語料切段 -> (自備 / LLM 生成) QA -> Example。
    生成的 QA 落地 data/<語料hash>/，同語料同題數之後直接重用不再花 API。"""
    corpus_text = req.corpus_text
    for f in (req.corpus_pdfs or []):
        _log(f"解析 PDF: {f.get('name', '?')} (文字層優先，掃描頁走 VLM)")
        corpus_text += "\n\n" + extract_pdf_text(
            base64.b64decode(f["b64"]), log_fn=_log)

    paragraphs = split_paragraphs(corpus_text)
    if not paragraphs:
        raise ValueError(
            "語料是空的: 請上傳/貼上 .txt/.md/.pdf 內容 (空行分段，段落 ≥30 字元)")
    if len(paragraphs) > MAX_PARAGRAPHS:
        raise ValueError(
            f"語料 {len(paragraphs)} 段超過上限 {MAX_PARAGRAPHS} 段，請先抽一個子集")

    if req.qa:
        qa = validate_qa(req.qa)
        _log(f"使用自備 QA {len(qa)} 題")
    else:
        corpus_id = hashlib.sha1(corpus_text.encode("utf-8")).hexdigest()[:10]
        ddir = DATA / corpus_id
        mh = "_mh" if req.multihop else ""
        qa_path = ddir / f"qa_gen_n{req.n}_seed42{mh}.jsonl"
        if qa_path.exists():
            qa = load_qa(qa_path)
            _log(f"重用先前生成的 QA ({len(qa)} 題)")
        else:
            _log(f"用 LLM 生成 {req.n} 題 QA 中… (每題約 1 次 API 呼叫)")
            qa = generate_qa(paragraphs, n=req.n, seed=42, log_fn=_log,
                             multihop=req.multihop)
            ddir.mkdir(parents=True, exist_ok=True)
            (ddir / "corpus.txt").write_text(corpus_text, encoding="utf-8")
            save_qa(qa, qa_path)
            _log("QA 已存檔 (data/)，同語料同題數之後直接重用")

    examples = build_examples(paragraphs, qa, n=req.n, seed=42, log_fn=_log)
    dkey = custom_dataset_key(dataset_fingerprint(paragraphs, qa),
                              len(examples), 42)
    _log(f"自訂資料集就緒: {len(paragraphs)} 段語料 / {len(examples)} 題")
    return examples, dkey


def _job(req: RunReq) -> None:
    try:
        if req.dataset_mode == "custom":
            examples, dkey = _prepare_custom(req)
        else:
            n_hard = min(req.n_hard, req.n)
            examples = _load_examples(req.n, n_hard)
            # 快取鍵含測試集 fingerprint: 改題數後不會沿用到舊測試集的分數
            dkey = dataset_key(req.n, n_hard)
        with _lock:
            STATE["dataset"] = dkey   # /api/results 的 Pareto 以此取同一批 record
        cache = ResultCache(dataset=dkey)
        traj_path = str(RESULTS / f"{req.strategy}.jsonl")

        feats = warmstart.dataset_features(examples)
        warm = None
        if req.warmstart:
            warm = warmstart.suggest(feats, k=2, exclude_dataset=dkey) or None
            _log(f"warm-start 建議: {warm or '無 (記憶庫還是空的)'}")

        # λ/μ 以參數傳入 (不動全域狀態)，且會記進每筆 Trial
        common = dict(examples=examples, cache=cache, budget=req.budget,
                      seed=42, traj_path=traj_path, verbose=False, log_fn=_log,
                      lam=req.lam, mu=req.mu)
        if req.strategy == "random":
            traj = random_search(**common)
        elif req.strategy == "bandit":
            traj = bandit_search(n_init=3, **common)
        elif req.strategy == "hybrid":
            traj = hybrid_search(n_init=3, warm_configs=warm, **common)
        else:
            traj = opro_search(n_init=3, warm_configs=warm, **common)

        # 跨 run 經驗記憶 + History 摘要 (報告/歷史面板用)
        best = traj.best()
        if best is not None:
            warmstart.record(dkey, feats, best.config, best.objective)
            _append_history(req, dkey, traj)
        _log("✅ 完成")
    except Exception as e:  # noqa: BLE001 — 回報給前端而非崩潰
        with _lock:
            STATE["error"] = f"{type(e).__name__}: {e}"
        _log(f"❌ 失敗: {e}")
    finally:
        with _lock:
            STATE["running"] = False


def _append_history(req: RunReq, dkey: str, traj) -> None:
    """把一次 run 的摘要 (參數/最佳配置/全部試驗/推理 log) 追加到 history.jsonl。"""
    b = traj.best()
    with _lock:
        run_id = STATE["run_id"]
        log_snapshot = list(STATE["log"])
    row = {
        "run_id": run_id,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "strategy": req.strategy,
        "params": {"n": req.n, "n_hard": req.n_hard, "budget": req.budget,
                   "lam": req.lam, "mu": req.mu,
                   "dataset_mode": req.dataset_mode, "multihop": req.multihop,
                   "warmstart": req.warmstart},
        "dataset": dkey,
        "best": {"config": b.config, "score": b.score, "objective": b.objective},
        "trials": [{"config": t.config, "score": t.score,
                    "objective": t.objective, "source": t.source}
                   for t in traj.trials],
        "log": log_snapshot,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    with HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


@app.post("/api/run")
def run(req: RunReq):
    with _lock:
        if STATE["running"]:
            return {"ok": False, "msg": "已有任務在執行"}
        STATE.update(running=True, strategy=req.strategy, done=0,
                     total=req.budget, log=[], error=None,
                     # custom 的 key 要等 QA 就緒才算得出來，先放占位符由 _job 更新
                     dataset=("custom|pending" if req.dataset_mode == "custom"
                              else dataset_key(req.n, min(req.n_hard, req.n))),
                     lam=req.lam, mu=req.mu,
                     run_id=time.strftime("%Y%m%d-%H%M%S"))
    threading.Thread(target=_job, args=(req,), daemon=True).start()
    return {"ok": True}


@app.get("/api/status")
def status():
    with _lock:
        snapshot = dict(STATE)
    # done 以當前策略軌跡檔的試驗數即時回報
    done = 0
    if snapshot["strategy"]:
        p = RESULTS / f"{snapshot['strategy']}.jsonl"
        if p.exists():
            done = sum(1 for _ in p.read_text(encoding="utf-8").splitlines() if _.strip())
    return {**snapshot, "done": done}


@app.get("/api/results")
def results():
    strategies, overall_best = {}, None
    for name in ("random", "bandit", "opro", "hybrid"):
        p = RESULTS / f"{name}.jsonl"
        if not p.exists():
            continue
        traj = Trajectory.load(p)
        if not traj.trials:
            continue
        strategies[name] = {
            "curve": traj.best_curve(),
            "lam": traj.trials[-1].lam,   # 該軌跡實際使用的 λ (跨 run 可能不同)
            "trials": [{"config": t.config, "score": t.score,
                        "objective": t.objective, "source": t.source}
                       for t in traj.trials],
        }
        b = traj.best()
        if overall_best is None or b.objective > overall_best["objective"]:
            overall_best = {"strategy": name, "config": b.config,
                            "score": b.score, "objective": b.objective}

    # Pareto 只取「目前這個測試集」的 record，不同題數的分數不混著畫
    with _lock:
        dataset = STATE["dataset"]
        lam = STATE["lam"]
        mu = STATE["mu"]
    cache = ResultCache(dataset=dataset)
    pts = [{"key": r["config"], "hallucination": 1 - r["score"]["faithfulness"],
            "correctness": r["score"]["correctness"],
            "tokens": round(r["score"].get("avg_tokens", 0))}   # 第三目標: 成本
           for r in cache.all_records()]
    front_set = set(pareto_front([(p["hallucination"], p["correctness"]) for p in pts]))
    front = sorted((p for p in pts
                    if (p["hallucination"], p["correctness"]) in front_set),
                   key=lambda p: (p["hallucination"], -p["correctness"]))
    return {"strategies": strategies, "best": overall_best,
            "pareto": {"points": pts, "front": front},
            "lam": lam, "mu": mu}


class AskReq(BaseModel):
    question: str
    config: dict | None = None      # None -> 用預設配置
    corpus_text: str = ""           # Playground 的檢索語料


@app.post("/api/ask")
def ask(req: AskReq):
    """Playground: 用指定配置對使用者語料即時問答。
    回傳答案 (含 [n] 引用)、檢索到的段落、NLI 忠實度判定。"""
    if not req.question.strip():
        return {"ok": False, "msg": "問題是空的"}
    paragraphs = split_paragraphs(req.corpus_text)
    if not paragraphs:
        return {"ok": False, "msg": "Playground 需要語料: 請在左側「自己的文件」提供內容"}
    cfg = dict_to_config(req.config or {})
    try:
        res = run_rag(req.question, paragraphs, cfg, cite=True)
        faithful = res.faithful
        if faithful is None:   # verify 沒開也要給忠實度判定 (Playground 的賣點)
            from judge import judge_faithfulness
            faithful, _ = judge_faithfulness(strip_citations(res.answer),
                                             res.retrieved_chunks)
        return {"ok": True, "answer": res.answer, "citations": res.citations,
                "chunks": res.retrieved_chunks, "faithful": faithful,
                "abstained": res.abstained or is_abstain(res.answer),
                "tokens": res.tokens, "latency": round(res.latency, 2)}
    except Exception as e:  # noqa: BLE001 — 回報給前端而非 500
        return {"ok": False, "msg": f"{type(e).__name__}: {e}"}


@app.get("/api/records")
def records():
    """A/B 對比用: 目前測試集所有已評估配置的完整 record (含逐題對錯)。"""
    with _lock:
        dataset = STATE["dataset"]
    cache = ResultCache(dataset=dataset)
    return {"records": [
        {"config": r["config"], "score": r["score"],
         "per_question_correct": r.get("per_question_correct", {}),
         "failures": r.get("failures", [])}
        for r in cache.all_records()]}


def _history_rows() -> list[dict]:
    if not HISTORY.exists():
        return []
    return [json.loads(ln) for ln in HISTORY.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


@app.get("/api/history")
def history():
    """過去 run 的摘要 (新的在前)，History 面板用。不含 trials/log (太大)。"""
    rows = _history_rows()
    slim = [{k: r[k] for k in ("run_id", "ts", "strategy", "params",
                               "dataset", "best")} for r in rows]
    return {"runs": slim[::-1]}


def _esc(v) -> str:
    return html.escape(str(v))


def _sensitivity(trials: list[dict]) -> list[tuple[str, object, object, float]]:
    """參數敏感度: 各維度「最佳取值均分 − 最差取值均分」的邊際效應差，遞減排序。
    只看已試樣本、未控制交互作用 —— 是啟發式排序，不是因果結論。"""
    stats: dict[tuple, tuple[int, float]] = {}
    for t in trials:
        for dim in SEARCH_SPACE:
            key = (dim, t["config"].get(dim))
            n, m = stats.get(key, (0, 0.0))
            stats[key] = (n + 1, m + (t["objective"] - m) / (n + 1))
    rows = []
    for dim in SEARCH_SPACE:
        vals = {v: m for (d, v), (_, m) in stats.items() if d == dim}
        if len(vals) >= 2:
            best_v = max(vals, key=vals.get)
            worst_v = min(vals, key=vals.get)
            rows.append((dim, best_v, worst_v, vals[best_v] - vals[worst_v]))
    rows.sort(key=lambda r: -r[3])
    return rows


@app.get("/api/report/{run_id}", response_class=HTMLResponse)
def report(run_id: str):
    """單次 run 的自包含 HTML 報告: 參數、最佳配置、全部試驗、推理軌跡。
    可直接存檔交付 (「為什麼選這組參數」的可稽核紀錄)。"""
    row = next((r for r in _history_rows() if r["run_id"] == run_id), None)
    if row is None:
        return HTMLResponse(f"<h1>找不到 run {_esc(run_id)}</h1>", status_code=404)

    b = row["best"]
    cfg_rows = "".join(
        f"<tr><td>{_esc(k)}</td><td><b>{_esc(v)}</b></td></tr>"
        for k, v in b["config"].items())
    param_rows = "".join(
        f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>"
        for k, v in row["params"].items())
    trial_rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{:.0%}</td><td>{:.0%}</td><td>{:.3f}</td></tr>".format(
            _esc(t["source"]),
            _esc(", ".join(f"{k}={v}" for k, v in t["config"].items())),
            t["score"].get("correctness", 0), 1 - t["score"].get("faithfulness", 1),
            t["objective"])
        for t in sorted(row["trials"], key=lambda t: -t["objective"]))
    sens_rows = "".join(
        f"<tr><td>{_esc(dim)}</td><td>{_esc(bv)}</td><td>{_esc(wv)}</td>"
        f"<td><b>{delta:+.3f}</b></td></tr>"
        for dim, bv, wv, delta in _sensitivity(row["trials"]))
    by_hop = row["best"]["score"].get("by_hop")
    hop_html = ""
    if by_hop:
        hop_html = "<p>難度分層正確率: " + "、".join(
            f"{'單跳' if h == '1' else '多跳'} {v:.0%}"
            for h, v in by_hop.items()) + "</p>"
    log_html = "<br>".join(_esc(ln) for ln in row["log"])

    return HTMLResponse(f"""<!doctype html><html lang="zh-Hant"><head>
<meta charset="utf-8"><title>Run 報告 {_esc(run_id)}</title>
<style>
 body{{font-family:system-ui,'Microsoft JhengHei',sans-serif;max-width:920px;
      margin:2rem auto;padding:0 1rem;color:#222}}
 table{{border-collapse:collapse;margin:.5rem 0 1.5rem}}
 td,th{{border:1px solid #ccc;padding:.3rem .7rem;font-size:.9rem}}
 th{{background:#f2f2f2}} .log{{background:#f8f8f8;border:1px solid #ddd;
 padding:1rem;font-size:.82rem;line-height:1.5}}
</style></head><body>
<h1>Agentic RAG + OPRO — Run 報告</h1>
<p>{_esc(row['ts'])} · 策略 <b>{_esc(row['strategy'])}</b> · run_id {_esc(run_id)}<br>
測試集: {_esc(row['dataset'])}</p>
<h2>參數</h2><table>{param_rows}</table>
<h2>勝出配置 (objective={b['objective']:.3f}，正確率 {b['score'].get('correctness', 0):.0%}，
幻覺率 {1 - b['score'].get('faithfulness', 1):.0%})</h2>
{hop_html}
<table>{cfg_rows}</table>
<h2>參數敏感度 (邊際效應，最佳取值均分 − 最差取值均分)</h2>
<p style="color:#666;font-size:.85rem">只看已試樣本、未控制交互作用，為啟發式排序。</p>
<table><tr><th>維度</th><th>最佳取值</th><th>最差取值</th><th>Δ objective</th></tr>
{sens_rows}</table>
<h2>全部試驗 (依 objective)</h2>
<table><tr><th>來源</th><th>配置</th><th>正確率</th><th>幻覺率</th><th>obj</th></tr>
{trial_rows}</table>
<h2>執行/推理軌跡</h2><div class="log">{log_html}</div>
</body></html>""")


# Docker/單容器部署: 前端 build 後由 FastAPI 直接服務 (web/dist 存在才掛)。
# 開發時仍走 Vite dev server (:5173 proxy 到這裡)，不受影響。
_DIST = Path(__file__).parent / "web" / "dist"
if _DIST.exists():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="web")


if __name__ == "__main__":
    import uvicorn
    # Docker 內設 HOST=0.0.0.0；本機開發維持 127.0.0.1
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "8000")))
