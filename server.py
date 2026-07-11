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

import hashlib
import threading
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from eval import load_hotpot
from cache import DEFAULT_DATASET, ResultCache, dataset_key, custom_dataset_key
from dataset import (split_paragraphs, generate_qa, validate_qa, save_qa,
                     load_qa, build_examples, dataset_fingerprint,
                     MAX_PARAGRAPHS)
from optimizer import DEFAULT_LAMBDA, random_search, opro_search
from hybrid import hybrid_search
from memory.trajectory import Trajectory
from pareto import pareto_front

load_dotenv()

RESULTS = Path(__file__).parent / "results"
DATA = Path(__file__).parent / "data"      # 自訂語料/生成 QA 的落地目錄
app = FastAPI(title="Agentic RAG + OPRO")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# --- 全域任務狀態 (Demo: 一次跑一個) --------------------------------------
STATE: dict = {"running": False, "strategy": None, "done": 0, "total": 0,
               "log": [], "error": None, "dataset": DEFAULT_DATASET,
               "lam": DEFAULT_LAMBDA}
_lock = threading.Lock()


@lru_cache(maxsize=4)
def _load_examples(n: int, n_hard: int) -> list:
    """同參數的測試集只載一次 (load_hotpot 要下載/掃 HotpotQA，很慢)。"""
    return load_hotpot(n=n, n_hard=n_hard)


class RunReq(BaseModel):
    strategy: str = "opro"      # random | opro | hybrid
    n: int = 20                 # 測試題數 (custom 模式 = 生成/取用的 QA 題數)
    n_hard: int = 10            # hotpot 專用; custom 模式忽略
    budget: int = 8             # 評估次數
    lam: float = 0.5            # 幻覺懲罰權重
    dataset_mode: str = "hotpot"        # hotpot | custom
    corpus_text: str = ""               # custom: 文件全文 (前端已把多檔串好)
    qa: list[dict] | None = None        # custom: 自備 QA; None -> 用 LLM 生成 n 題


def _log(msg: str) -> None:
    with _lock:
        STATE["log"].append(msg)
        # done 以軌跡檔行數估算在 status 端做; 這裡僅收集 log


def _prepare_custom(req: RunReq) -> tuple[list, str]:
    """自訂資料集: 語料切段 -> (自備 / LLM 生成) QA -> Example。
    生成的 QA 落地 data/<語料hash>/，同語料同題數之後直接重用不再花 API。"""
    paragraphs = split_paragraphs(req.corpus_text)
    if not paragraphs:
        raise ValueError(
            "語料是空的: 請上傳/貼上 .txt 或 .md 內容 (空行分段，段落 ≥30 字元)")
    if len(paragraphs) > MAX_PARAGRAPHS:
        raise ValueError(
            f"語料 {len(paragraphs)} 段超過上限 {MAX_PARAGRAPHS} 段，請先抽一個子集")

    if req.qa:
        qa = validate_qa(req.qa)
        _log(f"使用自備 QA {len(qa)} 題")
    else:
        corpus_id = hashlib.sha1(req.corpus_text.encode("utf-8")).hexdigest()[:10]
        ddir = DATA / corpus_id
        qa_path = ddir / f"qa_gen_n{req.n}_seed42.jsonl"
        if qa_path.exists():
            qa = load_qa(qa_path)
            _log(f"重用先前生成的 QA ({len(qa)} 題)")
        else:
            _log(f"用 LLM 生成 {req.n} 題 QA 中… (每題約 1 次 API 呼叫)")
            qa = generate_qa(paragraphs, n=req.n, seed=42, log_fn=_log)
            ddir.mkdir(parents=True, exist_ok=True)
            (ddir / "corpus.txt").write_text(req.corpus_text, encoding="utf-8")
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
        # λ 以參數傳入 (不動全域狀態)，且會記進每筆 Trial
        common = dict(examples=examples, cache=cache, budget=req.budget,
                      seed=42, traj_path=traj_path, verbose=False, log_fn=_log,
                      lam=req.lam)
        if req.strategy == "random":
            random_search(**common)
        elif req.strategy == "hybrid":
            hybrid_search(n_init=3, **common)
        else:
            opro_search(n_init=3, **common)
        _log("✅ 完成")
    except Exception as e:  # noqa: BLE001 — 回報給前端而非崩潰
        with _lock:
            STATE["error"] = f"{type(e).__name__}: {e}"
        _log(f"❌ 失敗: {e}")
    finally:
        with _lock:
            STATE["running"] = False


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
                     lam=req.lam)
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
    for name in ("random", "opro", "hybrid"):
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
    cache = ResultCache(dataset=dataset)
    pts = [{"key": r["config"], "hallucination": 1 - r["score"]["faithfulness"],
            "correctness": r["score"]["correctness"]} for r in cache.all_records()]
    front_set = set(pareto_front([(p["hallucination"], p["correctness"]) for p in pts]))
    front = sorted((p for p in pts
                    if (p["hallucination"], p["correctness"]) in front_set),
                   key=lambda p: (p["hallucination"], -p["correctness"]))
    return {"strategies": strategies, "best": overall_best,
            "pareto": {"points": pts, "front": front},
            "lam": lam}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
