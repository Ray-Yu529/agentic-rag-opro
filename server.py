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

import threading
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import optimizer
from eval import load_hotpot
from cache import ResultCache
from optimizer import random_search, opro_search, objective
from hybrid import hybrid_search
from memory.trajectory import Trajectory

load_dotenv()

RESULTS = Path(__file__).parent / "results"
app = FastAPI(title="Agentic RAG + OPRO")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# --- 全域任務狀態 (Demo: 一次跑一個) --------------------------------------
STATE: dict = {"running": False, "strategy": None, "done": 0, "total": 0,
               "log": [], "error": None}
_lock = threading.Lock()


class RunReq(BaseModel):
    strategy: str = "opro"      # random | opro | hybrid
    n: int = 20                 # 測試題數
    n_hard: int = 10
    budget: int = 8             # 評估次數
    lam: float = 0.5            # 幻覺懲罰權重


def _log(msg: str) -> None:
    with _lock:
        STATE["log"].append(msg)
        # done 以軌跡檔行數估算在 status 端做; 這裡僅收集 log


def _job(req: RunReq) -> None:
    try:
        optimizer.LAMBDA = req.lam  # 即時調整目標權重
        examples = load_hotpot(n=req.n, n_hard=min(req.n_hard, req.n))
        cache = ResultCache()
        traj_path = str(RESULTS / f"{req.strategy}.jsonl")
        common = dict(examples=examples, cache=cache, budget=req.budget,
                      seed=42, traj_path=traj_path, verbose=False, log_fn=_log)
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
                     total=req.budget, log=[], error=None)
    threading.Thread(target=_job, args=(req,), daemon=True).start()
    return {"ok": True}


@app.get("/api/status")
def status():
    # done 以當前策略軌跡檔的試驗數即時回報
    done = 0
    if STATE["strategy"]:
        p = RESULTS / f"{STATE['strategy']}.jsonl"
        if p.exists():
            done = sum(1 for _ in p.read_text(encoding="utf-8").splitlines() if _.strip())
    with _lock:
        return {**STATE, "done": done}


def _pareto(points):
    pts = sorted(points, key=lambda p: (p["hallucination"], -p["correctness"]))
    front, best = [], -1.0
    for p in pts:
        if p["correctness"] > best:
            front.append(p)
            best = p["correctness"]
    return front


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
            "trials": [{"config": t.config, "score": t.score,
                        "objective": t.objective, "source": t.source}
                       for t in traj.trials],
        }
        b = traj.best()
        if overall_best is None or b.objective > overall_best["objective"]:
            overall_best = {"strategy": name, "config": b.config,
                            "score": b.score, "objective": b.objective}

    cache = ResultCache()
    pts = [{"key": r["config"], "hallucination": 1 - r["score"]["faithfulness"],
            "correctness": r["score"]["correctness"]} for r in cache.all_records()]
    return {"strategies": strategies, "best": overall_best,
            "pareto": {"points": pts, "front": _pareto(pts)},
            "lam": optimizer.LAMBDA}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
