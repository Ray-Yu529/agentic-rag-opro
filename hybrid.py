"""
hybrid.py — 混合最佳化: LLM 縮範圍 -> Optuna 在範圍內收斂

動機 (前面討論的結論):
  LLM 擅長「看失敗案例、定大方向、砍掉無效區域」，但後期精細收斂不如數值方法;
  Optuna 擅長收斂卻「看不懂為什麼失敗」。兩者互補:
    1) 先隨機暖身幾組
    2) 讓 LLM 讀結果，把每個維度限縮到值得搜的子集 (region)
    3) Optuna 只在這個縮小的空間裡採樣收斂

與 optimizer.py 共用 cache (查過不重跑) 與 Trajectory (best-so-far 曲線可比)。
"""

from __future__ import annotations

import random

from rag import RagConfig
from cache import ResultCache, config_to_dict
from memory.trajectory import Trajectory
from optimizer import (SEARCH_SPACE, all_configs, chat_json, objective, _trial,
                       _make_say)


def _ask_region(traj: Trajectory) -> dict:
    """請 LLM 把每個維度限縮成子集。回傳 {dim: [values...]}，缺/錯的維度退回全集。"""
    ranked = sorted(traj.trials, key=lambda t: t.objective, reverse=True)
    lines = [
        f"  obj={t.objective:.3f}: chunk={t.config['chunk_size']}, top_k={t.config['top_k']}, "
        f"ret={t.config['retriever']}, rerank={int(t.config['rerank'])}, "
        f"decomp={int(t.config['query_decompose'])}, verify={int(t.config['verify'])}"
        for t in ranked
    ]
    prompt = f"""你是 RAG 最佳化架構師。以下是初步隨機實驗結果 (objective 越高越好):
{chr(10).join(lines)}

合法值:
  chunk_size ∈ {SEARCH_SPACE['chunk_size']}, top_k ∈ {SEARCH_SPACE['top_k']},
  retriever ∈ {SEARCH_SPACE['retriever']}, rerank/query_decompose/verify ∈ [false,true]

請判斷哪些值得繼續搜，把每個維度限縮成「值得搜的子集」(可只留 1~2 個值)，砍掉看起來沒用的。
只輸出 JSON: {{"reasoning":"...", "region":{{"chunk_size":[...],"top_k":[...],"retriever":[...],"rerank":[...],"query_decompose":[...],"verify":[...]}}}}"""

    region = chat_json(prompt, temperature=0.3, max_tokens=600).get("region", {})
    # 清洗: 子集必須 ⊆ 合法值，否則該維度退回全集
    clean = {}
    for dim, allowed in SEARCH_SPACE.items():
        sub = [v for v in region.get(dim, []) if v in allowed]
        clean[dim] = sub or list(allowed)
    return clean


def hybrid_search(examples, cache: ResultCache, budget: int, seed: int, traj_path: str,
                  n_init: int = 3, verbose: bool = True, log_fn=None) -> Trajectory:
    say = _make_say(log_fn, verbose)
    try:
        import optuna
    except ImportError as e:
        raise SystemExit("hybrid 需要 optuna: pip install optuna") from e
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    rng = random.Random(seed)
    grid = all_configs()
    rng.shuffle(grid)
    traj = Trajectory(traj_path)
    traj.reset()

    # 1) 暖身 (與其他策略同 seed)
    for cfg in grid[:n_init]:
        traj.add(_trial(cache.evaluate_cached(cfg, examples, verbose=verbose), "init"))
        say(f"[暖身] {config_to_dict(cfg)}")

    # 2) LLM 縮範圍
    region = _ask_region(traj)
    say(f"\n[hybrid] LLM 縮範圍後的搜索區域: {region}")

    # 3) Optuna 在縮小空間內收斂
    def opt_objective(trial):
        d = {dim: trial.suggest_categorical(dim, region[dim]) for dim in SEARCH_SPACE}
        cfg = RagConfig(**d)
        is_new = cfg.key() not in traj.tried_keys()
        # 已達預算且是沒評估過的新配置 -> 直接剪枝，不再花 API
        if is_new and len(traj.trials) >= budget:
            raise optuna.TrialPruned()
        rec = cache.evaluate_cached(cfg, examples, verbose=verbose)
        # 只把「新配置」記進軌跡，讓 best-so-far 曲線以「不同評估次數」計
        if is_new:
            traj.add(_trial(rec, "hybrid"))
        return objective(rec["score"])

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    # 多給一些 trial 數，因可能抽到重複 (重複查快取零成本，但不增評估次數)
    study.optimize(opt_objective, n_trials=(budget - n_init) * 3,
                   show_progress_bar=False)
    return traj
