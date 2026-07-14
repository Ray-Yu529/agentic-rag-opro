"""
hybrid.py — 混合最佳化: LLM 縮範圍 -> Optuna 在範圍內收斂

動機 (前面討論的結論):
  LLM 擅長「看失敗案例、定大方向、砍掉無效區域」，但後期精細收斂不如數值方法;
  Optuna 擅長收斂卻「看不懂為什麼失敗」。兩者互補:
    1) 先隨機暖身幾組
    2) 讓 LLM 讀結果，把每個維度限縮到值得搜的子集 (region)
    3) Optuna 只在這個縮小的空間裡採樣收斂
    4) 預算走到一半時，拿新結果讓 LLM「重新」縮一次範圍再收斂
       (只縮一次的話，早期縮錯就全程被關在壞區域)

與 optimizer.py 共用 cache (查過不重跑) 與 Trajectory (best-so-far 曲線可比)。
"""

from __future__ import annotations

import json
import random

from rag import RagConfig
from cache import ResultCache, config_to_dict
from memory.trajectory import Trajectory
from optimizer import (DEFAULT_LAMBDA, DEFAULT_MU, SEARCH_SPACE, all_configs,
                       chat_json, objective, _trial, _make_say, _incumbent,
                       _race_note, _warm_init)


def _ask_region(traj: Trajectory) -> dict:
    """請 LLM 把每個維度限縮成子集。回傳 {dim: [values...]}，缺/錯的維度退回全集。
    維度清單/JSON 範例從 SEARCH_SPACE 動態生成，之後加維度不會漏。"""
    ranked = sorted(traj.trials, key=lambda t: t.objective, reverse=True)
    lines = [
        f"  obj={t.objective:.3f}: "
        + ", ".join(f"{k}={t.config.get(k)}" for k in SEARCH_SPACE)
        for t in ranked
    ]
    dims = "\n".join(f"  {k} ∈ {v}" for k, v in SEARCH_SPACE.items())
    example = json.dumps({k: [] for k in SEARCH_SPACE}, ensure_ascii=False)
    prompt = f"""你是 RAG 最佳化架構師。以下是目前的實驗結果 (objective 越高越好):
{chr(10).join(lines)}

合法值:
{dims}

請判斷哪些值得繼續搜，把每個維度限縮成「值得搜的子集」(可只留 1~2 個值)，砍掉看起來沒用的。
只輸出 JSON: {{"reasoning":"...", "region":{example}}} (region 每個維度填入保留的值)"""

    region = chat_json(prompt, temperature=0.3, max_tokens=600,
                       task="hybrid_region").get("region", {})
    # 清洗: 子集必須 ⊆ 合法值，否則該維度退回全集
    clean = {}
    for dim, allowed in SEARCH_SPACE.items():
        sub = [v for v in region.get(dim, []) if v in allowed]
        clean[dim] = sub or list(allowed)
    return clean


def hybrid_search(examples, cache: ResultCache, budget: int, seed: int, traj_path: str,
                  n_init: int = 3, verbose: bool = True, log_fn=None,
                  lam: float = DEFAULT_LAMBDA, mu: float = DEFAULT_MU,
                  warm_configs: list[dict] | None = None) -> Trajectory:
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

    # 1) 暖身 (與其他策略同 seed；warm-start 開啟時取代部分暖身組)
    init_list, n_warm = _warm_init(grid, warm_configs, n_init)
    for i, cfg in enumerate(init_list):
        rec = cache.evaluate_cached(cfg, examples, verbose=verbose,
                                    prune_below=_incumbent(traj), prune_lam=lam)
        traj.add(_trial(rec, "init", lam, mu))
        say(f"[{'warm-start' if i < n_warm else '暖身'}] {config_to_dict(cfg)}"
            f"{_race_note(rec)}")

    def run_phase(region: dict, target: int, phase_seed: int) -> None:
        """Optuna 在 region 內收斂，直到軌跡達 target 次評估。"""
        def opt_objective(trial):
            d = {dim: trial.suggest_categorical(dim, region[dim]) for dim in SEARCH_SPACE}
            cfg = RagConfig(**d)
            is_new = cfg.key() not in traj.tried_keys()
            # 已達本階段目標且是沒評估過的新配置 -> 直接剪枝，不再花 API
            if is_new and len(traj.trials) >= target:
                raise optuna.TrialPruned()
            rec = cache.evaluate_cached(cfg, examples, verbose=verbose,
                                        prune_below=_incumbent(traj), prune_lam=lam)
            # 只把「新配置」記進軌跡，讓 best-so-far 曲線以「不同評估次數」計
            if is_new:
                traj.add(_trial(rec, "hybrid", lam, mu))
            # racing 提早停止的配置: 確定劣於當時最佳，對 Optuna 也標為剪枝
            if rec.get("pruned"):
                raise optuna.TrialPruned()
            return objective(rec["score"], lam, mu)

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=phase_seed))
        # 多給一些 trial 數，因可能抽到重複 (重複查快取零成本，但不增評估次數)
        n_trials = max(1, (target - len(traj.trials)) * 3)
        study.optimize(opt_objective, n_trials=n_trials, show_progress_bar=False)

    if len(traj.trials) >= budget:
        return traj

    # 2) LLM 縮範圍 -> Optuna 收斂 (前半預算)
    remaining = budget - len(traj.trials)
    mid = len(traj.trials) + (remaining + 1) // 2
    region = _ask_region(traj)
    say(f"\n[hybrid] LLM 縮範圍後的搜索區域: {region}")
    run_phase(region, mid, seed)

    # 3) 中場用累積結果「重新」縮範圍，後半預算再收斂
    if len(traj.trials) < budget:
        region = _ask_region(traj)
        say(f"\n[hybrid] 中場重新縮範圍: {region}")
        run_phase(region, budget, seed + 1)

    return traj
