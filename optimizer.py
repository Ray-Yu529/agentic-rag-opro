"""
optimizer.py — 搜索空間 + 多目標 + random baseline (D3) + OPRO 迴路 (D4)

搜索空間 (加了 3 個 agentic 開關後 = 216 組):
  chunk_size × top_k × retriever × rerank × query_decompose × verify
全掃不可行 -> random / OPRO 都用 cache.evaluate_cached 即時評估 (查過不重跑)。

多目標 objective: 最大化正確率、同時懲罰幻覺
  objective = correctness - LAMBDA * hallucination_rate
(成本 avg_latency/avg_tokens 不進純量，留給 Pareto 圖看權衡。)

去重 / 合法性檢查放程式裡 (is_valid / tried_keys)，不靠 prompt 約束。
"""

from __future__ import annotations

import json
import re
import random
from itertools import product

from rag import RagConfig, get_client
from cache import ResultCache, config_to_dict, dict_to_config
from memory.trajectory import Trajectory, Trial

META_MODEL = "nvidia/llama-3.1-nemotron-70b-instruct"  # meta-optimizer (推理量小)
LAMBDA = 0.5  # 幻覺懲罰權重

# --- 搜索空間 θ (216 組) ---------------------------------------------------
SEARCH_SPACE = {
    "chunk_size": [256, 512, 1024],
    "top_k": [3, 5, 8],
    "retriever": ["bm25", "dense", "hybrid"],
    "rerank": [False, True],
    "query_decompose": [False, True],
    "verify": [False, True],
}


def all_configs() -> list[RagConfig]:
    keys = list(SEARCH_SPACE)
    return [RagConfig(**dict(zip(keys, combo)))
            for combo in product(*(SEARCH_SPACE[k] for k in keys))]


def is_valid(d: dict) -> bool:
    """LLM 提案的合法性檢查 (每個欄位必須落在 grid 上)。"""
    try:
        return all(d[k] in SEARCH_SPACE[k] for k in SEARCH_SPACE)
    except (KeyError, TypeError):
        return False


def objective(score: dict) -> float:
    """多目標純量化: 正確率 - λ·幻覺率。"""
    hallucination = 1.0 - score.get("faithfulness", 1.0)
    return score.get("correctness", 0.0) - LAMBDA * hallucination


def _trial(record: dict, source: str) -> Trial:
    return Trial(config=record["config"], score=record["score"],
                 objective=objective(record["score"]),
                 failures=record["failures"], source=source)


def _make_say(log_fn, verbose):
    """log_fn 優先 (後端用); 否則 verbose 時印出。"""
    if log_fn is not None:
        return log_fn
    return print if verbose else (lambda *_: None)


# --- D3: Random search baseline -------------------------------------------
def random_search(examples, cache: ResultCache, budget: int, seed: int,
                  traj_path: str, verbose: bool = True, log_fn=None) -> Trajectory:
    say = _make_say(log_fn, verbose)
    rng = random.Random(seed)
    grid = all_configs()
    rng.shuffle(grid)
    traj = Trajectory(traj_path)
    traj.reset()
    for cfg in grid[:budget]:
        rec = cache.evaluate_cached(cfg, examples, verbose=verbose)
        traj.add(_trial(rec, source="random"))
        say(f"[random] 評估 {config_to_dict(cfg)} -> obj={objective(rec['score']):.3f}")
    return traj


# --- D4: OPRO meta-optimizer -----------------------------------------------
def _build_meta_prompt(traj: Trajectory, cache: ResultCache) -> str:
    ranked = sorted(traj.trials, key=lambda t: t.objective, reverse=True)
    lines = []
    for rank, t in enumerate(ranked, 1):
        c, s = t.config, t.score
        lines.append(
            f"  #{rank} chunk={c['chunk_size']}, top_k={c['top_k']}, ret={c['retriever']:6s}, "
            f"rerank={int(c['rerank'])}, decomp={int(c['query_decompose'])}, "
            f"verify={int(c['verify'])} "
            f"-> 正確率={s['correctness']:.2f}, 幻覺率={1-s['faithfulness']:.2f}, "
            f"recall={s['recall']:.2f}, obj={t.objective:.3f}, "
            f"成本={s['avg_latency']:.1f}s/{s['avg_tokens']:.0f}tok"
        )
    table = "\n".join(lines)

    # 最佳配置的失敗分群 + 代表失敗題 (錯誤分析素材)
    best = ranked[0]
    best_rec = cache.get(dict_to_config(best.config))
    clusters = best_rec.get("fail_clusters", {}) if best_rec else {}
    fail_lines = []
    for f in best.failures[:2]:
        fail_lines.append(
            f"  - [{f.get('fail_type','?')}] Q: {f['question']}\n"
            f"    正解: {f['gold']} | 系統答: {f['pred']} "
            f"(recall={f['recall']:.1f}, 忠實={f.get('faithful',0):.0f})\n"
            f"    檢索片段: {f['retrieved_preview'][:240]}"
        )
    fails = "\n".join(fail_lines) if fail_lines else "  (無)"

    tried = ", ".join(sorted(
        "|".join(str(t.config[k]) for k in SEARCH_SPACE) for t in traj.trials))

    return f"""你是 RAG 系統最佳化架構師。目標: 最大化 objective = 正確率 - {LAMBDA}×幻覺率。

合法參數值 (提案必須從這裡選):
  chunk_size      ∈ {SEARCH_SPACE['chunk_size']}
  top_k           ∈ {SEARCH_SPACE['top_k']}
  retriever       ∈ {SEARCH_SPACE['retriever']}
  rerank          ∈ [false, true]   (檢索後重排)
  query_decompose ∈ [false, true]   (多跳問題拆解，HotpotQA 是多跳)
  verify          ∈ [false, true]   (NLI 自我檢查，降幻覺但增成本)

已測配置 (依 objective 由高到低):
{table}

目前最佳配置的失敗分群: {clusters}  (retrieval=檢索沒撈全, generation=撈到卻答錯)
代表失敗題:
{fails}

已測過、請勿重複: {tried}

請依步驟思考:
1. 診斷: 從失敗分群看瓶頸在「檢索」還是「生成/幻覺」?
   - 檢索瓶頸 (retrieval 多) -> 試 query_decompose / rerank / 調大 top_k / 換 retriever
   - 生成瓶頸或幻覺高 -> 試 verify / 調整 chunk_size
2. 探索 vs 利用: 早期提差異大的組合; 後期在最佳配置附近微調。
3. 提案最多 2 組「未測過」且最值得測的配置。

只輸出 JSON，不要其他文字:
{{"reasoning": "你的診斷與推論 (繁體中文，2-4 句)",
  "proposals": [{{"chunk_size":512,"top_k":5,"retriever":"hybrid","rerank":true,"query_decompose":true,"verify":false}}]}}"""


def _call_meta(prompt: str) -> dict:
    client = get_client()
    resp = client.chat.completions.create(
        model=META_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=800,
    )
    m = re.search(r"\{.*\}", resp.choices[0].message.content, re.DOTALL)
    if not m:
        return {"reasoning": "(解析失敗)", "proposals": []}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"reasoning": "(JSON 解析失敗)", "proposals": []}


def _config_from_proposal(d: dict) -> RagConfig:
    return RagConfig(
        chunk_size=d["chunk_size"], top_k=d["top_k"], retriever=d["retriever"],
        rerank=bool(d["rerank"]), query_decompose=bool(d["query_decompose"]),
        verify=bool(d["verify"]),
    )


def _pick_valid_untried(proposals: list[dict], tried: set[tuple]) -> RagConfig | None:
    for d in proposals:
        if not is_valid(d):
            continue
        cfg = _config_from_proposal(d)
        if cfg.key() not in tried:
            return cfg
    return None


def _random_untried(tried: set[tuple], rng: random.Random) -> RagConfig | None:
    pool = [c for c in all_configs() if c.key() not in tried]
    return rng.choice(pool) if pool else None


def opro_search(examples, cache: ResultCache, budget: int, seed: int, traj_path: str,
                n_init: int = 3, verbose: bool = True, log_fn=None) -> Trajectory:
    say = _make_say(log_fn, verbose)
    rng = random.Random(seed)
    grid = all_configs()
    rng.shuffle(grid)

    traj = Trajectory(traj_path)
    traj.reset()

    # 暖身: 與 random baseline 相同 seed 的前 n_init 組 (公平起跑)
    for cfg in grid[:n_init]:
        rec = cache.evaluate_cached(cfg, examples, verbose=verbose)
        traj.add(_trial(rec, source="init"))
        say(f"[暖身] {config_to_dict(cfg)} -> obj={objective(rec['score']):.3f}")

    while len(traj.trials) < budget:
        result = _call_meta(_build_meta_prompt(traj, cache))
        say(f"\n[OPRO 第 {len(traj.trials) - n_init + 1} 輪] 推理: "
            f"{result.get('reasoning', '')}")
        tried = traj.tried_keys()
        cfg = _pick_valid_untried(result.get("proposals", []), tried)
        if cfg is None:
            cfg = _random_untried(tried, rng)
            if cfg is None:
                break
            say("  (LLM 無有效提案，後備隨機抽)")
        say(f"  -> 採用 {config_to_dict(cfg)}")
        rec = cache.evaluate_cached(cfg, examples, verbose=verbose)
        traj.add(_trial(rec, source="opro"))
        say(f"  -> obj={objective(rec['score']):.3f}")

    return traj
