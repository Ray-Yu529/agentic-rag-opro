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
import os
import re
import random
from itertools import product

from openai import BadRequestError

from rag import RagConfig, api_call, get_client
from cache import ResultCache, config_to_dict, dict_to_config
from memory.trajectory import Trajectory, Trial

# meta-optimizer (推理量小)；端點不穩時可在 .env 用 META_MODEL 覆寫
META_MODEL = os.environ.get("META_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct")
DEFAULT_LAMBDA = 0.5  # 幻覺懲罰權重預設值 (各 search 函式可用 lam 參數覆寫)


def chat_json(prompt: str, temperature: float, max_tokens: int,
              model: str = META_MODEL) -> dict:
    """呼叫 LLM 並解析 JSON (預設 meta 模型，dataset.py 的 QA 生成也共用)。
    優先用 API 的 JSON mode；模型不支援時退回一般模式 + regex 擷取。
    解析失敗回 {}。"""
    client = get_client()
    kwargs = dict(model=model,
                  messages=[{"role": "user", "content": prompt}],
                  temperature=temperature, max_tokens=max_tokens)
    try:
        resp = api_call(client.chat.completions.create,
                        response_format={"type": "json_object"}, **kwargs)
    except BadRequestError:  # 端點不支援 JSON mode (400) 才退回一般模式；其他錯誤照拋
        resp = api_call(client.chat.completions.create, **kwargs)
    m = re.search(r"\{.*\}", resp.choices[0].message.content, re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}

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


def objective(score: dict, lam: float = DEFAULT_LAMBDA) -> float:
    """多目標純量化: 正確率 - λ·幻覺率。"""
    hallucination = 1.0 - score.get("faithfulness", 1.0)
    return score.get("correctness", 0.0) - lam * hallucination


def _trial(record: dict, source: str, lam: float) -> Trial:
    return Trial(config=record["config"], score=record["score"],
                 objective=objective(record["score"], lam),
                 failures=record["failures"], source=source,
                 lam=lam)  # 記下當時的 λ，避免不同權重的軌跡混著比


def _make_say(log_fn, verbose):
    """log_fn 優先 (後端用); 否則 verbose 時印出。"""
    if log_fn is not None:
        return log_fn
    return print if verbose else (lambda *_: None)


# --- D3: Random search baseline -------------------------------------------
def random_search(examples, cache: ResultCache, budget: int, seed: int,
                  traj_path: str, verbose: bool = True, log_fn=None,
                  lam: float = DEFAULT_LAMBDA) -> Trajectory:
    say = _make_say(log_fn, verbose)
    rng = random.Random(seed)
    grid = all_configs()
    rng.shuffle(grid)
    traj = Trajectory(traj_path)
    traj.reset()
    for cfg in grid[:budget]:
        rec = cache.evaluate_cached(cfg, examples, verbose=verbose)
        traj.add(_trial(rec, source="random", lam=lam))
        say(f"[random] 評估 {config_to_dict(cfg)} -> obj={objective(rec['score'], lam):.3f}")
    return traj


# --- D4: OPRO meta-optimizer -----------------------------------------------
def _marginal_lines(traj: Trajectory) -> str:
    """每個參數取值的平均 objective (已試樣本內的邊際統計)，
    讓 meta LLM 的診斷有數據支撐，而不是只看排行榜猜。"""
    lines = []
    for dim, values in SEARCH_SPACE.items():
        parts = []
        for v in values:
            objs = [t.objective for t in traj.trials if t.config.get(dim) == v]
            parts.append(f"{v}→{sum(objs) / len(objs):.3f}(n={len(objs)})"
                         if objs else f"{v}→未試")
        lines.append(f"  {dim}: " + ", ".join(parts))
    return "\n".join(lines)


def _build_meta_prompt(traj: Trajectory, cache: ResultCache,
                       lam: float = DEFAULT_LAMBDA) -> str:
    ranked = sorted(traj.trials, key=lambda t: t.objective, reverse=True)
    lines = []
    for rank, t in enumerate(ranked, 1):
        c, s = t.config, t.score
        lines.append(
            f"  #{rank} chunk={c['chunk_size']}, top_k={c['top_k']}, ret={c['retriever']:6s}, "
            f"rerank={int(c['rerank'])}, decomp={int(c['query_decompose'])}, "
            f"verify={int(c['verify'])} "
            f"-> 正確率={s['correctness']:.2f}, 幻覺率={1-s['faithfulness']:.2f}, "
            f"棄答率={s.get('abstain_rate', 0):.2f}, "
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

    return f"""你是 RAG 系統最佳化架構師。目標: 最大化 objective = 正確率 - {lam}×幻覺率。

合法參數值 (提案必須從這裡選):
  chunk_size      ∈ {SEARCH_SPACE['chunk_size']}
  top_k           ∈ {SEARCH_SPACE['top_k']}
  retriever       ∈ {SEARCH_SPACE['retriever']}
  rerank          ∈ [false, true]   (檢索後重排)
  query_decompose ∈ [false, true]   (多跳問題拆解，HotpotQA 是多跳)
  verify          ∈ [false, true]   (NLI 自我檢查，降幻覺但增成本)

已測配置 (依 objective 由高到低):
{table}

各參數取值的平均 objective (已測樣本內的邊際統計，樣本少僅供參考):
{_marginal_lines(traj)}

目前最佳配置的失敗分群: {clusters}
(retrieval=檢索沒撈全, generation=撈到卻答錯, abstain=守門員/生成器棄答)
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
    result = chat_json(prompt, temperature=0.4, max_tokens=800)
    if not result:
        return {"reasoning": "(JSON 解析失敗)", "proposals": []}
    return result


def _config_from_proposal(d: dict) -> RagConfig:
    return RagConfig(
        chunk_size=d["chunk_size"], top_k=d["top_k"], retriever=d["retriever"],
        rerank=bool(d["rerank"]), query_decompose=bool(d["query_decompose"]),
        verify=bool(d["verify"]),
    )


def _valid_untried(proposals: list[dict], tried: set[tuple]) -> list[RagConfig]:
    """合法且未測過的提案 (含提案間去重)。"""
    out, seen = [], set(tried)
    for d in proposals:
        if not is_valid(d):
            continue
        cfg = _config_from_proposal(d)
        if cfg.key() not in seen:
            seen.add(cfg.key())
            out.append(cfg)
    return out


def _random_untried(tried: set[tuple], rng: random.Random) -> RagConfig | None:
    pool = [c for c in all_configs() if c.key() not in tried]
    return rng.choice(pool) if pool else None


def opro_search(examples, cache: ResultCache, budget: int, seed: int, traj_path: str,
                n_init: int = 3, verbose: bool = True, log_fn=None,
                lam: float = DEFAULT_LAMBDA) -> Trajectory:
    say = _make_say(log_fn, verbose)
    rng = random.Random(seed)
    grid = all_configs()
    rng.shuffle(grid)

    traj = Trajectory(traj_path)
    traj.reset()

    # 暖身: 與 random baseline 相同 seed 的前 n_init 組 (公平起跑)
    for cfg in grid[:n_init]:
        rec = cache.evaluate_cached(cfg, examples, verbose=verbose)
        traj.add(_trial(rec, source="init", lam=lam))
        say(f"[暖身] {config_to_dict(cfg)} -> obj={objective(rec['score'], lam):.3f}")

    while len(traj.trials) < budget:
        result = _call_meta(_build_meta_prompt(traj, cache, lam))
        say(f"\n[OPRO 第 {len(traj.trials) - n_init + 1} 輪] 推理: "
            f"{result.get('reasoning', '')}")
        tried = traj.tried_keys()
        # prompt 允許最多 2 組提案；兩組都評，讓每次 meta 呼叫的成本攤平
        cfgs = _valid_untried(result.get("proposals", []), tried)[:2]
        if not cfgs:
            cfg = _random_untried(tried, rng)
            if cfg is None:
                break
            say("  (LLM 無有效提案，後備隨機抽)")
            cfgs = [cfg]
        for cfg in cfgs:
            if len(traj.trials) >= budget:
                break
            say(f"  -> 採用 {config_to_dict(cfg)}")
            rec = cache.evaluate_cached(cfg, examples, verbose=verbose)
            traj.add(_trial(rec, source="opro", lam=lam))
            say(f"  -> obj={objective(rec['score'], lam):.3f}")

    return traj
