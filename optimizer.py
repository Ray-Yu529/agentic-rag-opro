"""
optimizer.py — 搜索空間 + random baseline (D3) + OPRO 迴路 (D4)

關鍵設計 (省錢可重現):
  搜索空間只有 27 組，先用 sweep.py 把每組的真實分數+失敗案例存成 oracle.json。
  之後 random / OPRO 兩種策略「查 oracle 取分數」(零額外 API)，
  OPRO 唯一的 API 花費就是 meta-agent 的推理呼叫。

去重與合法性檢查放在程式裡 (is_valid / tried_keys)，不靠 prompt 約束 —
  因為 LLM 常常嘴上說探索新區、實際給你重複的舊值。
"""

from __future__ import annotations

import json
import re
import random
from itertools import product
from pathlib import Path

from rag import RagConfig, GEN_MODEL, get_client
from memory.trajectory import Trajectory, Trial

# meta-optimizer 用大模型 (推理量小，每輪 1 次)
META_MODEL = "nvidia/llama-3.1-nemotron-70b-instruct"

ORACLE_PATH = Path(__file__).parent / "oracle.json"

# --- 搜索空間 θ (Demo 縮到 3 維, 共 27 組) --------------------------------
SEARCH_SPACE = {
    "chunk_size": [256, 512, 1024],
    "top_k": [3, 5, 8],
    "retriever": ["bm25", "dense", "hybrid"],
}


def all_configs() -> list[RagConfig]:
    combos = product(SEARCH_SPACE["chunk_size"],
                     SEARCH_SPACE["top_k"],
                     SEARCH_SPACE["retriever"])
    return [RagConfig(chunk_size=c, top_k=k, retriever=r) for c, k, r in combos]


def key_str(cfg_or_dict) -> str:
    """config 的唯一字串鍵 (oracle 查表 / 去重用)。"""
    if isinstance(cfg_or_dict, RagConfig):
        c, k, r = cfg_or_dict.chunk_size, cfg_or_dict.top_k, cfg_or_dict.retriever
    else:
        c, k, r = cfg_or_dict["chunk_size"], cfg_or_dict["top_k"], cfg_or_dict["retriever"]
    return f"{c}-{k}-{r}"


def is_valid(d: dict) -> bool:
    """LLM 提案的合法性檢查 (必須落在 grid 上)。"""
    try:
        return (d["chunk_size"] in SEARCH_SPACE["chunk_size"]
                and d["top_k"] in SEARCH_SPACE["top_k"]
                and d["retriever"] in SEARCH_SPACE["retriever"])
    except (KeyError, TypeError):
        return False


# --- Oracle 存取 -----------------------------------------------------------
def load_oracle(path: Path = ORACLE_PATH) -> dict[str, dict]:
    """回傳 {key_str: {config, score, objective, failures}}。"""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"找不到 {path}。請先跑 sweep.py 產生 oracle (D2.5)。"
        )
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return {key_str(r["config"]): r for r in records}


def _trial_from_oracle(oracle: dict, cfg: RagConfig, source: str) -> Trial:
    rec = oracle[key_str(cfg)]
    return Trial(config=rec["config"], score=rec["score"],
                 objective=rec["objective"], failures=rec["failures"], source=source)


# --- D3: Random search baseline -------------------------------------------
def random_search(oracle: dict, budget: int, seed: int, traj_path: str) -> Trajectory:
    rng = random.Random(seed)
    grid = all_configs()
    rng.shuffle(grid)
    traj = Trajectory(traj_path)
    traj.reset()
    for cfg in grid[:budget]:
        traj.add(_trial_from_oracle(oracle, cfg, source="random"))
    return traj


# --- D4: OPRO meta-optimizer -----------------------------------------------
def _build_meta_prompt(traj: Trajectory) -> str:
    """把軌跡 (已試配置 + 失敗案例) 整理成給 meta-agent 的反思+提案任務。"""
    # 1) 已試配置，依 F1 排名 (先幫它算好排名，LLM 對純數字不敏感)
    ranked = sorted(traj.trials, key=lambda t: t.objective, reverse=True)
    lines = []
    for rank, t in enumerate(ranked, 1):
        c = t.config
        lines.append(
            f"  #{rank} chunk_size={c['chunk_size']}, top_k={c['top_k']}, "
            f"retriever={c['retriever']:6s} -> F1={t.score['f1']:.3f}, "
            f"recall={t.score['recall']:.3f}, EM={t.score['em']:.3f}"
        )
    table = "\n".join(lines)

    # 2) 最佳配置仍答錯的題 (錯誤分析素材，含檢索預覽看真因)
    best = ranked[0]
    fail_lines = []
    for f in best.failures[:2]:
        fail_lines.append(
            f"  - Q: {f['question']}\n"
            f"    正解: {f['gold']} | 系統答: {f['pred']} (F1={f['f1']:.2f}, recall={f['recall']:.2f})\n"
            f"    檢索到的片段: {f['retrieved_preview'][:300]}"
        )
    fails = "\n".join(fail_lines) if fail_lines else "  (無)"

    tried = ", ".join(sorted(key_str(t.config) for t in traj.trials))

    return f"""你是一個 RAG 系統最佳化架構師。我們在用實驗逐步調參，目標是最大化 F1。

合法參數值 (提案必須從這裡選，不可超出):
  chunk_size ∈ {SEARCH_SPACE['chunk_size']}
  top_k      ∈ {SEARCH_SPACE['top_k']}
  retriever  ∈ {SEARCH_SPACE['retriever']}

目前已測試的配置 (依 F1 由高到低):
{table}

目前最佳配置 (#{1}) 仍然答錯的代表題:
{fails}

已測過、請勿重複的配置: {tried}

請依下列步驟思考:
1. 分析: 從失敗案例與分數排名，推論目前的瓶頸在「檢索」(recall 低) 還是「生成/雜訊」(recall 高但 F1 低)?
2. 策略: 若早期 (探索) 請提差異大的組合; 若已逼近最佳 (利用) 請在最佳配置附近微調。
3. 提案: 給出最多 2 組「未測過」且最值得測的配置。

只輸出 JSON，格式如下，不要其他文字:
{{"reasoning": "你的分析與推論 (繁體中文，2-4 句)",
  "proposals": [{{"chunk_size": 512, "top_k": 5, "retriever": "hybrid"}}]}}"""


def _call_meta(prompt: str) -> dict:
    """呼叫 meta-LLM 並解析 JSON。失敗回空 proposals。"""
    client = get_client()
    resp = client.chat.completions.create(
        model=META_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=600,
    )
    text = resp.choices[0].message.content
    m = re.search(r"\{.*\}", text, re.DOTALL)  # 容忍模型多吐的前後綴
    if not m:
        return {"reasoning": "(解析失敗)", "proposals": []}
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"reasoning": "(JSON 解析失敗)", "proposals": []}


def _pick_valid_untried(proposals: list[dict], tried: set[tuple],
                        oracle: dict) -> RagConfig | None:
    """程式端硬性篩選: 合法 + 未試過 + oracle 有分數。"""
    for d in proposals:
        if not is_valid(d):
            continue
        cfg = RagConfig(chunk_size=d["chunk_size"], top_k=d["top_k"], retriever=d["retriever"])
        if cfg.key() in tried:
            continue
        if key_str(cfg) not in oracle:
            continue
        return cfg
    return None


def _random_untried(tried: set[tuple], oracle: dict, rng: random.Random) -> RagConfig | None:
    """LLM 沒給有效提案時的後備: 隨機抓一個沒試過的。"""
    pool = [c for c in all_configs()
            if c.key() not in tried and key_str(c) in oracle]
    return rng.choice(pool) if pool else None


def opro_search(oracle: dict, budget: int, seed: int, traj_path: str,
                n_init: int = 3, verbose: bool = True) -> Trajectory:
    """
    n_init 組隨機暖身 (與 random_search 同 seed，公平起跑)，
    其餘預算交給 OPRO meta-agent 反思後提案。
    """
    rng = random.Random(seed)
    grid = all_configs()
    rng.shuffle(grid)

    traj = Trajectory(traj_path)
    traj.reset()

    # 暖身: 與 random baseline 相同的前 n_init 組
    for cfg in grid[:n_init]:
        traj.add(_trial_from_oracle(oracle, cfg, source="init"))

    # OPRO 迴路
    while len(traj.trials) < budget:
        prompt = _build_meta_prompt(traj)
        result = _call_meta(prompt)
        if verbose:
            print(f"\n[OPRO round {len(traj.trials) - n_init + 1}] "
                  f"reasoning: {result.get('reasoning', '')}")

        tried = traj.tried_keys()
        cfg = _pick_valid_untried(result.get("proposals", []), tried, oracle)
        if cfg is None:
            cfg = _random_untried(tried, oracle, rng)  # 後備
            if cfg is None:
                break  # 全部試完了
            if verbose:
                print("  (LLM 無有效提案，後備隨機抽)")
        if verbose:
            print(f"  -> 採用配置 {key_str(cfg)}")
        traj.add(_trial_from_oracle(oracle, cfg, source="opro"))

    return traj
