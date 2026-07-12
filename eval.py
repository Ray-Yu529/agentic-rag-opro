"""
eval.py — HotpotQA 子集載入 + 評估指標 (Demo 版)

提供:
- load_hotpot(): 分層抽樣的測試集 (困難多跳 + 一般)
- evaluate(cfg, examples): 對一組 RagConfig 跑完所有題，回傳 EM / F1 / recall@k
- __main__: 對預設 config 跑一遍，當作冒煙測試與 D2 完成標準

指標:
- EM / F1   : SQuAD 風格的字串正確性
- recall@k  : 檢索到的 chunk 是否覆蓋到 gold supporting 段落 (檢索品質)

這支檔輸出的 (em, f1, recall) 就是 OPRO/Optuna 要最佳化的目標訊號。
"""

from __future__ import annotations

import hashlib
import re
import string
import random
from collections import Counter
from dataclasses import dataclass

from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

from rag import RagConfig, run_rag, is_abstain

load_dotenv()  # 讀 .env 的 NVIDIA_API_KEY

MAX_ERROR_RATE = 0.10   # 單輪容忍的單題失敗比例；超過就整輪作廢 (不落 cache)
MIN_RACE_QUESTIONS = 5  # racing: 至少評這麼多題才允許提早停止


class EvalPruned(Exception):
    """Racing 提早停止: objective 的樂觀上界已追不上目前最佳配置。
    帶部分平均分數供軌跡/排行榜顯示 (不會寫入 cache)。"""

    def __init__(self, n_done: int, bound: float, partial_score: dict):
        super().__init__(f"提早停止 @{n_done} 題 (上界 {bound:.3f})")
        self.n_done = n_done
        self.bound = bound
        self.partial_score = partial_score


# --- 資料載入 + 分層抽樣 ---------------------------------------------------
@dataclass
class Example:
    question: str
    answer: str
    paragraphs: list[str]        # 該題 context 的段落文字 (含 distractor)
    gold_paragraphs: list[str]   # supporting_facts 指到的 gold 段落文字
    level: str                   # "easy" | "medium" | "hard"


def _build_example(row: dict) -> Example:
    titles = row["context"]["title"]
    sentences = row["context"]["sentences"]
    # 每個段落 = 該 title 底下所有句子串起來
    paragraphs = [" ".join(sents).strip() for sents in sentences]
    title_to_para = dict(zip(titles, paragraphs))

    gold_titles = set(row["supporting_facts"]["title"])
    gold_paragraphs = [title_to_para[t] for t in gold_titles if t in title_to_para]

    return Example(
        question=row["question"],
        answer=row["answer"],
        paragraphs=paragraphs,
        gold_paragraphs=gold_paragraphs,
        level=row.get("level", "medium"),
    )


def load_hotpot(n: int = 30, n_hard: int = 15, seed: int = 42) -> list[Example]:
    """
    從 HotpotQA distractor validation 抽 n 題，分層: n_hard 題 hard + 其餘 easy/medium。
    這樣測試集才對「參數變化」有區分度 (純隨機會抽到一堆太簡單的題)。
    """
    # 明確用 hotpotqa/hotpot_qa (Hub 上已是 parquet 格式)，
    # datasets 3.x+ 移除 script 載入後仍可正常使用
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    rng = random.Random(seed)

    hard_idx = [i for i, lv in enumerate(ds["level"]) if lv == "hard"]
    other_idx = [i for i, lv in enumerate(ds["level"]) if lv != "hard"]
    rng.shuffle(hard_idx)
    rng.shuffle(other_idx)

    picked = hard_idx[:n_hard] + other_idx[: max(0, n - n_hard)]
    rng.shuffle(picked)
    return [_build_example(ds[i]) for i in picked]


# --- SQuAD 風格 EM / F1 (含 CJK 支援) ---------------------------------------
_CJK_RE = re.compile(r"[一-鿿]")
_CJK_PUNCT = "，。、；：「」『』（）《》〈〉！？．·—～“”‘’"


def normalize_answer(s: str) -> str:
    s = s.lower()
    drop = set(string.punctuation) | set(_CJK_PUNCT)
    s = "".join(ch for ch in s if ch not in drop)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    if _CJK_RE.search(s):
        s = s.replace(" ", "")   # CJK 不以空白斷詞；去空白讓 EM/子字串比對穩定
    return s


def _tokens(s: str) -> list[str]:
    """F1 的斷詞: 含 CJK 用字元級 (空白斷詞會把整句中文當一個 token，F1 失真)。"""
    return list(s) if _CJK_RE.search(s) else s.split()


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    pred_toks = _tokens(normalize_answer(pred))
    gold_toks = _tokens(normalize_answer(gold))
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


# --- 檢索 recall ----------------------------------------------------------
def retrieval_recall(retrieved_chunks: list[str], gold_paragraphs: list[str],
                     min_coverage: float = 0.5) -> float:
    """
    gold 段落被檢索 chunk 覆蓋的「字元比例」達 min_coverage 才算命中。
    以前任一子字串就整段算命中，會偏袒小 chunk (撈到 256 字皮毛就滿分 recall，
    但生成器實際拿到的資訊可能不含答案)。
    (chunk 不跨段落，子字串包含關係成立；chunk_overlap=0 時同段 chunk 不重疊。)
    """
    if not gold_paragraphs:
        return 0.0
    hits = 0
    for gold in gold_paragraphs:
        covered = sum(len(c) for c in set(retrieved_chunks) if c and c in gold)
        if covered >= min_coverage * len(gold):
            hits += 1
    return hits / len(gold_paragraphs)


# --- 評估迴圈 --------------------------------------------------------------
@dataclass
class QuestionDetail:
    """單題結果，供 OPRO 挑失敗案例做錯誤分析。"""
    question: str
    gold: str
    pred: str
    em: float
    f1: float
    recall: float           # 檢索品質: gold 段落是否被撈到
    faithful: float         # NLI 忠實度: 答案是否被 context 支持 (1/0)
    correct: float          # LLM 判官語意正確性 (1/0)
    fail_type: str          # "ok" | "retrieval" | "generation" — 失敗分群用
    retrieved_preview: str  # 檢索到的 chunk 前段，讓 LLM 看真因

    def as_dict(self) -> dict:
        return {
            "question": self.question, "gold": self.gold, "pred": self.pred,
            "em": self.em, "f1": self.f1, "recall": self.recall,
            "faithful": self.faithful, "correct": self.correct,
            "fail_type": self.fail_type, "retrieved_preview": self.retrieved_preview,
        }


@dataclass
class EvalScore:
    em: float
    f1: float
    recall: float
    faithfulness: float        # 平均忠實度 (1 - 幻覺率)
    correctness: float         # LLM 判官平均正確率
    abstain_rate: float        # verify 守門員棄答比例 (棄答≠幻覺，單獨追蹤)
    avg_latency: float         # 成本: 每題平均秒數
    avg_tokens: float          # 成本: 每題平均 token
    n: int                     # 實際成功評估的題數 (分母)
    n_errors: int = 0          # API 失敗被跳過的題數 (過多會整輪作廢，見 evaluate)
    judge_agreement: float | None = None   # 主/稽核判官一致率 (抽查關閉時 None)
    details: list[QuestionDetail] = None

    def as_dict(self) -> dict:
        return {"em": self.em, "f1": self.f1, "recall": self.recall,
                "faithfulness": self.faithfulness, "correctness": self.correctness,
                "abstain_rate": self.abstain_rate,
                "avg_latency": self.avg_latency, "avg_tokens": self.avg_tokens,
                "n": self.n, "n_errors": self.n_errors,
                "judge_agreement": self.judge_agreement}

    def hallucination_rate(self) -> float:
        return 1.0 - self.faithfulness

    def worst_cases(self, k: int = 2) -> list[QuestionDetail]:
        """挑最該檢討的 k 題: 先看答錯 (correct=0)，再依 f1 排。"""
        if not self.details:
            return []
        return sorted(self.details, key=lambda d: (d.correct, d.f1))[:k]

    def fail_clusters(self) -> dict[str, int]:
        """失敗分群計數，餵給 OPRO 推論瓶頸在檢索還是生成。"""
        clusters = {"retrieval": 0, "generation": 0, "abstain": 0}
        for d in (self.details or []):
            if d.fail_type in clusters:
                clusters[d.fail_type] += 1
        return clusters


def _classify(recall: float, correct: float, abstained: bool) -> str:
    """棄答自成一群 (讓 OPRO 看得到答案被守門員/生成器擋掉了多少)。
    其餘先看答對與否: 答對就是 ok (HotpotQA 撈到一段 gold 也常答得對)。
    答錯才分群: recall<1 -> 檢索沒撈全; 撈全了仍答錯 -> 生成端問題。"""
    if abstained:
        return "abstain"
    if correct >= 1.0:
        return "ok"
    if recall < 1.0:
        return "retrieval"
    return "generation"


def _audit_pick(question: str) -> bool:
    """雙判官抽查的確定性抽樣: 同一題在所有配置下都被抽 (跨配置可比)。"""
    from judge import AUDIT_RATE
    if AUDIT_RATE <= 0:
        return False
    h = int(hashlib.sha1(question.encode("utf-8")).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < AUDIT_RATE


def _partial_score(em_s, f1_s, rec_s, faith_s, corr_s, abst_s, lat_s, tok_s,
                   n_ok: int, errors: int) -> dict:
    """racing 提早停止時的部分平均 (供軌跡顯示，欄位對齊 EvalScore.as_dict)。"""
    n = max(1, n_ok)
    return {"em": em_s / n, "f1": f1_s / n, "recall": rec_s / n,
            "faithfulness": faith_s / n, "correctness": corr_s / n,
            "abstain_rate": abst_s / n, "avg_latency": lat_s / n,
            "avg_tokens": tok_s / n, "n": n_ok, "n_errors": errors,
            "pruned": True}


def evaluate(cfg: RagConfig, examples: list[Example], verbose: bool = True,
             prune_below: float | None = None, prune_lam: float = 0.5) -> EvalScore:
    """評估一組配置。prune_below 給定時啟用 racing:
    評到一半若「剩餘題全對且零幻覺」的樂觀上界仍低於 prune_below，
    拋 EvalPruned 提早停止，省下注定追不上的評估成本。"""
    # 局部 import 避免模組載入順序問題
    from judge import audit_correctness, judge_correctness, judge_faithfulness

    em_s = f1_s = rec_s = faith_s = corr_s = abst_s = lat_s = tok_s = 0.0
    agree_s, audit_n = 0.0, 0
    errors = 0
    details: list[QuestionDetail] = []
    iterator = tqdm(examples, desc=str(cfg.key()), disable=not verbose)
    for ex in iterator:
        # judge 呼叫也要在 try 內: 一題爆 429/網路錯誤不該讓整輪評估
        # (及已花掉的 API 成本) 全部作廢
        try:
            res = run_rag(ex.question, ex.paragraphs, cfg)
            abstained = res.abstained or is_abstain(res.answer)
            if abstained:
                # 棄答統一計分 (不論 verify 開關): 沒有捏造內容 -> 不算幻覺；
                # 答錯由 correctness=0 懲罰，另記 abstain_rate。
                # 不統一的話 verify=False 的誠實棄答會被判官記成幻覺，
                # objective 就隱性補貼 verify。順便省下兩次判官呼叫。
                faithful, correct = 1.0, 0.0
            else:
                # verify 已算過忠實度就重用，否則補算 (確保所有配置都有此指標可比)
                if res.faithful is not None:
                    faithful = res.faithful
                else:
                    faithful, _ = judge_faithfulness(res.answer, res.retrieved_chunks)
                correct, _ = judge_correctness(ex.question, res.answer, ex.answer)
                # 雙判官交叉驗證 (JUDGE_AUDIT_RATE>0 才啟用): 抽查一致率
                if _audit_pick(ex.question):
                    correct2, _ = audit_correctness(ex.question, res.answer, ex.answer)
                    agree_s += float(correct2 == correct)
                    audit_n += 1
        except Exception as e:  # noqa: BLE001  — 單題偶發失敗不中斷，但過多會整輪作廢
            errors += 1
            if verbose:
                tqdm.write(f"[warn] 題目失敗，跳過: {e}")
            continue
        em = exact_match(res.answer, ex.answer)
        f1 = f1_score(res.answer, ex.answer)
        rec = retrieval_recall(res.retrieved_chunks, ex.gold_paragraphs)

        em_s += em; f1_s += f1; rec_s += rec; faith_s += faithful; corr_s += correct
        abst_s += float(abstained); lat_s += res.latency; tok_s += res.tokens
        details.append(QuestionDetail(
            question=ex.question, gold=ex.answer, pred=res.answer,
            em=em, f1=f1, recall=rec, faithful=faithful, correct=correct,
            fail_type=_classify(rec, correct, abstained),
            retrieved_preview=" | ".join(c[:120] for c in res.retrieved_chunks[:3]),
        ))

        # --- racing: 樂觀上界追不上就提早停止 --------------------------------
        done = len(details) + errors
        if (prune_below is not None and len(details) > 0
                and done >= max(MIN_RACE_QUESTIONS, len(examples) // 3)):
            remaining = len(examples) - done
            n_final = len(details) + remaining      # 假設剩餘題全部評估成功
            halluc_so_far = len(details) - faith_s
            # 上界: 剩餘題全對、零幻覺 (成本項 μ 只會再扣分，忽略仍是合法上界)
            bound = (corr_s + remaining) / n_final - prune_lam * halluc_so_far / n_final
            if bound < prune_below:
                if verbose:
                    tqdm.write(f"[race] 提早停止 @{done} 題: 上界 {bound:.3f} "
                               f"< 目前最佳 {prune_below:.3f}")
                raise EvalPruned(done, bound, _partial_score(
                    em_s, f1_s, rec_s, faith_s, corr_s, abst_s, lat_s, tok_s,
                    len(details), errors))

    n_ok = len(details)
    # 失敗題過多 (例如整段 429/5xx 風暴) -> 分數已失真，拋出且「不落 cache」，
    # 否則壞分數會被永久快取、三種策略之後都拿它來比。
    # 已完成的其他配置仍在 cache 裡，稍後重跑會從中斷處續跑。
    if n_ok == 0 or errors / len(examples) > MAX_ERROR_RATE:
        raise RuntimeError(
            f"評估失敗題過多 ({errors}/{len(examples)})，本輪分數作廢、不寫入快取；"
            "多半是 API 持續 429/5xx，稍後重跑即可續跑。")
    return EvalScore(em=em_s / n_ok, f1=f1_s / n_ok, recall=rec_s / n_ok,
                     faithfulness=faith_s / n_ok, correctness=corr_s / n_ok,
                     abstain_rate=abst_s / n_ok,
                     avg_latency=lat_s / n_ok, avg_tokens=tok_s / n_ok,
                     n=n_ok, n_errors=errors,
                     judge_agreement=(agree_s / audit_n if audit_n else None),
                     details=details)


if __name__ == "__main__":
    examples = load_hotpot(n=30, n_hard=15)
    print(f"載入 {len(examples)} 題 "
          f"(hard={sum(e.level == 'hard' for e in examples)})")

    cfg = RagConfig(chunk_size=512, top_k=5, retriever="hybrid")
    score = evaluate(cfg, examples)
    print(f"\n配置 {cfg.key()}")
    print(f"  EM           = {score.em:.3f}")
    print(f"  F1           = {score.f1:.3f}")
    print(f"  recall       = {score.recall:.3f}")
    print(f"  correctness  = {score.correctness:.3f} (LLM judge)")
    print(f"  faithfulness = {score.faithfulness:.3f} (幻覺率 {score.hallucination_rate():.3f})")
    print(f"  abstain rate = {score.abstain_rate:.3f} (守門員棄答比例)")
    if score.judge_agreement is not None:
        print(f"  judge agree  = {score.judge_agreement:.3f} (雙判官一致率；<0.8 該換判官)")
    print(f"  cost         = {score.avg_latency:.2f}s, {score.avg_tokens:.0f} tok/題")
