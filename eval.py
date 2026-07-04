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

import re
import string
import random
from collections import Counter
from dataclasses import dataclass

from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm

from rag import RagConfig, run_rag

load_dotenv()  # 讀 .env 的 NVIDIA_API_KEY


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
    ds = load_dataset("hotpot_qa", "distractor", split="validation")
    rng = random.Random(seed)

    hard_idx = [i for i, lv in enumerate(ds["level"]) if lv == "hard"]
    other_idx = [i for i, lv in enumerate(ds["level"]) if lv != "hard"]
    rng.shuffle(hard_idx)
    rng.shuffle(other_idx)

    picked = hard_idx[:n_hard] + other_idx[: max(0, n - n_hard)]
    rng.shuffle(picked)
    return [_build_example(ds[i]) for i in picked]


# --- SQuAD 風格 EM / F1 ----------------------------------------------------
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
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
def retrieval_recall(retrieved_chunks: list[str], gold_paragraphs: list[str]) -> float:
    """
    gold 段落只要有任一被檢索到的 chunk 是它的子字串，就算命中。
    (chunk 不跨段落，所以子字串包含關係成立。)
    """
    if not gold_paragraphs:
        return 0.0
    hits = 0
    for gold in gold_paragraphs:
        if any(chunk in gold for chunk in retrieved_chunks):
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
    n: int
    details: list[QuestionDetail] = None

    def as_dict(self) -> dict:
        return {"em": self.em, "f1": self.f1, "recall": self.recall,
                "faithfulness": self.faithfulness, "correctness": self.correctness,
                "abstain_rate": self.abstain_rate,
                "avg_latency": self.avg_latency, "avg_tokens": self.avg_tokens, "n": self.n}

    def hallucination_rate(self) -> float:
        return 1.0 - self.faithfulness

    def worst_cases(self, k: int = 2) -> list[QuestionDetail]:
        """挑最該檢討的 k 題: 先看答錯 (correct=0)，再依 f1 排。"""
        if not self.details:
            return []
        return sorted(self.details, key=lambda d: (d.correct, d.f1))[:k]

    def fail_clusters(self) -> dict[str, int]:
        """失敗分群計數，餵給 OPRO 推論瓶頸在檢索還是生成。"""
        clusters = {"retrieval": 0, "generation": 0}
        for d in (self.details or []):
            if d.fail_type in clusters:
                clusters[d.fail_type] += 1
        return clusters


def _classify(recall: float, correct: float) -> str:
    """先看答對與否: 答對就是 ok (HotpotQA 撈到一段 gold 也常答得對)。
    答錯才分群: recall<1 -> 檢索沒撈全; 撈全了仍答錯 -> 生成端問題。"""
    if correct >= 1.0:
        return "ok"
    if recall < 1.0:
        return "retrieval"
    return "generation"


def evaluate(cfg: RagConfig, examples: list[Example], verbose: bool = True) -> EvalScore:
    # 局部 import 避免模組載入順序問題
    from judge import judge_correctness, judge_faithfulness

    em_s = f1_s = rec_s = faith_s = corr_s = abst_s = lat_s = tok_s = 0.0
    details: list[QuestionDetail] = []
    iterator = tqdm(examples, desc=str(cfg.key()), disable=not verbose)
    for ex in iterator:
        # judge 呼叫也要在 try 內: 一題爆 429/網路錯誤不該讓整輪評估
        # (及已花掉的 API 成本) 全部作廢
        try:
            res = run_rag(ex.question, ex.paragraphs, cfg)
            # verify 已算過忠實度就重用，否則補算 (確保所有配置都有此指標可比)
            if res.faithful is not None:
                faithful = res.faithful
            else:
                faithful, _ = judge_faithfulness(res.answer, res.retrieved_chunks)
            correct, _ = judge_correctness(ex.question, res.answer, ex.answer)
        except Exception as e:  # noqa: BLE001  — 單題失敗不該中斷整輪
            if verbose:
                tqdm.write(f"[warn] 題目失敗，記 0 分: {e}")
            continue
        em = exact_match(res.answer, ex.answer)
        f1 = f1_score(res.answer, ex.answer)
        rec = retrieval_recall(res.retrieved_chunks, ex.gold_paragraphs)

        em_s += em; f1_s += f1; rec_s += rec; faith_s += faithful; corr_s += correct
        abst_s += float(res.abstained); lat_s += res.latency; tok_s += res.tokens
        details.append(QuestionDetail(
            question=ex.question, gold=ex.answer, pred=res.answer,
            em=em, f1=f1, recall=rec, faithful=faithful, correct=correct,
            fail_type=_classify(rec, correct),
            retrieved_preview=" | ".join(c[:120] for c in res.retrieved_chunks[:3]),
        ))

    n = len(examples)
    return EvalScore(em=em_s / n, f1=f1_s / n, recall=rec_s / n,
                     faithfulness=faith_s / n, correctness=corr_s / n,
                     abstain_rate=abst_s / n,
                     avg_latency=lat_s / n, avg_tokens=tok_s / n,
                     n=n, details=details)


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
    print(f"  cost         = {score.avg_latency:.2f}s, {score.avg_tokens:.0f} tok/題")
