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
    recall: float
    retrieved_preview: str  # 檢索到的 chunk 前段，讓 LLM 看真因 (檢索 vs 生成)

    def as_dict(self) -> dict:
        return {
            "question": self.question, "gold": self.gold, "pred": self.pred,
            "em": self.em, "f1": self.f1, "recall": self.recall,
            "retrieved_preview": self.retrieved_preview,
        }


@dataclass
class EvalScore:
    em: float
    f1: float
    recall: float
    n: int
    details: list[QuestionDetail] = None  # 每題明細 (verbose 不影響，總是收集)

    def as_dict(self) -> dict:
        return {"em": self.em, "f1": self.f1, "recall": self.recall, "n": self.n}

    def worst_cases(self, k: int = 2) -> list[QuestionDetail]:
        """挑 f1 最低的 k 題 (OPRO 錯誤分析用)。"""
        if not self.details:
            return []
        return sorted(self.details, key=lambda d: d.f1)[:k]


def evaluate(cfg: RagConfig, examples: list[Example], verbose: bool = True) -> EvalScore:
    em_sum = f1_sum = rec_sum = 0.0
    details: list[QuestionDetail] = []
    iterator = tqdm(examples, desc=str(cfg.key()), disable=not verbose)
    for ex in iterator:
        try:
            res = run_rag(ex.question, ex.paragraphs, cfg)
        except Exception as e:  # noqa: BLE001  — Demo 期間單題失敗不該中斷整輪
            if verbose:
                tqdm.write(f"[warn] 題目失敗，記 0 分: {e}")
            continue
        em = exact_match(res.answer, ex.answer)
        f1 = f1_score(res.answer, ex.answer)
        rec = retrieval_recall(res.retrieved_chunks, ex.gold_paragraphs)
        em_sum += em
        f1_sum += f1
        rec_sum += rec
        preview = " | ".join(c[:120] for c in res.retrieved_chunks[:3])
        details.append(QuestionDetail(
            question=ex.question, gold=ex.answer, pred=res.answer,
            em=em, f1=f1, recall=rec, retrieved_preview=preview,
        ))

    n = len(examples)
    return EvalScore(em=em_sum / n, f1=f1_sum / n, recall=rec_sum / n, n=n, details=details)


if __name__ == "__main__":
    examples = load_hotpot(n=30, n_hard=15)
    print(f"載入 {len(examples)} 題 "
          f"(hard={sum(e.level == 'hard' for e in examples)})")

    cfg = RagConfig(chunk_size=512, top_k=5, retriever="hybrid")
    score = evaluate(cfg, examples)
    print(f"\n配置 {cfg.key()}")
    print(f"  EM     = {score.em:.3f}")
    print(f"  F1     = {score.f1:.3f}")
    print(f"  recall = {score.recall:.3f}")
