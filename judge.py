"""
judge.py — LLM-as-judge: NLI 忠實度 + 語意正確性

兩個用途:
1. faithfulness (NLI): 答案是否「完全由檢索到的 context 支持」-> 量化幻覺。
2. correctness: 答案語意上是否等同正解 (EM/F1 對自由文本太嚴，補一個語意判官)。

重要: 判官模型刻意與 RAG 生成模型「不同家族」(Mixtral vs Llama)，
      同家族大小模型仍可能有 self-preference bias。
"""

from __future__ import annotations

import os
import re

from rag import api_call, get_client, GEN_MODEL

# 判官與生成模型不同家族以避免自我偏袒；端點 404/不穩時可在 .env 用
# JUDGE_MODEL 覆寫 (建議仍選非 Llama 家族)。
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "mistralai/mixtral-8x7b-instruct-v0.1")

assert JUDGE_MODEL != GEN_MODEL, "判官模型不可與生成模型相同 (會自我偏袒)"


def _ask_yes_no(prompt: str) -> tuple[bool, int]:
    """問判官一個是非題，回傳 (True=YES, 用掉的 token)。容忍模型多話，只抓第一個 YES/NO。"""
    client = get_client()
    resp = api_call(
        client.chat.completions.create,
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=8,
    )
    tokens = resp.usage.total_tokens if resp.usage else 0
    text = resp.choices[0].message.content.upper()
    m = re.search(r"\b(YES|NO)\b", text)
    return m is not None and m.group(1) == "YES", tokens


def judge_faithfulness(answer: str, context_chunks: list[str]) -> tuple[float, int]:
    """
    NLI 蘊含: 答案是否完全由 context 支持?
    回傳 (1.0 忠實 / 0.0 幻覺, 用掉的 token)。
    """
    if not answer.strip():
        return 0.0, 0
    context = "\n".join(f"- {c}" for c in context_chunks)
    prompt = (
        "You are a strict fact-checker. Decide if the ANSWER is fully supported "
        "by the CONTEXT (entailment). If any part of the answer is not stated in "
        "the context, it is NOT supported.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"ANSWER: {answer}\n\n"
        "Is the answer fully supported by the context? Reply with only YES or NO."
    )
    yes, tokens = _ask_yes_no(prompt)
    return (1.0 if yes else 0.0), tokens


def judge_correctness(question: str, pred: str, gold: str) -> tuple[float, int]:
    """
    語意正確性: 預測答案是否與正解等價? (容忍措辭/格式差異)
    回傳 (1.0 / 0.0, 用掉的 token)。
    """
    if not pred.strip():
        return 0.0, 0
    prompt = (
        "You are grading a question-answering system. Decide if the PREDICTED "
        "answer is semantically equivalent to the GOLD answer (ignore phrasing, "
        "case, or extra words as long as the core answer matches).\n\n"
        f"QUESTION: {question}\n"
        f"GOLD: {gold}\n"
        f"PREDICTED: {pred}\n\n"
        "Is the predicted answer correct? Reply with only YES or NO."
    )
    yes, tokens = _ask_yes_no(prompt)
    return (1.0 if yes else 0.0), tokens
