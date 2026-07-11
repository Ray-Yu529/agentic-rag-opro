"""
dataset.py — 自訂資料集: 使用者自己的文件 + (LLM 生成或自備的) QA

讓「調參對象」從 HotpotQA 換成使用者自己的知識庫，後面 sweep / 最佳化 /
評估 / 圖表全部沿用現有管線:

  1. load_corpus()     讀 .txt/.md 檔或整個資料夾 -> 段落語料 (空行分段)
  2a. generate_qa()    用 LLM 對隨機段落生成「可抽取式」短答 QA (存 jsonl 可重用)
  2b. load_qa()        或載入使用者自備的 QA (jsonl / json array)
  3. build_examples()  組成 eval.Example — 每題的 context = 整份語料
                       (真實 RAG 情境: 在你的整個知識庫上調參)

gold 段落的決定順序: QA 裡的 "gold" 欄位 -> 生成來源段落 -> 答案子字串定位；
三者都找不到的題會被剔除 (答案不在文件裡的題無法評檢索品質)。

快取鍵 = 語料 + QA 內容 hash (見 cache.custom_dataset_key)，
改文件或 QA 後舊分數自動失效。

CLI (生成 QA，一次性，之後 run.py/sweep.py 用 --qa 載入):
  python dataset.py --corpus docs/ --n 20 --out data/qa.jsonl
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

from eval import Example, normalize_answer
from optimizer import META_MODEL, chat_json

# QA 生成模型: 預設用 meta 模型 (題目品質比 8b 生成器好)，可在 .env 覆寫
QA_GEN_MODEL = os.environ.get("QA_GEN_MODEL", META_MODEL)
# 掃描 PDF 頁的 VLM 轉錄模型 (只有無文字層的頁面才會用到)
VLM_MODEL = os.environ.get("VLM_MODEL", "meta/llama-3.2-11b-vision-instruct")

PARA_MIN_CHARS = 30      # 太短的段落 (標題/雜訊) 不當語料 (CJK 密度高，別設太嚴)
MAX_PARAGRAPHS = 500     # 語料上限: 這是 demo 級調參台，不是向量資料庫
MAX_GOLD_PER_Q = 2       # 答案出現在很多段時最多取 2 段當 gold (對齊 HotpotQA)
_PAGE_MIN_CHARS = 50     # PDF 頁文字層低於此字數視為掃描頁 -> VLM fallback


# --- 1. 語料載入 -------------------------------------------------------------
def split_paragraphs(text: str) -> list[str]:
    """空行分段 + 壓平段內空白 (跨來源的段落字串才能穩定比對)。"""
    paras = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text)]
    return [p for p in paras if len(p) >= PARA_MIN_CHARS]


# --- 1b. PDF: 文字層優先，掃描頁走 VLM 轉錄 ----------------------------------
def _vlm_transcribe(jpg_b64: str) -> str:
    """把一頁 PDF 的影像丟給 NIM 的 VLM 轉錄成文字 (OpenAI 多模態訊息格式)。"""
    from rag import api_call, get_client   # 局部 import 避免循環
    resp = api_call(
        get_client().chat.completions.create,
        model=VLM_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "text", "text":
                "Transcribe ALL text on this document page into plain text. "
                "Preserve the reading order and paragraph breaks (blank line "
                "between paragraphs). Output ONLY the transcription."},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{jpg_b64}"}},
        ]}],
        temperature=0.0,
        max_tokens=2000,
    )
    return resp.choices[0].message.content.strip()


def extract_pdf_text(source: str | Path | bytes, log_fn=None) -> str:
    """PDF -> 文字。逐頁先抽文字層 (原生數位 PDF 免費零錯字)；
    無文字層/垃圾頁 (掃描頁) 才送 VLM 轉錄。
    注意: VLM 轉錄可能有錯字，會成為評估語料的一部分，掃描檔請抽查品質。"""
    say = log_fn or (lambda *_: None)
    try:
        import fitz  # pymupdf
    except ImportError as e:
        raise SystemExit("PDF 支援需要 pymupdf: pip install pymupdf") from e
    doc = (fitz.open(stream=source, filetype="pdf") if isinstance(source, bytes)
           else fitz.open(str(source)))
    pages: list[str] = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if len(text) >= _PAGE_MIN_CHARS:
            pages.append(text)
            continue
        say(f"  [pdf] 第 {i + 1} 頁無文字層 (掃描頁?)，送 VLM 轉錄…")
        try:
            pix = page.get_pixmap(dpi=96)
            b64 = base64.b64encode(pix.tobytes("jpg")).decode()
            t = _vlm_transcribe(b64)
            if t:
                pages.append(t)
        except Exception as e:  # noqa: BLE001 — 單頁失敗不中斷整份文件
            say(f"  [pdf][warn] 第 {i + 1} 頁 VLM 轉錄失敗，跳過: {e}")
    doc.close()
    return "\n\n".join(pages)


def load_corpus(path: str | Path, log_fn=None) -> list[str]:
    """讀單一檔或資料夾內全部 .txt/.md/.pdf，回傳段落列表。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"找不到語料路徑: {p}")
    files = sorted(f for f in (p.rglob("*") if p.is_dir() else [p])
                   if f.is_file() and f.suffix.lower() in {".txt", ".md", ".pdf"})
    if not files:
        raise ValueError(f"{p} 底下沒有 .txt/.md/.pdf 檔")
    paras: list[str] = []
    for f in files:
        if f.suffix.lower() == ".pdf":
            text = extract_pdf_text(f, log_fn=log_fn)
        else:
            text = f.read_text(encoding="utf-8", errors="ignore")
        paras += split_paragraphs(text)
    if not paras:
        raise ValueError(f"語料是空的 (段落須 ≥{PARA_MIN_CHARS} 字元，空行分段)")
    if len(paras) > MAX_PARAGRAPHS:
        raise ValueError(
            f"語料 {len(paras)} 段超過上限 {MAX_PARAGRAPHS} 段；"
            "這是 demo 級調參台，請先抽一個代表性子集")
    return paras


# --- 2a. LLM 生成 QA ---------------------------------------------------------
_QA_PROMPT = (
    "You are creating a QA evaluation set for a retrieval-augmented QA system.\n"
    "Given the PASSAGE, write ONE factual question answerable using ONLY this "
    "passage, plus its answer.\n"
    "Rules:\n"
    "- The answer must be a short exact span copied from the passage "
    "(a name, date, number, entity, or short phrase — at most 8 words).\n"
    "- The question must be self-contained (understandable without the passage).\n"
    "- Write the question and answer in the same language as the passage.\n"
    'Output JSON only: {"question": "...", "answer": "..."}\n\n'
    "PASSAGE:\n"
)


_MULTIHOP_PROMPT = (
    "You are creating a MULTI-HOP QA evaluation set for a retrieval-augmented "
    "QA system.\nGiven TWO passages, write ONE question that can only be "
    "answered by combining information from BOTH passages, plus its answer.\n"
    "Rules:\n"
    "- The answer must be a short exact span copied from one of the passages "
    "(a name, date, number, entity, or short phrase — at most 8 words).\n"
    "- The question must be self-contained and must genuinely require both "
    "passages (do not write a question answerable from a single passage).\n"
    "- Write the question and answer in the same language as the passages.\n"
    'Output JSON only: {"question": "...", "answer": "..."}\n\n'
)


def _rare_tokens(p: str) -> set[str]:
    """段落的「稀有 token 候選」: 拉丁字 (≥4 字元) + CJK 連續片段的 2-gram。"""
    toks = {t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", p)}
    for run in re.findall(r"[一-鿿]{2,}", p):
        toks.update(run[i:i + 2] for i in range(len(run) - 1))
    return toks


def _related_pairs(paragraphs: list[str], rng: random.Random) -> list[tuple[int, int]]:
    """找「共享稀有 token」的段落對 (多跳題需要兩段有實體關聯，隨機配對會不自然)。
    稀有 = 該 token 只出現在 2~3 個段落。依共享數排序，同分隨機。"""
    tok_paras: dict[str, list[int]] = {}
    for i, p in enumerate(paragraphs):
        for t in _rare_tokens(p):
            tok_paras.setdefault(t, []).append(i)
    pair_score: Counter = Counter()
    for t, idxs in tok_paras.items():
        if 2 <= len(idxs) <= 3:
            for a, b in combinations(idxs, 2):
                pair_score[(a, b)] += 1
    pairs = list(pair_score)
    rng.shuffle(pairs)                       # 同分時順序隨機
    pairs.sort(key=lambda pr: -pair_score[pr])
    return pairs


def _gen_one(prompt: str, passages: list[str]) -> dict | None:
    """呼叫 LLM 生成一題；答案必須是某段落內的 span，否則回 None (品質守門)。"""
    result = chat_json(prompt, temperature=0.3, max_tokens=300, model=QA_GEN_MODEL)
    q = str(result.get("question", "")).strip()
    a = str(result.get("answer", "")).strip()
    if not q or not a:
        return None
    if not any(normalize_answer(a) in normalize_answer(p) for p in passages):
        return None
    return {"question": q, "answer": a, "gold": list(passages)}


def generate_qa(paragraphs: list[str], n: int, seed: int = 42,
                log_fn=None, multihop: bool = False) -> list[dict]:
    """生成可抽取式 QA。答案必須 (正規化後) 出現在來源段落 —— 保證 gold
    可定位、判官與 recall 都有依據。multihop=True 時約一半的題改成
    「跨兩個相關段落」的多跳題 (讓 query_decompose 開關有東西可學)。
    回傳 [{question, answer, gold:[來源段落...], hop}]。"""
    say = log_fn or (lambda *_: None)
    rng = random.Random(seed)
    qa: list[dict] = []

    # 多跳題: 從共享稀有 token 的段落對生成
    if multihop:
        n_mh = n // 2
        for i, j in _related_pairs(paragraphs, rng):
            if len(qa) >= n_mh:
                break
            pair = [paragraphs[i], paragraphs[j]]
            prompt = (_MULTIHOP_PROMPT
                      + f"PASSAGE 1:\n{pair[0]}\n\nPASSAGE 2:\n{pair[1]}")
            rec = _gen_one(prompt, pair)
            if rec:
                rec["hop"] = 2
                qa.append(rec)
                say(f"  QA 生成 (多跳) {len(qa)}/{n}")
        if not qa:
            say("  [warn] 找不到可配對的相關段落，多跳題略過 (全部生成單跳)")

    # 單跳題: 補滿剩餘題數
    order = list(range(len(paragraphs)))
    rng.shuffle(order)
    for idx in order:
        if len(qa) >= n:
            break
        para = paragraphs[idx]
        rec = _gen_one(_QA_PROMPT + para, [para])
        if rec:
            rec["hop"] = 1
            qa.append(rec)
            if len(qa) % 5 == 0 or len(qa) == n:
                say(f"  QA 生成 {len(qa)}/{n}")

    if not qa:
        raise RuntimeError("QA 生成全數失敗 (檢查 API 金鑰 / 語料是否為可讀文字)")
    if len(qa) < n:
        say(f"  [warn] 段落不夠或生成品質不佳，只生成 {len(qa)}/{n} 題")
    return qa


# --- 2b. 自備 QA -------------------------------------------------------------
def load_qa(path: str | Path) -> list[dict]:
    """載入 QA 檔: jsonl (每行一筆) 或 json array。
    每筆至少要有 question / answer；gold (list[str]) 選填。"""
    text = Path(path).read_text(encoding="utf-8")
    try:
        records = json.loads(text)
        if isinstance(records, dict):
            records = [records]
    except json.JSONDecodeError:
        records = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    return validate_qa(records)


def validate_qa(records: list[dict]) -> list[dict]:
    for i, r in enumerate(records):
        if not isinstance(r, dict) or not str(r.get("question", "")).strip() \
                or not str(r.get("answer", "")).strip():
            raise ValueError(
                f"QA 第 {i + 1} 筆格式錯誤: 需要 question 與 answer 欄位，"
                'gold (list[str]) 選填。範例: {"question":"...","answer":"..."}')
    return records


def save_qa(qa: list[dict], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in qa) + "\n",
                 encoding="utf-8")


# --- 3. 組成 Example (沿用現有評估管線) ---------------------------------------
def _locate_gold(paragraphs: list[str], rec: dict) -> list[str]:
    """決定該題的 gold 段落 (recall 的分母):
    1) QA 的 gold 欄位 (可為段落全文或其中一段文字，映射回語料段落)
    2) 答案 (正規化後) 子字串定位到含答案的段落
    找不到回空 list -> 該題會被剔除。"""
    golds: list[str] = []
    for g in rec.get("gold") or []:
        g = " ".join(str(g).split())
        for p in paragraphs:
            if g in p and p not in golds:
                golds.append(p)
                break
    if not golds:
        na = normalize_answer(str(rec["answer"]))
        if na:
            golds = [p for p in paragraphs if na in normalize_answer(p)]
    return golds[:MAX_GOLD_PER_Q]


def build_examples(paragraphs: list[str], qa_records: list[dict],
                   n: int | None = None, seed: int = 42,
                   log_fn=None) -> list[Example]:
    """QA -> Example。context = 整份語料 (真實 RAG: 在整個知識庫上檢索)。
    gold 定位不到 (答案不在文件裡) 的題剔除並警告。"""
    say = log_fn or (lambda *_: None)
    rng = random.Random(seed)
    recs = list(qa_records)
    rng.shuffle(recs)

    examples: list[Example] = []
    dropped = 0
    for r in recs:
        if n is not None and len(examples) >= n:
            break
        golds = _locate_gold(paragraphs, r)
        if not golds:
            dropped += 1
            continue
        examples.append(Example(
            question=str(r["question"]), answer=str(r["answer"]),
            paragraphs=paragraphs, gold_paragraphs=golds, level="custom"))
    if dropped:
        say(f"  [warn] {dropped} 題的答案/gold 在語料中定位不到，已剔除")
    if not examples:
        raise ValueError("沒有可用的題目: 所有 QA 的答案都不在語料裡")
    return examples


def dataset_fingerprint(paragraphs: list[str], qa_records: list[dict]) -> str:
    """語料 + QA 內容指紋: 改文件或改 QA 後，快取分數自動失效。"""
    blob = "\n".join(paragraphs) + "\n##QA##\n" + json.dumps(
        qa_records, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:10]


# --- CLI: 生成 QA (一次性；之後 run.py/sweep.py 用 --qa 載入) -----------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="用 LLM 對你的文件生成 QA 評估集 (可抽取式短答)")
    ap.add_argument("--corpus", required=True, help=".txt/.md/.pdf 檔或資料夾")
    ap.add_argument("--n", type=int, default=20, help="生成題數 (預設 20)")
    ap.add_argument("--out", default="data/qa.jsonl", help="輸出 jsonl 路徑")
    ap.add_argument("--multihop", action="store_true",
                    help="約一半生成跨段落多跳題 (query_decompose 才有東西可學)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    paras = load_corpus(args.corpus, log_fn=print)
    print(f"語料載入: {len(paras)} 段。開始生成 {args.n} 題 (模型: {QA_GEN_MODEL})…")
    qa = generate_qa(paras, n=args.n, seed=args.seed, log_fn=print,
                     multihop=args.multihop)
    save_qa(qa, args.out)
    print(f"\n已存 {len(qa)} 題 -> {args.out}")
    print(f"範例: Q: {qa[0]['question']}\n      A: {qa[0]['answer']}")
    print(f"\n下一步:\n  python run.py --corpus {args.corpus} --qa {args.out}")
