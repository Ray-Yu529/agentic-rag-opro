"""自訂資料集 (文件 -> 語料 -> QA -> Example) 的離線測試 (不打 API)。"""

import json

import pytest

import dataset as ds
from dataset import (_locate_gold, _related_pairs, build_examples,
                     dataset_fingerprint, generate_qa, load_corpus, load_qa,
                     save_qa, split_paragraphs, validate_qa)

TEXT = """# 標題短

巴黎鐵塔位於法國巴黎，於 1889 年完工，是世界著名的地標建築物之一。

聖母院是巴黎另一座知名建築，
以哥德式風格聞名，始建於 1163 年。


短段落
"""

QA = [{"question": "巴黎鐵塔何時完工?", "answer": "1889 年"},
      {"question": "聖母院始建於哪一年?", "answer": "1163 年",
       "gold": ["始建於 1163 年"]}]


@pytest.fixture()
def paras():
    return split_paragraphs(TEXT)


def test_split_paragraphs(paras):
    assert len(paras) == 2                       # 標題與短段落被過濾
    assert "1163" in paras[1] and "\n" not in paras[1]   # 段內換行壓平


def test_load_corpus(tmp_path):
    (tmp_path / "a.txt").write_text(TEXT, encoding="utf-8")
    (tmp_path / "b.md").write_text(
        "Mount Everest is the highest mountain on Earth, located in the "
        "Himalayas of Asia.", encoding="utf-8")
    (tmp_path / "c.docx").write_text("ignored", encoding="utf-8")
    assert len(load_corpus(tmp_path)) == 3       # a 2 段 + b 1 段, docx 跳過
    assert len(load_corpus(tmp_path / "a.txt")) == 2
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "nope.txt")


def test_pdf_text_layer(tmp_path):
    fitz = pytest.importorskip("fitz")           # pymupdf
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Retrieval augmented generation lets a model "
                               "ground its answers in retrieved documents.")
    pdf = tmp_path / "t.pdf"
    doc.save(pdf)
    doc.close()
    text = ds.extract_pdf_text(pdf)              # 有文字層 -> 不會呼叫 VLM
    assert "Retrieval augmented generation" in text
    assert len(load_corpus(pdf)) >= 1


def test_qa_io(tmp_path, paras):
    p = tmp_path / "qa.jsonl"
    save_qa(QA, p)
    assert load_qa(p) == QA
    pj = tmp_path / "qa.json"
    pj.write_text(json.dumps(QA, ensure_ascii=False), encoding="utf-8")
    assert load_qa(pj) == QA                     # json array 也吃
    with pytest.raises(ValueError):
        validate_qa([{"question": "只有問題"}])


def test_locate_gold(paras):
    assert _locate_gold(paras, QA[0]) == [paras[0]]   # 答案子字串定位
    assert _locate_gold(paras, QA[1]) == [paras[1]]   # gold 子字串映射回段落
    assert _locate_gold(paras, {"question": "?", "answer": "不存在zzz"}) == []


def test_build_examples(paras):
    warns = []
    exs = build_examples(paras, QA + [{"question": "?", "answer": "不存在zzz"}],
                         seed=42, log_fn=warns.append)
    assert len(exs) == 2                         # 定位不到的被剔除
    assert any("剔除" in w for w in warns)
    assert exs[0].paragraphs == paras            # context = 整份語料
    assert exs[0].level == "custom"
    with pytest.raises(ValueError):
        build_examples(paras, [{"question": "?", "answer": "zzz不存在"}])


def test_fingerprint(paras):
    fp = dataset_fingerprint(paras, QA)
    assert fp != dataset_fingerprint(paras, QA[:1])       # 改 QA 指紋變
    assert fp != dataset_fingerprint(paras[:1], QA)       # 改語料指紋變


def test_generate_qa_guard(paras, monkeypatch):
    def fake_chat_json(prompt, temperature, max_tokens, model=None, task=None):
        if "巴黎鐵塔" in prompt:
            return {"question": "巴黎鐵塔位於哪裡?", "answer": "法國巴黎"}
        return {"question": "聖母院始建於哪一年?", "answer": "1163 年"}

    monkeypatch.setattr(ds, "chat_json", fake_chat_json)
    qa = generate_qa(paras, n=2, seed=1)
    assert len(qa) == 2 and all(q["gold"] and q["hop"] == 1 for q in qa)

    # 答案不是段落內 span -> 全數被守門丟棄 -> RuntimeError
    monkeypatch.setattr(ds, "chat_json",
                        lambda *a, **k: {"question": "q", "answer": "zzz nope"})
    with pytest.raises(RuntimeError):
        generate_qa(paras, n=1, seed=1)


def test_multihop_pairs_and_generation(monkeypatch):
    import random
    # A/B 共享稀有詞 "quantum"，C 無關 -> 只配 (A,B)
    paras = [
        "Quantum computing uses qubits to represent superposition states "
        "in modern physics laboratories worldwide.",
        "The quantum research center in Geneva was founded in 1998 by a "
        "consortium of European universities.",
        "Bananas are rich in potassium and grow in tropical climates "
        "across South America and Asia.",
    ]
    pairs = _related_pairs(paras, random.Random(0))
    assert (0, 1) in pairs and (0, 2) not in pairs

    def fake_chat_json(prompt, temperature, max_tokens, model=None, task=None):
        if "PASSAGE 2" in prompt:   # 多跳 prompt
            return {"question": "When was the center researching qubits founded?",
                    "answer": "1998"}
        return {"question": "What grows in tropical climates?", "answer": "Bananas"}

    monkeypatch.setattr(ds, "chat_json", fake_chat_json)
    qa = generate_qa(paras, n=2, seed=0, multihop=True)
    hops = sorted(q["hop"] for q in qa)
    assert hops == [1, 2]
    mh = next(q for q in qa if q["hop"] == 2)
    assert len(mh["gold"]) == 2                  # gold = 兩個來源段落
