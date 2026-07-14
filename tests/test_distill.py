"""蒸餾功能的離線測試: 蒐集鉤子、資料處理、GPU guard、BASE_URL 覆寫。
train() 真正的 GPU 訓練不在這裡跑 (見 distill.py 頂部說明)。"""

import sys

import pytest

import distill as ds
import rag


# --- rag.py: BASE_URL 覆寫 -----------------------------------------------------
def test_base_url_override(monkeypatch):
    monkeypatch.setattr(rag, "NVIDIA_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("NVIDIA_API_KEY", "ollama")
    rag.get_client.cache_clear()
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setattr(rag, "OpenAI", FakeOpenAI)
    rag.get_client()
    assert captured["base_url"] == "http://localhost:11434/v1"
    assert captured["timeout"] == 30.0
    rag.get_client.cache_clear()


# --- rag.py: distill 蒐集鉤子 ---------------------------------------------------
class _FakeMsg:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMsg(content)


class _FakeResp:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]
        self.usage = None


def test_distill_log_records_chat_completions(tmp_path, monkeypatch):
    log = tmp_path / "distill.jsonl"
    monkeypatch.setenv("DISTILL_LOG_PATH", str(log))

    def fake_fn(**kw):
        return _FakeResp("Paris")

    rag.api_call(fake_fn, model="meta/llama-3.1-8b-instruct",
                messages=[{"role": "user", "content": "Where is the tower?"}],
                distill_task="generate")

    rows = [__import__("json").loads(ln) for ln in log.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["task"] == "generate"
    assert rows[0]["completion"] == "Paris"
    assert rows[0]["messages"][0]["content"] == "Where is the tower?"


def test_distill_log_skips_embeddings_and_unset_env(tmp_path, monkeypatch):
    log = tmp_path / "distill.jsonl"
    monkeypatch.setenv("DISTILL_LOG_PATH", str(log))

    class FakeEmbedResp:
        pass

    # 沒有 messages kwarg (embedding 呼叫的形狀) -> 不記
    rag.api_call(lambda **kw: FakeEmbedResp(), model="embed-x", input=["hi"])
    assert not log.exists()

    # DISTILL_LOG_PATH 沒設 -> 不記
    monkeypatch.delenv("DISTILL_LOG_PATH")
    rag.api_call(lambda **kw: _FakeResp("ok"), model="m",
                messages=[{"role": "user", "content": "q"}], distill_task="generate")
    assert not log.exists()


def test_distill_log_failure_does_not_break_call(tmp_path, monkeypatch):
    # DISTILL_LOG_PATH 指到一個不能建立的路徑 (以檔案當父目錄) -> 記錄失敗仍要吞掉
    bad_parent = tmp_path / "not_a_dir"
    bad_parent.write_text("x", encoding="utf-8")
    monkeypatch.setenv("DISTILL_LOG_PATH", str(bad_parent / "log.jsonl"))
    resp = rag.api_call(lambda **kw: _FakeResp("ok"), model="m",
                        messages=[{"role": "user", "content": "q"}],
                        distill_task="generate")
    assert resp.choices[0].message.content == "ok"   # 呼叫本身正常完成


# --- distill.py: 讀取/去重/統計 -------------------------------------------------
def _write_log(path, rows):
    import json
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")


def test_load_examples_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        ds.load_examples(tmp_path / "nope.jsonl")


def test_dedupe_and_stats(tmp_path):
    rows = [
        {"task": "generate", "model": "m1",
         "messages": [{"role": "user", "content": "q1"}], "completion": "a1"},
        {"task": "generate", "model": "m1",
         "messages": [{"role": "user", "content": "q1"}], "completion": "a1"},  # 重複
        {"task": "generate", "model": "m1",
         "messages": [{"role": "user", "content": "q2"}], "completion": "a2 longer"},
        {"task": "judge_correctness", "model": "m2",
         "messages": [{"role": "user", "content": "q3"}], "completion": "YES"},
    ]
    p = tmp_path / "log.jsonl"
    _write_log(p, rows)
    loaded = ds.load_examples(p)
    assert len(loaded) == 4
    deduped = ds.dedupe(loaded)
    assert len(deduped) == 3

    s = ds.stats(deduped)
    assert s["total"] == 3
    assert s["by_task"] == {"generate": 2, "judge_correctness": 1}
    assert s["by_model"] == {"m1": 2, "m2": 1}


# --- distill.py: export (篩選 + SFT 格式) ---------------------------------------
def test_export_filters_task_and_converts_sft(tmp_path):
    rows = [
        {"task": "generate", "model": "m1",
         "messages": [{"role": "user", "content": "q1"}], "completion": "a1"},
        {"task": "judge_correctness", "model": "m2",
         "messages": [{"role": "user", "content": "q2"}], "completion": "YES"},
    ]
    log = tmp_path / "log.jsonl"
    _write_log(log, rows)
    out = tmp_path / "sft.jsonl"

    n = ds.export(log, out, task="generate")
    assert n == 1
    import json
    written = json.loads(out.read_text(encoding="utf-8").strip())
    assert written["messages"] == [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]


def test_export_empty_after_filter_raises(tmp_path):
    log = tmp_path / "log.jsonl"
    _write_log(log, [{"task": "generate", "model": "m",
                      "messages": [{"role": "user", "content": "q"}],
                      "completion": "a"}])
    with pytest.raises(ValueError):
        ds.export(log, tmp_path / "out.jsonl", task="nonexistent_task")


# --- distill.py: modelfile (純字串模板) -----------------------------------------
def test_modelfile_template():
    text = ds.modelfile("my-model.gguf", model_name="test-model")
    assert "FROM my-model.gguf" in text
    assert "PARAMETER temperature 0.0" in text
    text2 = ds.modelfile("m.gguf", system_prompt="Be concise.")
    assert 'SYSTEM """Be concise."""' in text2


# --- distill.py: train() 的 GPU guard --------------------------------------------
def test_train_requires_cuda_when_torch_present():
    # torch 不是 requirements.txt 的依賴 (CI/一般環境不會裝)；只有在剛好裝了
    # torch 的機器上才驗證「有 torch 但無 CUDA」這個分支，其餘環境乾淨跳過。
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():
        pytest.skip("這台機器有 CUDA，guard 分支不適用")
    with pytest.raises(SystemExit, match="CUDA|GPU"):
        ds.train("nope.jsonl", "some/base-model", "out/")


def test_train_requires_distill_deps_when_torch_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)   # 模擬沒裝 requirements-distill.txt
    with pytest.raises(SystemExit, match="requirements-distill"):
        ds.train("nope.jsonl", "some/base-model", "out/")
