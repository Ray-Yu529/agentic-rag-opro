"""
distill.py — 把這個專案「餵過的 LLM 呼叫」蒸餾成一個可離線跑的小模型

動機: 整個專案所有 LLM 呼叫 (生成器、判官、query 分解、HyDE、gap query、
OPRO 推理、QA 生成) 都經過 rag.py 的 api_call() 這一個出口。設
DISTILL_LOG_PATH 環境變數後，每通成功的 chat completion 會被記成一筆
(messages, completion) 訓練樣本，標上來源任務 (見 rag.py 的 distill_task)。

流程:
  1. 蒐集: 設 DISTILL_LOG_PATH，照平常跑 eval.py / sweep.py / run.py / server.py
     (呼叫大模型的同時，順便留下教材，不用另外寫資料生成腳本)。
  2. python distill.py stats  --log data/distill.jsonl        # 看蒐集到什麼
  3. python distill.py export --log data/distill.jsonl --task generate --out data/sft_generate.jsonl
  4. python distill.py train  --data data/sft_generate.jsonl --base-model Qwen/Qwen2.5-1.5B-Instruct --out distilled/generate
     (這步需要 GPU；requirements-distill.txt 的重依賴只在這裡才 import)
  5. 用 llama.cpp 把 base model + LoRA adapter 匯出成 GGUF (見 README)，
     python distill.py modelfile --gguf my-model.gguf --out Modelfile
     ollama create my-distilled -f Modelfile
  6. .env: BASE_URL=http://localhost:11434/v1, GEN_MODEL=my-distilled,
     NVIDIA_API_KEY=ollama (Ollama 不驗證但 openai client 要求非空字串)
     整條既有管線 (rag.py/judge.py/optimizer.py/dataset.py) 不用改就指向本地模型。

刻意把「不同任務混一個模型」與「單一任務專精模型」都留給使用者選：
export 可用 --task 篩選只留一種行為的樣本，訓練出來的小模型才不會
同時被生成、判官、JSON 結構化推理這些性質不同的任務拉扯。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


# --- 1. 讀取 / 統計 / 清洗 ----------------------------------------------------
def load_examples(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"找不到 {p}。先設 DISTILL_LOG_PATH={p} 環境變數，"
            "再照平常跑 eval.py/run.py 等指令蒐集資料。")
    rows = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        if ln.strip():
            rows.append(json.loads(ln))
    return rows


def dedupe(examples: list[dict]) -> list[dict]:
    """丟掉完全重複的樣本 (同 task + messages + completion)。
    評估同一組配置常會產生大量重複 prompt (例如同一份文件重跑)，
    重複樣本對訓練沒有額外資訊，只會讓那些 prompt 被過度加權。"""
    seen, out = set(), []
    for ex in examples:
        key = (ex.get("task"), json.dumps(ex.get("messages"), sort_keys=True),
               ex.get("completion"))
        if key not in seen:
            seen.add(key)
            out.append(ex)
    return out


def stats(examples: list[dict]) -> dict:
    tasks = Counter(ex.get("task", "unknown") for ex in examples)
    models = Counter(ex.get("model", "unknown") for ex in examples)
    avg_len = {}
    for task in tasks:
        lens = [len(ex.get("completion") or "") for ex in examples
                if ex.get("task") == task]
        avg_len[task] = round(sum(lens) / len(lens), 1) if lens else 0.0
    return {"total": len(examples), "by_task": dict(tasks),
            "by_model": dict(models), "avg_completion_chars": avg_len}


def print_stats(examples: list[dict]) -> None:
    s = stats(examples)
    print(f"總樣本數: {s['total']}")
    print("依任務分布 (task -> 筆數, 平均回答長度):")
    for task, n in sorted(s["by_task"].items(), key=lambda kv: -kv[1]):
        print(f"  {task:20s} {n:5d}   avg_chars={s['avg_completion_chars'][task]}")
    print("依模型分布:")
    for model, n in sorted(s["by_model"].items(), key=lambda kv: -kv[1]):
        print(f"  {model:40s} {n}")
    if s["total"] < 200:
        print(f"\n[提醒] 只有 {s['total']} 筆，LoRA SFT 建議至少數百到數千筆同任務樣本。"
              "繼續照平常跑 eval.py/run.py/sweep.py 蒐集更多，或先用小規模試跑管線通不通。")


# --- 2. 匯出成 SFT 格式 -------------------------------------------------------
def to_sft_example(ex: dict) -> dict:
    """(messages, completion) -> 標準 chat-SFT 格式 (補上 assistant 回合)，
    多數 chat 微調框架 (trl SFTTrainer 等) 吃這個格式。"""
    return {"messages": ex["messages"] + [{"role": "assistant", "content": ex["completion"]}]}


def export(log_path: str | Path, out_path: str | Path,
          task: str | None = None, dedup: bool = True) -> int:
    """篩選 (可選單一 task) + 去重 + 轉成 SFT 格式，寫成新 jsonl。回傳筆數。"""
    examples = load_examples(log_path)
    if task:
        examples = [e for e in examples if e.get("task") == task]
    if dedup:
        examples = dedupe(examples)
    if not examples:
        raise ValueError(
            f"篩選後沒有樣本 (task={task!r})。用 `distill.py stats` 看看實際蒐集到哪些 task。")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(to_sft_example(e), ensure_ascii=False) for e in examples) + "\n",
        encoding="utf-8")
    return len(examples)


# --- 3. 訓練 (需要 GPU；重依賴延後到這裡才 import) ----------------------------
def _require_cuda() -> None:
    try:
        import torch
    except ImportError as e:
        raise SystemExit(
            "訓練需要 requirements-distill.txt 的依賴: "
            "pip install -r requirements-distill.txt") from e
    if not torch.cuda.is_available():
        raise SystemExit(
            "沒有偵測到可用的 CUDA GPU。訓練 (LoRA SFT) 需要 GPU，"
            "這一步無法在純 CPU 環境跑；資料蒐集/匯出 (stats/export) 不需要 GPU，"
            "可以先在這台機器做完，把 SFT jsonl 帶去有 GPU 的環境再跑 train。")


def train(data_path: str | Path, base_model: str, out_dir: str | Path,
          epochs: int = 3, lr: float = 2e-4, lora_r: int = 16,
          max_seq_len: int = 1024, batch_size: int = 4) -> None:
    """LoRA SFT：在 base_model 上微調，只更新 LoRA adapter (不動 base 權重)。
    輸出 out_dir 是一個 adapter checkpoint，之後用 peft 合併回 base model
    再匯出 GGUF (見 README「本地/離線模型」章節的完整步驟)。"""
    _require_cuda()
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"找不到 {data_path}，先用 `distill.py export` 產生。")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto")

    lora_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_r * 2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files=str(data_path), split="train")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sft_cfg = SFTConfig(
        output_dir=str(out_dir), num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, learning_rate=lr,
        max_seq_length=max_seq_len, logging_steps=10,
        save_strategy="epoch", bf16=True, report_to=[])
    trainer = SFTTrainer(model=model, args=sft_cfg, train_dataset=ds,
                         processing_class=tokenizer)
    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"\n完成，LoRA adapter 存到 {out_dir}")
    print("下一步 (llama.cpp 環境): 合併 adapter -> 轉 GGUF -> "
          "python distill.py modelfile --gguf <path>.gguf --out Modelfile")


# --- 4. 部署輔助 (純字串模板，不需要 GPU/重依賴) --------------------------------
def modelfile(gguf_path: str, model_name: str = "agentic-rag-distilled",
             system_prompt: str | None = None) -> str:
    """產生 Ollama Modelfile 內容。之後 `ollama create <model_name> -f Modelfile`
    即可在本機架起 OpenAI 相容端點 (預設 http://localhost:11434/v1)。"""
    lines = [f"FROM {gguf_path}"]
    if system_prompt:
        lines.append(f'SYSTEM """{system_prompt}"""')
    lines.append('PARAMETER temperature 0.0')
    return "\n".join(lines) + "\n"


# --- CLI ----------------------------------------------------------------------
def _cmd_stats(args):
    print_stats(load_examples(args.log))


def _cmd_export(args):
    n = export(args.log, args.out, task=args.task, dedup=not args.no_dedup)
    print(f"已匯出 {n} 筆 -> {args.out}")


def _cmd_train(args):
    train(args.data, args.base_model, args.out, epochs=args.epochs,
         lr=args.lr, lora_r=args.lora_r, batch_size=args.batch_size)


def _cmd_modelfile(args):
    text = modelfile(args.gguf, model_name=args.name)
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"已寫入 {args.out}\n\n{text}")
    print(f"下一步: ollama create {args.name} -f {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="把專案的 LLM 呼叫蒐集/蒸餾成可離線跑的小模型")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("stats", help="看蒐集到的訓練樣本分布")
    p1.add_argument("--log", default="data/distill.jsonl")
    p1.set_defaults(func=_cmd_stats)

    p2 = sub.add_parser("export", help="篩選/去重/轉成 SFT 格式")
    p2.add_argument("--log", default="data/distill.jsonl")
    p2.add_argument("--out", required=True)
    p2.add_argument("--task", default=None,
                    help="只留單一任務 (generate/judge_correctness/judge_faithfulness/"
                         "decompose/hyde/gap_query/compress/opro_meta/hybrid_region/"
                         "qa_gen/qa_gen_multihop)；不給則全部混合匯出")
    p2.add_argument("--no-dedup", action="store_true")
    p2.set_defaults(func=_cmd_export)

    p3 = sub.add_parser("train", help="LoRA SFT (需要 GPU)")
    p3.add_argument("--data", required=True)
    p3.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p3.add_argument("--out", required=True)
    p3.add_argument("--epochs", type=int, default=3)
    p3.add_argument("--lr", type=float, default=2e-4)
    p3.add_argument("--lora-r", type=int, default=16)
    p3.add_argument("--batch-size", type=int, default=4)
    p3.set_defaults(func=_cmd_train)

    p4 = sub.add_parser("modelfile", help="產生 Ollama Modelfile")
    p4.add_argument("--gguf", required=True)
    p4.add_argument("--name", default="agentic-rag-distilled")
    p4.add_argument("--out", default="Modelfile")
    p4.set_defaults(func=_cmd_modelfile)

    args = ap.parse_args()
    args.func(args)
