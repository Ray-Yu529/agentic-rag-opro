// 設定表單: 取代「改 Python 變數」的負擔
import { useState } from "react";

const STRATEGIES = [
  { id: "opro", label: "OPRO（LLM 推理提案）" },
  { id: "hybrid", label: "Hybrid（LLM 縮範圍 + Optuna）" },
  { id: "bandit", label: "Bandit（UCB，零 LLM）" },
  { id: "random", label: "Random（baseline）" },
];

export default function ConfigForm({ params, setParams, onRun, running }) {
  // QA 來源只是 UI 狀態: "generate" -> params.qa = null; "upload" -> params.qa = [...]
  const [qaSource, setQaSource] = useState("generate");
  const [qaError, setQaError] = useState("");

  const set = (k) => (e) => {
    const v = e.target.type === "number" ? Number(e.target.value) : e.target.value;
    setParams({ ...params, [k]: v });
  };

  const isCustom = params.dataset_mode === "custom";

  // .txt/.md 在瀏覽器端讀成文字；.pdf 轉 base64 由後端抽取 (文字層優先，掃描頁走 VLM)
  const onCorpusFiles = async (e) => {
    const files = [...e.target.files];
    if (files.length === 0) return;
    const texts = [];
    const pdfs = [];
    for (const f of files) {
      if (f.name.toLowerCase().endsWith(".pdf")) {
        const buf = new Uint8Array(await f.arrayBuffer());
        let bin = "";
        const CHUNK = 0x8000;
        for (let i = 0; i < buf.length; i += CHUNK) {
          bin += String.fromCharCode(...buf.subarray(i, i + CHUNK));
        }
        pdfs.push({ name: f.name, b64: btoa(bin) });
      } else {
        texts.push(await f.text());
      }
    }
    setParams({
      ...params,
      corpus_text: texts.join("\n\n"),
      corpus_pdfs: pdfs.length ? pdfs : null,
    });
  };

  // QA 檔: jsonl (每行一筆) 或 json array
  const onQaFile = async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    try {
      const text = await f.text();
      let qa;
      try {
        qa = JSON.parse(text);
        if (!Array.isArray(qa)) qa = [qa];
      } catch {
        qa = text.split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l));
      }
      const bad = qa.findIndex((r) => !r || !r.question || !r.answer);
      if (bad >= 0) throw new Error(`第 ${bad + 1} 筆缺 question/answer`);
      setQaError("");
      setParams({ ...params, qa });
    } catch (err) {
      setQaError(`QA 檔解析失敗: ${err.message}`);
      setParams({ ...params, qa: null });
    }
  };

  const pickQaSource = (src) => {
    setQaSource(src);
    if (src === "generate") setParams({ ...params, qa: null });
  };

  const hasCorpus = !!params.corpus_text.trim() ||
                    (params.corpus_pdfs && params.corpus_pdfs.length > 0);
  const customNotReady =
    isCustom && (!hasCorpus || (qaSource === "upload" && !params.qa));

  return (
    <div className="card">
      <h2>1. 設定</h2>

      <label>資料集</label>
      <div className="strategy-group">
        <button className={"chip" + (!isCustom ? " chip-active" : "")}
                onClick={() => setParams({ ...params, dataset_mode: "hotpot" })}
                disabled={running}>
          內建 HotpotQA
        </button>
        <button className={"chip" + (isCustom ? " chip-active" : "")}
                onClick={() => setParams({ ...params, dataset_mode: "custom" })}
                disabled={running}>
          自己的文件
        </button>
      </div>

      {isCustom && (
        <>
          <label>文件（.txt / .md / .pdf，可多選）</label>
          <input type="file" multiple accept=".txt,.md,.pdf,text/plain,text/markdown"
                 onChange={onCorpusFiles} disabled={running} />
          <label>或直接貼上內容（空行分段）</label>
          <textarea rows={5} value={params.corpus_text}
                    onChange={set("corpus_text")} disabled={running}
                    placeholder="貼上你的文件內容…" />
          {hasCorpus && (
            <p className="hint">
              語料 ≈ {params.corpus_text.length} 字元
              {params.corpus_pdfs && ` + ${params.corpus_pdfs.length} 個 PDF`}
            </p>
          )}

          <label>QA 來源</label>
          <div className="strategy-group">
            <button className={"chip" + (qaSource === "generate" ? " chip-active" : "")}
                    onClick={() => pickQaSource("generate")} disabled={running}>
              LLM 自動生成 {params.n} 題
            </button>
            <button className={"chip" + (qaSource === "upload" ? " chip-active" : "")}
                    onClick={() => pickQaSource("upload")} disabled={running}>
              上傳自備 QA
            </button>
          </div>
          {qaSource === "upload" && (
            <>
              <input type="file" accept=".jsonl,.json" onChange={onQaFile}
                     disabled={running} />
              <p className="hint">
                jsonl 每行一筆：{'{"question":"...","answer":"...","gold":["選填"]}'}
                {params.qa && ` — 已載入 ${params.qa.length} 題`}
              </p>
              {qaError && <p className="error">{qaError}</p>}
            </>
          )}
          {qaSource === "generate" && (
            <label className="check-row">
              <input type="checkbox" checked={params.multihop} disabled={running}
                     onChange={(e) => setParams({ ...params, multihop: e.target.checked })} />
              包含跨段落多跳題（query_decompose 才有東西可學）
            </label>
          )}
        </>
      )}

      <label>策略</label>
      <div className="strategy-group">
        {STRATEGIES.map((s) => (
          <button
            key={s.id}
            className={"chip" + (params.strategy === s.id ? " chip-active" : "")}
            onClick={() => setParams({ ...params, strategy: s.id })}
            disabled={running}
          >
            {s.label}
          </button>
        ))}
      </div>

      <label>測試題數 n：{params.n}</label>
      <input type="range" min="5" max="50" value={params.n}
             onChange={set("n")} disabled={running} />

      {!isCustom && (
        <>
          <label>困難題數 n_hard：{params.n_hard}</label>
          <input type="range" min="0" max={params.n} value={params.n_hard}
                 onChange={set("n_hard")} disabled={running} />
        </>
      )}

      <label>評估次數 budget：{params.budget}</label>
      <input type="range" min="3" max="20" value={params.budget}
             onChange={set("budget")} disabled={running} />

      <label>幻覺懲罰權重 λ：{params.lam}</label>
      <input type="range" min="0" max="1" step="0.1" value={params.lam}
             onChange={set("lam")} disabled={running} />

      <label>成本懲罰權重 μ：{params.mu}</label>
      <input type="range" min="0" max="0.3" step="0.05" value={params.mu}
             onChange={set("mu")} disabled={running} />
      <p className="hint">
        objective = 正確率 − {params.lam}×幻覺率
        {params.mu > 0 && ` − ${params.mu}×每題成本(ktok)`}
      </p>

      <label className="check-row">
        <input type="checkbox" checked={params.warmstart} disabled={running}
               onChange={(e) => setParams({ ...params, warmstart: e.target.checked })} />
        warm-start（OPRO/Hybrid 用過去 run 的最佳配置暖身；對照實驗別開）
      </label>

      <button className="run-btn" onClick={onRun}
              disabled={running || customNotReady}>
        {running ? "執行中…" : "▶ 開始最佳化"}
      </button>
      {customNotReady && (
        <p className="hint">
          ⚠️ 請先{!params.corpus_text.trim() ? "提供文件內容" : "上傳 QA 檔"}
        </p>
      )}
      <p className="hint">
        ⚠️ 約 {params.budget} × {params.n} × 3 個 API 呼叫
        {isCustom && qaSource === "generate" && `（首次另加 ${params.n} 次 QA 生成）`}
        ；越大越久越貴。
      </p>
    </div>
  );
}
