// 設定表單: 取代「改 Python 變數」的負擔

const STRATEGIES = [
  { id: "opro", label: "OPRO（LLM 推理提案）" },
  { id: "random", label: "Random（baseline）" },
  { id: "hybrid", label: "Hybrid（LLM 縮範圍 + Optuna）" },
];

export default function ConfigForm({ params, setParams, onRun, running }) {
  const set = (k) => (e) => {
    const v = e.target.type === "number" ? Number(e.target.value) : e.target.value;
    setParams({ ...params, [k]: v });
  };

  return (
    <div className="card">
      <h2>1. 設定</h2>

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

      <label>困難題數 n_hard：{params.n_hard}</label>
      <input type="range" min="0" max={params.n} value={params.n_hard}
             onChange={set("n_hard")} disabled={running} />

      <label>評估次數 budget：{params.budget}</label>
      <input type="range" min="3" max="20" value={params.budget}
             onChange={set("budget")} disabled={running} />

      <label>幻覺懲罰權重 λ：{params.lam}</label>
      <input type="range" min="0" max="1" step="0.1" value={params.lam}
             onChange={set("lam")} disabled={running} />
      <p className="hint">objective = 正確率 − {params.lam} × 幻覺率</p>

      <button className="run-btn" onClick={onRun} disabled={running}>
        {running ? "執行中…" : "▶ 開始最佳化"}
      </button>
      <p className="hint">
        ⚠️ 約 {params.budget} × {params.n} × 3 個 API 呼叫；越大越久越貴。
      </p>
    </div>
  );
}
