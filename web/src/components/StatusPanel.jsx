// 即時進度條 + OPRO 推理 log: 讓使用者「看得到它在想什麼」

export default function StatusPanel({ status }) {
  if (!status) return null;
  const { running, done, total, log, error } = status;
  const pct = total ? Math.round((done / total) * 100) : 0;
  const state = running ? "run" : done ? "done" : "idle";
  const stateText = running ? "執行中…" : done ? "已完成" : "尚未開始";

  return (
    <div className="card">
      <h2>2. 進度</h2>
      <div className="progress-row">
        <div className="progress-bar">
          <div className="progress-fill" style={{ width: `${pct}%` }} />
        </div>
        <span>{done}/{total}</span>
      </div>
      <p className="status-line">
        <span className={`status-dot ${state}`} />
        {stateText}
      </p>
      {error && <p className="error">❌ {error}</p>}

      <div className="log-box">
        {(log || []).length === 0 && <p className="log-line">（推理紀錄會即時出現在這裡）</p>}
        {(log || []).map((line, i) => (
          <div key={i} className={line.includes("推理") ? "log-reason" : "log-line"}>
            {line}
          </div>
        ))}
      </div>
    </div>
  );
}
