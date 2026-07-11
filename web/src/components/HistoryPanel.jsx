// 過去 run 的歷史: 每列附自包含 HTML 報告連結 (可稽核、可交付)
export default function HistoryPanel({ runs }) {
  if (!runs || runs.length === 0) return null;

  return (
    <div className="card">
      <h2>Run 歷史</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>時間</th><th>策略</th><th>資料集</th><th>n</th><th>budget</th>
              <th>best obj</th><th>正確率</th><th>幻覺率</th><th>報告</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((r) => (
              <tr key={r.run_id}>
                <td>{r.ts}</td>
                <td><span className="badge-sm">{r.strategy}</span></td>
                <td>{r.params.dataset_mode === "custom" ? "自訂文件" : "HotpotQA"}</td>
                <td>{r.params.n}</td>
                <td>{r.params.budget}</td>
                <td><b>{r.best.objective.toFixed(3)}</b></td>
                <td>{(r.best.score.correctness * 100).toFixed(0)}%</td>
                <td>{((1 - r.best.score.faithfulness) * 100).toFixed(0)}%</td>
                <td>
                  <a href={`/api/report/${r.run_id}`} target="_blank" rel="noreferrer">
                    開啟
                  </a>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
