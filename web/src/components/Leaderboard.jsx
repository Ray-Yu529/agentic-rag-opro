// 試過的配置排行榜: 取代翻 jsonl
const fmt = (v) => (v === true ? "✓" : v === false ? "·" : v);

export default function Leaderboard({ strategies }) {
  const rows = [];
  Object.entries(strategies || {}).forEach(([name, s]) => {
    s.trials.forEach((t) => rows.push({ strategy: name, ...t }));
  });
  if (rows.length === 0) return null;
  rows.sort((a, b) => b.objective - a.objective);

  return (
    <div className="card">
      <h2>試過的配置（依 objective 排序）</h2>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>策略</th><th>chunk</th><th>top_k</th><th>retriever</th>
              <th>rerank</th><th>分解</th><th>verify</th>
              <th>正確率</th><th>幻覺率</th><th>obj</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className={i === 0 ? "row-best" : ""}>
                <td><span className="badge-sm">{r.strategy}</span></td>
                <td>{r.config.chunk_size}</td>
                <td>{r.config.top_k}</td>
                <td>{r.config.retriever}</td>
                <td>{fmt(r.config.rerank)}</td>
                <td>{fmt(r.config.query_decompose)}</td>
                <td>{fmt(r.config.verify)}</td>
                <td>{(r.score.correctness * 100).toFixed(0)}%</td>
                <td>{((1 - r.score.faithfulness) * 100).toFixed(0)}%</td>
                <td><b>{r.objective.toFixed(3)}</b></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
