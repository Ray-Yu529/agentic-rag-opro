// 最佳配置卡: 一眼看到「該用哪組設定」，不用讀 JSON

const KNOBS = [
  ["chunk_size", "Chunk 大小"],
  ["chunk_overlap", "Chunk 重疊"],
  ["top_k", "檢索段數"],
  ["retriever", "檢索器"],
  ["hybrid_alpha", "混合權重 α"],
  ["rerank", "重排"],
  ["query_decompose", "查詢分解"],
  ["hyde", "HyDE 改寫"],
  ["iterative", "迭代檢索"],
  ["compress", "壓縮上下文"],
  ["parent_child", "父段落餵入"],
  ["verify", "NLI 守門員"],
];

const fmt = (v) => (v === true ? "開" : v === false ? "關" : v);

export default function BestConfigCard({ best }) {
  if (!best) return (
    <div className="card"><h2>3. 最佳配置</h2>
      <p className="hint">執行後這裡會顯示勝出的設定。</p></div>
  );
  const s = best.score;
  return (
    <div className="card">
      <h2>3. 最佳配置 <span className="badge">{best.strategy}</span></h2>
      <div className="metric-row">
        <Metric label="objective" value={best.objective.toFixed(3)} big />
        <Metric label="正確率" value={(s.correctness * 100).toFixed(0) + "%"} />
        <Metric label="幻覺率" value={((1 - s.faithfulness) * 100).toFixed(0) + "%"} warn />
        {s.by_hop && Object.entries(s.by_hop).map(([h, v]) => (
          <Metric key={h} label={h === "2" ? "多跳正確率" : "單跳正確率"}
                  value={(v * 100).toFixed(0) + "%"} />
        ))}
        <Metric label="成本" value={s.avg_latency.toFixed(1) + "s"} />
      </div>
      <div className="knob-grid">
        {KNOBS.map(([k, label]) => (
          <div className="knob" key={k}>
            <span className="knob-label">{label}</span>
            <span className="knob-value">{fmt(best.config[k])}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function Metric({ label, value, big, warn }) {
  return (
    <div className={"metric" + (warn ? " metric-warn" : "")}>
      <div className={"metric-value" + (big ? " metric-big" : "")}>{value}</div>
      <div className="metric-label">{label}</div>
    </div>
  );
}
