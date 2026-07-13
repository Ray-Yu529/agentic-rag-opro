// 配置 A/B 對比: 同題集逐題比較，「為什麼這組贏」從數字變成看得見的證據
// 資料全部來自 cache record (per_question_correct)，零額外 API
import { useEffect, useState } from "react";
import { getRecords } from "../api.js";

const FLAG_NAMES = {
  rerank: "rerank", hyde: "hyde", iterative: "iter", query_decompose: "decomp",
  compress: "cmp", parent_child: "pc", verify: "verify",
};

const label = (r) => {
  const c = r.config;
  const flags = Object.keys(FLAG_NAMES).filter((k) => c[k])
    .map((k) => FLAG_NAMES[k]);
  return `${c.retriever} c${c.chunk_size} k${c.top_k}` +
    (flags.length ? ` +${flags.join("+")}` : "") +
    ` ｜ 正確率 ${Math.round(r.score.correctness * 100)}%`;
};

function SideStats({ name, rec }) {
  const s = rec.score;
  return (
    <div className="metric">
      <div className="metric-label">{name}</div>
      <div className="metric-value">{(s.correctness * 100).toFixed(0)}%</div>
      <div className="metric-label">
        幻覺 {((1 - s.faithfulness) * 100).toFixed(0)}% ·{" "}
        {Math.round(s.avg_tokens)} tok/題
      </div>
    </div>
  );
}

export default function ComparePanel({ bump }) {
  const [records, setRecords] = useState([]);
  const [ia, setIa] = useState(-1);
  const [ib, setIb] = useState(-1);

  // run 結束 (bump: true -> false) 時重新抓
  useEffect(() => {
    if (!bump) getRecords().then((r) => setRecords(r.records || [])).catch(() => {});
  }, [bump]);

  if (records.length < 2) return null;
  const A = records[ia];
  const B = records[ib];

  let diff = null;
  if (A && B && ia !== ib) {
    const qa = A.per_question_correct || {};
    const qb = B.per_question_correct || {};
    const common = Object.keys(qa).filter((q) => q in qb);
    diff = {
      aWin: common.filter((q) => qa[q] > qb[q]),
      bWin: common.filter((q) => qb[q] > qa[q]),
      n: common.length,
    };
  }

  return (
    <div className="card">
      <h2>配置 A/B 對比（同題集逐題）</h2>
      <div className="ab-selects">
        <select value={ia} onChange={(e) => setIa(+e.target.value)}>
          <option value={-1}>選擇配置 A…</option>
          {records.map((r, i) => (
            <option key={i} value={i}>{label(r)}</option>
          ))}
        </select>
        <select value={ib} onChange={(e) => setIb(+e.target.value)}>
          <option value={-1}>選擇配置 B…</option>
          {records.map((r, i) => (
            <option key={i} value={i}>{label(r)}</option>
          ))}
        </select>
      </div>

      {diff && (
        <>
          <div className="metric-row" style={{ marginTop: 12 }}>
            <SideStats name="A 正確率" rec={A} />
            <div className="metric">
              <div className="metric-value">
                {diff.aWin.length} : {diff.bWin.length}
              </div>
              <div className="metric-label">
                A 贏 : B 贏（平手 {diff.n - diff.aWin.length - diff.bWin.length}）
              </div>
            </div>
            <SideStats name="B 正確率" rec={B} />
          </div>
          <div className="ab-cols">
            <div>
              <label>只有 A 答對的題（{diff.aWin.length}）</label>
              {diff.aWin.map((q, i) => <div className="ab-q" key={i}>{q}</div>)}
            </div>
            <div>
              <label>只有 B 答對的題（{diff.bWin.length}）</label>
              {diff.bWin.map((q, i) => <div className="ab-q" key={i}>{q}</div>)}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
