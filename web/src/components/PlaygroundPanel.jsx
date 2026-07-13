// Playground: 用勝出配置對你的語料即時問答 —— 從「看指標」到「用它」
import { useState } from "react";
import { ask } from "../api.js";

export default function PlaygroundPanel({ best, corpusText, running }) {
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const [resp, setResp] = useState(null);

  const hasCorpus = !!corpusText.trim();
  const canAsk = q.trim() && hasCorpus && !busy && !running;

  const onAsk = async () => {
    setBusy(true);
    setResp(null);
    try {
      const r = await ask({
        question: q,
        config: best?.config || null,
        corpus_text: corpusText,
      });
      setResp(r);
    } catch (e) {
      setResp({ ok: false, msg: String(e) });
    }
    setBusy(false);
  };

  return (
    <div className="card">
      <h2>Playground（用勝出配置直接問答）</h2>
      {!hasCorpus ? (
        <p className="hint">
          左側資料集選「自己的文件」並提供內容後，這裡就能用最佳配置對你的語料即時問答。
        </p>
      ) : (
        <>
          <p className="hint">
            使用配置：
            {best
              ? `${best.strategy} 的最佳配置（obj=${best.objective.toFixed(3)}）`
              : "預設配置（還沒跑最佳化）"}
          </p>
          <textarea rows={2} value={q} onChange={(e) => setQ(e.target.value)}
                    placeholder="輸入你的問題…" disabled={busy} />
          <button className="run-btn" onClick={onAsk} disabled={!canAsk}>
            {busy ? "檢索與生成中…" : "▶ 問答"}
          </button>

          {resp && !resp.ok && <p className="error">{resp.msg}</p>}
          {resp && resp.ok && (
            <div className="qa-result">
              <div className="qa-answer">
                {resp.abstained ? (
                  <span className="verdict verdict-abstain">— 棄答（證據不足）</span>
                ) : resp.faithful >= 1 ? (
                  <span className="verdict verdict-ok">✓ 被 context 支持</span>
                ) : (
                  <span className="verdict verdict-warn">⚠ 未被 context 支持</span>
                )}
                <p className="qa-text">{resp.answer}</p>
                <p className="hint">{resp.latency}s · {resp.tokens} tokens</p>
              </div>
              <label>檢索到的段落（{resp.chunks.length}，藍框 = 答案引用）</label>
              {resp.chunks.map((c, i) => (
                <div key={i}
                     className={"chunk-item" +
                       (resp.citations?.includes(i + 1) ? " chunk-cited" : "")}>
                  <span className="chunk-no">[{i + 1}]</span> {c}
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
