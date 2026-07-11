import { useEffect, useRef, useState } from "react";
import { startRun, getStatus, getResults } from "./api.js";
import ConfigForm from "./components/ConfigForm.jsx";
import StatusPanel from "./components/StatusPanel.jsx";
import BestConfigCard from "./components/BestConfigCard.jsx";
import ObjectiveChart from "./components/ObjectiveChart.jsx";
import ParetoChart from "./components/ParetoChart.jsx";
import Leaderboard from "./components/Leaderboard.jsx";

export default function App() {
  const [params, setParams] = useState({
    strategy: "opro", n: 20, n_hard: 10, budget: 8, lam: 0.5,
    dataset_mode: "hotpot",   // hotpot | custom
    corpus_text: "",          // custom: 文件全文 (多檔已串接)
    qa: null,                 // custom: 自備 QA; null = 用 LLM 生成 n 題
  });
  const [status, setStatus] = useState(null);
  const [results, setResults] = useState(null);
  const [running, setRunning] = useState(false);
  const timer = useRef(null);

  const refreshResults = () => getResults().then(setResults).catch(() => {});

  // 進入頁面先載一次既有結果 (若之前跑過)
  useEffect(() => { refreshResults(); }, []);

  // 執行中: 每 1.5s 輪詢狀態 + 結果
  useEffect(() => {
    if (!running) return;
    timer.current = setInterval(async () => {
      try {
        const st = await getStatus();
        setStatus(st);
        refreshResults();
        if (!st.running) {
          setRunning(false);
          clearInterval(timer.current);
        }
      } catch {
        // 後端暫時連不上 (重啟/網路抖動): 跳過這一輪，下一輪再試
      }
    }, 1500);
    return () => clearInterval(timer.current);
  }, [running]);

  const onRun = async () => {
    const res = await startRun(params);
    if (res.ok) {
      setRunning(true);
      setStatus({ running: true, done: 0, total: params.budget, log: [] });
    } else {
      alert(res.msg || "啟動失敗");
    }
  };

  return (
    <div className="app">
      <header>
        <h1>Agentic RAG + OPRO 調參台</h1>
        <p>填表單 → 按執行 → 看圖表，不用碰 CLI 或 JSON。</p>
      </header>

      <div className="layout">
        <aside>
          <ConfigForm params={params} setParams={setParams}
                      onRun={onRun} running={running} />
          <StatusPanel status={status} />
        </aside>

        <main>
          <BestConfigCard best={results?.best} />
          <ObjectiveChart strategies={results?.strategies} />
          <ParetoChart pareto={results?.pareto} />
          <Leaderboard strategies={results?.strategies} />
        </main>
      </div>
    </div>
  );
}
