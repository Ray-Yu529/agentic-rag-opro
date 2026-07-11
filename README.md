# Agentic RAG + OPRO optimizer (Demo)

用 OPRO-style 的 LLM meta-optimizer，觀察 RAG 失敗案例後縮小搜索範圍，
自動最佳化一個 Agentic RAG 的配置，並對比 random / 數值最佳化所需的評估預算。

全部模型走 **NVIDIA NIM API**（OpenAI 相容），本機不需要 GPU。

---

## 這東西解決什麼問題？(Why)

任何人要把資料丟進 RAG（檢索增強生成）系統，都會卡在同一個問題：
**chunk 要切多大？要檢索幾段？用關鍵字還是語意檢索？** 這些參數沒有通用最佳解，
傳統做法是工程師憑經驗手調，或用 Optuna 暴力窮舉——前者靠運氣，後者燒錢又看不懂「為什麼這組爛」。

本專案讓 **LLM 當調參員**：它讀過少數幾組實驗的失敗案例後，像資深工程師一樣推論
「這題答錯是因為檢索沒撈到關鍵段落」還是「撈到了但被雜訊淹沒」，
再據此提出下一組值得試的參數——用**更少的測試次數**逼近最佳配置，
而且每一步都留下**看得懂的推理紀錄**。

## 誰可以用？(Who)

**🏢 企業 / 團隊**
- **內部知識庫問答調優**：把客服 FAQ、產品手冊、法遵文件接進 RAG 後，不需要 ML 專家，
  讓系統自動找出最適合「你的文件特性」的配置，降低導入門檻。
- **省 API / 算力成本**：以往調參要跑上百次評估，這裡用 LLM 推理把評估次數壓到個位數，
  對「每次評估都要呼叫付費大模型」的場景特別有感。
- **可稽核、可交付**：每輪的 reasoning 軌跡可直接寫進技術報告或交付文件，
  向主管/客戶解釋「為什麼選這組參數」，而不是一句「調出來的」。
- **可換模型/換領域複用**：把 NVIDIA NIM 換成自家私有模型端點（OpenAI 相容介面通用），
  同一套最佳化流程即可套到不同部門的知識庫。

**👤 個人 / 研究者**
- **個人文件助手調優**：論文庫、筆記、電子書做問答時，自動挑出最佳 chunk/檢索設定。
- **學術專案 / 課程作業**：附完整 baseline（random / 數值）對照與 ablation 結構，
  reasoning 軌跡是寫論文「最佳化過程分析」的現成素材。
- **學 RAG 與 LLM-as-optimizer**：程式精簡、模組清楚，是理解 OPRO 概念的最小可跑範例。

> ⚠️ 定位誠實說：這是**研究/Demo 等級**，測試集 30 題、搜索空間 216 組。
> 要上正式生產，需放大測試集到上百題、擴大參數空間、加入更嚴謹的成本/延遲監控（見下方 Roadmap）。

## 核心能力

**Agentic RAG（三個可被優化的 agentic 行為）**
- **rerank**：檢索後用語意相似度對候選重排再取 top_k。
- **query_decompose**：多跳問題先拆成子問題各自檢索（HotpotQA 是多跳）。
- **verify（NLI 自我檢查守門員）**：生成後用 NLI 判斷答案是否被 context 支持；不過關 → 自動拓寬檢索重答，仍不忠實 → **棄答**而非硬掰。這是「降幻覺」的關鍵。
  棄答**不計入幻覺**（沒有捏造內容），另以獨立的 abstain rate 追蹤，避免 objective 雙重懲罰而教壞最佳化器。
  **不論 verify 開關**，生成器自行棄答（"I don't have enough evidence"）也按同一規則計分，
  否則 verify=False 的誠實棄答會被判官記成幻覺，objective 隱性補貼 verify、對比失真。

**多目標最佳化**
- objective = `正確率 − 0.5 × 幻覺率`，並追蹤成本（latency / tokens）。
- OPRO 看的是**失敗分群**（檢索失敗 vs 生成幻覺），據此推論該開哪個 agentic 開關。

**嚴謹評估**
- LLM-as-judge 算**語意正確性**（EM/F1 對自由文本太嚴）＋ **NLI 忠實度**（幻覺率）＋ 棄答率。
- 判官模型刻意與生成模型**不同家族**（Mixtral vs Llama），同家族大小模型仍可能自我偏袒。
- 失敗分群分「檢索 / 生成 / 棄答」三類；答對但 recall 不滿的題不算失敗。
- retrieval recall 要求 gold 段落被覆蓋 ≥50% 字元才算命中，避免「撈到 256 字皮毛就滿分」偏袒小 chunk。
- token 成本涵蓋生成、query 分解、embedding 與守門員判官；所有 API 呼叫遇 429/5xx 自動指數退避重試。
- 單題偶發失敗會跳過，但**失敗率 >10% 整輪作廢、不寫入快取**（避免 429 風暴的壞分數被永久快取）。
- sweep 的 go/no-go 用**成對符號檢定**（同題集逐題比較）判斷區分度，而非只看平均分差距（n=30 時噪音 ≈0.09）。

**三種最佳化策略對照**
- `random`（baseline）／`OPRO`（LLM 推理提案）／`hybrid`（LLM 縮範圍 → Optuna 收斂）。

## 架構

兩層：上層 meta-optimizer 提配置，下層 Agentic RAG 被評估，分數＋失敗案例回灌給上層推理。

```
┌──────────────────────────────────────────────────────────────┐
│  Meta-Optimizer 層   (optimizer.py / hybrid.py)                │
│                                                                │
│   經驗軌跡庫 ──► 反思: 瓶頸在檢索? 還是生成幻覺?  ──► 提下一組配置 │
│   (trajectory)      (讀失敗分群 + 失敗案例)        (去重/合法性檢查)│
└───────────────▲────────────────────────────────────┬──────────┘
       分數+失敗案例│                                  │ 配置 θ
                  │              ┌───────────────────▼──────────┐
                  │   cache.json │  Agentic RAG 層  (rag.py)     │
                  │  (查過不重跑)│                               │
                  │              │  Q ─►[query 分解]─►檢索─►[rerank]│
                  │              │     ─► 生成 ─►[NLI 守門員]─► 答案 │
                  └──────────────┤                               │
                                 └───────┬───────────────────────┘
                                         │ 答案
                          ┌──────────────▼───────────────┐
                          │  評估  (eval.py + judge.py)   │
                          │  EM/F1 · recall · 正確性(judge)│
                          │  · 忠實度(NLI) · 成本          │
                          └──────────────────────────────┘
```

- **[ ] 內是 agentic 開關**（rerank / query 分解 / NLI 守門員），由 meta-optimizer 決定要不要開。
- 評估結果（含「檢索失敗 vs 生成幻覺」的失敗分群）寫回軌跡庫，下一輪 LLM 據此推理。
- 分數一律先過 `cache.json`：三種策略試到同一組配置時零額外 API。

## 安裝 (PowerShell)

conda base 已啟用時直接用 `python` 即可 (提示字元有 `(base)`)。
**注意**: PowerShell 不吃 `/c/Users/...` 這種 Git Bash 路徑；要嘛用 `python`，
要嘛用 Windows 路徑 `C:\Users\r9105\miniconda3\python.exe`。

```powershell
cd C:\Users\r9105\VScode\agentic-rag-opro   # 先進專案資料夾!
python -m pip install -r requirements.txt
Copy-Item .env.example .env                 # 然後編輯 .env 填入 nvapi- 金鑰
```

## 檔案

| 檔案 | 角色 |
|------|------|
| `rag.py` | Agentic RAG：chunk + bm25/dense/hybrid + rerank + query 分解 + NLI 守門員；進程級 embedding 快取（同文字不重打 API）＋分批送 |
| `judge.py` | LLM-as-judge：NLI 忠實度 + 語意正確性（判官與生成模型不同家族） |
| `eval.py` | HotpotQA 分層子集 + EM/F1/recall/忠實度/正確率/成本 + 失敗分群（檢索/生成/棄答） |
| `cache.py` | 快取式評估（查過不重跑），鍵含測試集 fingerprint＋模型/prompt 指紋，改模型或 prompt 舊分數自動失效 |
| `sweep.py` | 核心網格 27 組掃描 → 填 cache + go/no-go（成對符號檢定） |
| `memory/trajectory.py` | 經驗軌跡庫（config+分數+失敗案例, jsonl） |
| `optimizer.py` | 搜索空間(216) + 多目標 + random baseline + OPRO 迴路（含邊際統計、每輪最多 2 提案） |
| `hybrid.py` | LLM 縮範圍（兩階段：中場重新縮一次）→ Optuna 在範圍內收斂 |
| `pareto.py` | Pareto 前緣（plot.py 與 server.py 共用） |
| `run.py` | 編排 random vs OPRO vs hybrid 對照（CLI） |
| `plot.py` | best-objective 對比曲線 + 正確率/幻覺率 Pareto 散點（CLI） |
| `server.py` | FastAPI 後端：把 optimizer 包成 API 給 Web UI 用 |
| `web/` | React 前端（見下方「圖形介面 Web UI」） |

## 跑法 (依序)

```powershell
python rag.py     # 0. 冒煙測試: 測 API 通不通 (含 verify 守門員, 不連 HotpotQA)
python eval.py    # 1. 對預設配置跑 30 題，吐全部指標 (含幻覺率/成本)
python sweep.py   # 2. 核心 27 組掃描填 cache，看 go/no-go
python run.py     # 3. random vs OPRO vs hybrid 對照 (分數查 cache，僅 meta 呼叫大模型)
python plot.py    # 4. 出 results/comparison.png (對比曲線 + Pareto)
```

> 💰 成本提醒：加了 LLM 判官（正確性+忠實度）後，每題每配置約 3 個 API 呼叫。
> 想省錢可調低 `eval.py` 的 `n`（題數）或 `run.py` 的 `BUDGET`（每策略評估次數）。

## 圖形介面 Web UI（推薦，免 CLI）

不想碰 CLI／讀 JSON 的話，用網頁版：填表單 → 按執行 → 看即時推理與圖表
（best-objective 曲線、正確率/幻覺率 Pareto、配置排行榜、最佳配置卡）。

開**兩個終端機**：

```powershell
# 終端機 1：後端 API (FastAPI, :8000) — 需先填好 .env 金鑰
python server.py

# 終端機 2：前端 (Vite + React, :5173)
cd web
npm install      # 只需第一次
npm run dev
```

瀏覽器開 http://localhost:5173 。前端 `/api` 會自動代理到後端 :8000。

| 前端檔案 | 角色 |
|----------|------|
| `web/src/App.jsx` | 主畫面 + 輪詢狀態 |
| `web/src/components/ConfigForm.jsx` | 設定表單（策略 / 題數 / budget / λ） |
| `web/src/components/StatusPanel.jsx` | 即時進度條 + OPRO 推理 log |
| `web/src/components/BestConfigCard.jsx` | 勝出配置卡（含正確率/幻覺率/成本） |
| `web/src/components/ObjectiveChart.jsx` | best-objective 對比曲線 |
| `web/src/components/ParetoChart.jsx` | 正確率 vs 幻覺率 Pareto 散點 |
| `web/src/components/Leaderboard.jsx` | 試過的配置排行榜 |

## 被優化的搜索空間 θ (216 組)

```python
chunk_size      : [256, 512, 1024]     # 數值
top_k           : [3, 5, 8]            # 數值
retriever       : ["bm25", "dense", "hybrid"]
rerank          : [False, True]        # agentic: 檢索後重排
query_decompose : [False, True]        # agentic: 多跳拆解
verify          : [False, True]        # agentic: NLI 自我檢查 (降幻覺)
```
> 216 組全掃不可行 → 這正是需要最佳化器的理由。`cache.json` 讓三種策略重疊到的配置零成本。

## 里程碑

- [x] D1–D5 RAG / 評估 / 掃描 / 軌跡 / OPRO / 對照曲線
- [x] NLI 忠實度（metric + 推理時守門員，降幻覺）
- [x] 多目標：正確率 vs 幻覺率（+ 成本追蹤）
- [x] Query 分解、rerank 兩個 agentic 開關
- [x] LLM-as-judge 語意正確性（判官與生成模型不同家族）
- [x] 混合最佳化（LLM 縮範圍 → Optuna 收斂，中場重新縮範圍）
- [x] 快取式評估（支援大搜索空間，重疊零成本；鍵含模型/prompt 指紋）
- [x] 棄答統一計分（不論 verify 開關）＋ abstain 失敗分群
- [x] 進程級 embedding 快取＋分批；rerank 只 embed 候選子集
- [x] 評估失敗率門檻（>10% 整輪作廢不落 cache）＋ 成對符號檢定 go/no-go

> 程式全部完成、import 圖無循環、語法通過。實際數字要填好 `.env` 金鑰、`pip install` 後跑出來。

## Roadmap (要上生產 / 投論文還缺的)

- [ ] 測試集放大到上百題（成對符號檢定已內建於 sweep，樣本大才有檢定力）
- [ ] 成本進 objective：做「正確率 / 幻覺 / 成本」三目標 Pareto
- [ ] 接 RAGAS 或專用 NLI 模型，交叉驗證 LLM 判官
- [ ] 跨 run 經驗記憶（warm-start：用過去相似資料集的最佳配置暖身）
- [ ] 換私有模型端點（任何 OpenAI 相容 API 皆可），方便企業內網部署
