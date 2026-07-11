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
- **rerank**：檢索後用**專用 reranker 模型**（NIM `nv-rerankqa`）對候選重排再取 top_k；端點不可用時自動退回 dense 相似度重排。
- **query_decompose**：多跳問題先拆成子問題各自檢索（HotpotQA 是多跳）。
- **verify（NLI 自我檢查守門員）**：生成後用 NLI 判斷答案是否被 context 支持；不過關 → 自動拓寬檢索重答，仍不忠實 → **棄答**而非硬掰。這是「降幻覺」的關鍵。
  棄答**不計入幻覺**（沒有捏造內容），另以獨立的 abstain rate 追蹤，避免 objective 雙重懲罰而教壞最佳化器。
  **不論 verify 開關**，生成器自行棄答（"I don't have enough evidence"）也按同一規則計分，
  否則 verify=False 的誠實棄答會被判官記成幻覺，objective 隱性補貼 verify、對比失真。

**多目標最佳化**
- objective = `正確率 − λ×幻覺率 − μ×每題成本(ktok)`；λ、μ 都是 UI/CLI 可調的權重（μ=0 時成本只追蹤不懲罰）。
- OPRO 看的是**失敗分群**（檢索失敗 vs 生成幻覺 vs 棄答）＋**各參數取值的邊際統計**，據此推論該開哪個 agentic 開關。
- **Racing 提早停止**：評到一半若「剩餘題全對」的樂觀上界仍追不上目前最佳配置，直接中止該配置的評估——同樣預算能試更多配置。
- **warm-start（選配）**：記住每個資料集的贏家配置，新資料集用「最相似的過去經驗」暖身（會破壞與 random 的公平對照，故 opt-in）。

**嚴謹評估**
- LLM-as-judge 算**語意正確性**（EM/F1 對自由文本太嚴）＋ **NLI 忠實度**（幻覺率）＋ 棄答率。
- 判官模型刻意與生成模型**不同家族**（Mixtral vs Llama），同家族大小模型仍可能自我偏袒。
- 失敗分群分「檢索 / 生成 / 棄答」三類；答對但 recall 不滿的題不算失敗。
- retrieval recall 要求 gold 段落被覆蓋 ≥50% 字元才算命中，避免「撈到 256 字皮毛就滿分」偏袒小 chunk。
- **中文支援**：棄答偵測繁/簡中文 pattern、CJK 標點正規化、字元級 F1（空白斷詞對中文整句只算一個 token）。
- token 成本涵蓋生成、query 分解、embedding、reranker 與守門員判官；所有 API 呼叫遇 429/5xx 自動指數退避重試。
- 單題偶發失敗會跳過，但**失敗率 >10% 整輪作廢、不寫入快取**（避免 429 風暴的壞分數被永久快取）。
- sweep 的 go/no-go 用**成對符號檢定**（同題集逐題比較）判斷區分度，而非只看平均分差距（n=30 時噪音 ≈0.09）。
- `run.py --seeds K` 跑多 seed，plot 畫 **mean±range 帶狀**——「OPRO 贏過 random」的結論不再只憑單一 seed（cache 跨 seed 共用，重疊配置免費）。

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
| `dataset.py` | **自訂資料集**：你的 .txt/.md/.pdf 文件 → 段落語料 → LLM 生成可抽取式 QA（單跳＋選配多跳，或自備 QA）→ 沿用整條評估/最佳化管線；PDF 文字層優先、掃描頁走 VLM 轉錄 |
| `memory/warmstart.py` | 跨 run 經驗記憶：記住各資料集的贏家配置，相似資料集 warm-start 暖身 |
| `tests/` | 離線測試（pytest，不打 API）＋ GitHub Actions CI |
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

> 💰 成本提醒：加了 LLM 判官（正確性+忠實度）後，每題每配置約 3 個 API 呼叫；
> Racing 提早停止會自動砍掉「注定追不上」的評估。想再省可調低 `eval.py` 的 `n`（題數）
> 或 `run.py` 的 `BUDGET`（每策略評估次數）。
>
> `run.py` 進階參數：`--seeds K`（多 seed，plot 畫 mean±range 帶狀）、
> `--mu M`（成本懲罰進 objective）、`--warmstart`（用過去 run 的贏家配置暖身）。

## 用自己的文件調參 (Bring your own docs)

不想用 HotpotQA、想對「你的知識庫」找最佳 RAG 配置：

```powershell
# 1. 用 LLM 對你的文件生成 QA 評估集 (一次性；.txt/.md/.pdf 檔或資料夾皆可)
#    --multihop: 約一半生成「跨段落多跳題」，query_decompose 開關才有東西可學
python dataset.py --corpus docs/ --n 20 --out data/qa.jsonl --multihop

# 2. 之後 sweep / 對照 / 出圖全部沿用，只是多帶兩個參數
python sweep.py --corpus docs/ --qa data/qa.jsonl
python run.py   --corpus docs/ --qa data/qa.jsonl --seeds 3
python plot.py
```

也可以**自備 QA**（跳過生成）：jsonl 每行一筆
`{"question": "...", "answer": "...", "gold": ["支撐段落原文，選填"]}`。

規則與限制：
- 語料以**空行分段**，每題的檢索範圍 = 整份語料（真實 RAG 情境）；上限 500 段（demo 級）。
- **PDF**：逐頁先抽文字層（原生數位 PDF 免費零錯字）；無文字層的掃描頁才送 VLM 轉錄。
  ⚠️ VLM 轉錄可能有錯字且會成為評估語料，掃描檔請抽查品質。
- LLM 生成的 QA 是**可抽取式短答**（答案必為文件內的 span），不合格的自動丟棄。
- 自備 QA 若沒給 `gold`，會用「答案子字串」自動定位支撐段落；**答案不在文件裡的題會被剔除**（無法評檢索品質）。
- 繁/簡中文文件可用（棄答偵測、字元級 F1、標點正規化都有處理）；判官預設 Mixtral，
  全中文評估集建議在 `.env` 用 `JUDGE_MODEL` 換中文較強的非 Llama 模型。
- 快取鍵含語料+QA 內容 hash：改文件或 QA 後舊分數自動失效。

Web UI 同樣支援：資料集選「自己的文件」→ 上傳/貼上文件 → 選「LLM 自動生成」或「上傳自備 QA」→ 開始最佳化。首次生成的 QA 會存到 `data/`，同語料同題數之後直接重用。

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
| `web/src/components/ConfigForm.jsx` | 設定表單（資料集：HotpotQA 或自己的文件＋QA / 策略 / 題數 / budget / λ） |
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
- [x] 自訂資料集：使用者文件 + LLM 生成/自備 QA（CLI 與 Web UI）
- [x] PDF 支援（文字層優先，掃描頁 VLM 轉錄）＋ 多跳 QA 生成
- [x] 中文支援（棄答偵測、字元級 F1、標點正規化）
- [x] Racing 提早停止＋多 seed 帶狀圖＋成本進 objective（μ）
- [x] 專用 reranker 模型＋跨 run warm-start＋Run 歷史/HTML 報告
- [x] 離線測試 (pytest) + GitHub Actions CI

> 程式全部完成、import 圖無循環、語法通過。實際數字要填好 `.env` 金鑰、`pip install` 後跑出來。

## Roadmap (要上生產 / 投論文還缺的)

- [ ] 測試集放大到上百題（成對符號檢定/多 seed 已內建，樣本大才有檢定力）
- [ ] 「正確率 / 幻覺 / 成本」三目標 Pareto 圖（μ 純量化已支援，缺 3D 前緣視覺化）
- [ ] 接 RAGAS 或專用 NLI 模型，交叉驗證 LLM 判官
- [ ] Ablation：OPRO 拿掉失敗分群/邊際統計後還剩多少優勢（驗證「推理」真的有用）
- [ ] 換私有模型端點：所有模型 id 已可用 .env 覆寫（任何 OpenAI 相容 API），缺整套內網部署文件
