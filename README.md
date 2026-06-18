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

**多目標最佳化**
- objective = `正確率 − 0.5 × 幻覺率`，並追蹤成本（latency / tokens）。
- OPRO 看的是**失敗分群**（檢索失敗 vs 生成幻覺），據此推論該開哪個 agentic 開關。

**嚴謹評估**
- LLM-as-judge 算**語意正確性**（EM/F1 對自由文本太嚴）＋ **NLI 忠實度**（幻覺率）。
- 判官模型刻意與生成模型不同，避免自我偏袒。

**三種最佳化策略對照**
- `random`（baseline）／`OPRO`（LLM 推理提案）／`hybrid`（LLM 縮範圍 → Optuna 收斂）。

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
| `rag.py` | Agentic RAG：chunk + bm25/dense/hybrid + rerank + query 分解 + NLI 守門員 |
| `judge.py` | LLM-as-judge：NLI 忠實度 + 語意正確性（判官 ≠ 生成模型） |
| `eval.py` | HotpotQA 分層子集 + EM/F1/recall/忠實度/正確率/成本 + 失敗分群 |
| `cache.py` | 快取式評估（查過不重跑），取代不可行的全掃 |
| `sweep.py` | 核心網格 27 組掃描 → 填 cache + go/no-go 報告 |
| `memory/trajectory.py` | 經驗軌跡庫（config+分數+失敗案例, jsonl） |
| `optimizer.py` | 搜索空間(216) + 多目標 + random baseline + OPRO 迴路 |
| `hybrid.py` | LLM 縮範圍 → Optuna 在範圍內收斂 |
| `run.py` | 編排 random vs OPRO vs hybrid 對照 |
| `plot.py` | best-objective 對比曲線 + 正確率/幻覺率 Pareto 散點 |

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
- [x] LLM-as-judge 語意正確性（判官 ≠ 生成模型）
- [x] 混合最佳化（LLM 縮範圍 → Optuna 收斂）
- [x] 快取式評估（支援大搜索空間，重疊零成本）

> 程式全部完成、import 圖無循環、語法通過。實際數字要填好 `.env` 金鑰、`pip install` 後跑出來。

## Roadmap (要上生產 / 投論文還缺的)

- [ ] 測試集放大到上百題並做統計顯著性
- [ ] 成本進 objective：做「正確率 / 幻覺 / 成本」三目標 Pareto
- [ ] 接 RAGAS 或專用 NLI 模型，交叉驗證 LLM 判官
- [ ] 跨 run 經驗記憶（warm-start：用過去相似資料集的最佳配置暖身）
- [ ] 換私有模型端點（任何 OpenAI 相容 API 皆可），方便企業內網部署
