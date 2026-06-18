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

> ⚠️ 定位誠實說：這是**研究/Demo 等級的最小骨架**，搜索空間刻意縮成 27 組以方便教學與對照。
> 要上正式生產，需擴大參數空間、加入更嚴謹的評估集與成本/延遲監控（見下方 Roadmap）。

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

| 檔案 | 角色 | 狀態 |
|------|------|------|
| `rag.py` | 被優化的 Agentic RAG (chunk + bm25/dense/hybrid + generate) | ✅ D1 |
| `eval.py` | HotpotQA 子集 + EM/F1/recall@k 評估 (含 per-題明細) | ✅ D2 |
| `sweep.py` | 全 27 組掃描 → `oracle.json` + go/no-go 報告 | ✅ D2.5 |
| `memory/trajectory.py` | 經驗軌跡庫 (config+分數+失敗案例, jsonl) | ✅ D3 |
| `optimizer.py` | search_space + 去重 + random baseline + OPRO 迴路 | ✅ D3/D4 |
| `run.py` | 編排 random vs OPRO 對照實驗 | ✅ D5 |
| `plot.py` | 評估次數 vs best-F1 對比曲線 | ✅ D5 |

## 跑法 (依序)

```powershell
python rag.py     # 0. 冒煙測試: 只測 API 通不通 (不連 HotpotQA)
python eval.py    # 1. D2: 對預設配置跑 30 題，吐 EM/F1/recall
python sweep.py   # 2. D2.5: 全 27 組掃描存 oracle.json (唯一一次大量 API)，看 go/no-go
python run.py     # 3. D5: random vs OPRO 對照 (分數查 oracle，僅 OPRO 呼叫 meta-LLM)
python plot.py    # 4. D5: 出 results/comparison.png
```

## 被優化的搜索空間 θ (Demo 縮到 3 維, 共 27 組)

```python
chunk_size : [256, 512, 1024]
top_k      : [3, 5, 8]
retriever  : ["bm25", "dense", "hybrid"]
```

## 里程碑

- [x] D1 接通 NVIDIA API + 最陽春 RAG (`rag.py`)
- [x] D2 評估腳本 + 30 題分層測試集 (`eval.py`)
- [x] **D2.5 go/no-go**: 全 27 組存 oracle，看最佳與最差 F1 差距 (`sweep.py`)
- [x] D3 軌跡庫 + random search baseline (`memory/trajectory.py`, `optimizer.py`)
- [x] D4 OPRO 迴路 — 反思 → 縮範圍 → 提案，去重/合法性檢查放程式裡 (`optimizer.py`)
- [x] D5 random vs OPRO 對照曲線 (`run.py`, `plot.py`)

> 程式骨架全部完成、語法通過。實際數字要等你填好 `.env` 金鑰、`pip install` 後跑出來。
> 跑 `sweep.py` 後若 go/no-go 顯示差距太小，先調高 `eval.py` 的 `n_hard` 再重跑。

## Roadmap (要上生產 / 投論文還缺的)

- [ ] 擴大搜索空間：加入 reranker 開關、query rewrite、prompt 模板變體（目前刻意縮成 27 組）
- [ ] 混合最佳化：LLM 縮範圍 → Optuna 在範圍內收斂（純 LLM 後期收斂較弱）
- [ ] 更嚴謹評估：加忠實度（NLI/RAGAS faithfulness），測試集放大到上百題
- [ ] 成本/延遲監控：把每組配置的 token 成本與回應延遲也納入目標
- [ ] 換私有模型端點（任何 OpenAI 相容 API 皆可），方便企業內網部署
