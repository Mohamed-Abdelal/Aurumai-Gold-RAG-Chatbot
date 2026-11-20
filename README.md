# 🪙 AurumAI — Gold Investment Recommendation Assistant

**Course:** AIE417 — Selected Topics in AI 2 (Assignment 2)  
**Stack:** Python · Pure NumPy (Cosine Similarity & Top-K) · SentenceTransformers (`all-MiniLM-L6-v2`) · Groq API (`openai/gpt-oss-120b`) · Streamlit

---

## 1. Bot Idea, Purpose & Target Users

### Problem Statement
Retail gold investors—especially beginners and savers in Egypt—struggle with fragmented, conflicting advice across physical bullion dealers, gold ETFs, mutual funds, and social media. General-purpose pretrained LLMs fall short in three critical ways:
1. **Stale & Generic Advice:** They miss current 2026 pullback dynamics, exact ETF expense ratios, and local Egyptian market rules (*masna'eya* tiers, sealed-pack cashback rules, and FRA-regulated AZ-Gold redemption schedules).
2. **One-Size-Fits-All Answers:** Without structured investor decision rules, base models fail to tailor recommendations strictly to a user's budget, time horizon, and liquidity needs (e.g., failing to warn short-term cash holders away from gold).
3. **Weak Domain Boundaries:** Ungrounded LLMs happily answer out-of-scope financial queries (such as cryptocurrency yield farming or real estate mortgages) instead of refusing safely.

### Solution: AurumAI
**AurumAI** is a domain-specific RAG-powered Gold Investment Recommendation Assistant grounded over a curated **20-article knowledge base** covering global gold investing and Egypt-specific bullion/fund rules. For every user query, AurumAI:
- Embeds the question into a 384-dimensional dense vector (`all-MiniLM-L6-v2`),
- Computes **pure-NumPy cosine similarity** (`np.dot`, `np.linalg.norm`, `np.argsort`) across all **44 title-anchored chunks** to retrieve the **Top-4** most relevant passages,
- Grounds a fast free-tier **Groq LLM** (`openai/gpt-oss-120b`, `max_completion_tokens=1024`, `reasoning_effort="low"`, `temperature=0.2`) strictly in the retrieved articles—tailoring guidance to the user's budget and horizon, citing exact source filenames, refusing out-of-scope queries, and appending an educational disclaimer.

### Target Users
- **Students & Small-Budget Savers** building disciplined monthly Dollar-Cost Averaging (DCA) plans without losing money to high small-bar premiums.
- **Long-Term Physical Bullion Savers (Egypt & Global)** comparing 24K bars (*sabeeka*) vs. 21K gold pounds, *masna'eya* cashback rules, hallmarks, and tax invoices.
- **Diversified Portfolio Builders** implementing 5%–15% or 60/20/20 allocations and navigating 2026 gold volatility.

---

## 2. Curated Knowledge Base (`data/`)

The knowledge base consists of **20 clean `.txt` articles** (each formatted with a standardized 4-line header: `Title:`, `Category:`, `Date:`, `Source:`) spanning 5 thematic pillars:

| Pillar / Category | Files | Key Topics Covered |
| :--- | :---: | :--- |
| **1. Beginner Investment Vehicles** | `01`–`05` (5 files) | Physical bullion bars & coins, Gold ETFs (`GLD`, `IAU`, `GLDM`), digital gold platforms, gold mining & royalty stocks, side-by-side vehicle comparison |
| **2. Portfolio Allocation & Strategies** | `06`–`10` (5 files) | 5%–15% allocation rules, 60/20/20 balanced portfolio split, monthly DCA vs. lump sum, rebalancing & 3–5+ year horizon, inflation & currency-depreciation hedging |
| **3. Risks, Costs & 2026 Market** | `11`–`15` (5 files) | Opportunity cost & drawdown risks, bid-ask spreads & hidden transaction costs, home/bank/vault storage fees, 2026 rally & 8%–15% pullback dynamics, volatility management |
| **4. Egypt Gold Buying Guide** | `16`–`19` (4 files) | 24K bullion bars (*sabeeka*) vs. 21K gold pounds (8g / 7g pure), making charges (*masna'eya*) & 35%–50% sealed-pack cashback, FRA-regulated gold mutual funds (`AZ-Gold` by Azimut Egypt), official *Maslahat Al-Damgha* hallmark & 7-field tax invoice/buyback checklist |
| **5. Investor Decision Matrix** | `20` (1 file) | `20_gold_investor_decision_matrix.txt` mapping 4 concrete profiles (**Student/Small Monthly Budget**, **Long-Term Physical Saver**, **Diversified Portfolio Builder**, **Short-Term Cash Holder**) to actionable recommendations & pitfalls |

---

## 3. RAG Implementation Architecture (Pure NumPy)

```text
20 Curated Gold Investment Articles (data/*.txt)
   │
   ▼  src/loader.py       — Clean .txt/.md reader + 4-line header parser (Title, Category, Date, Source)
20 Structured Documents
   │
   ▼  src/chunker.py      — Word-based overlapping chunker (chunk_size=220 words, overlap=40 words)
44 Overlapping Chunks
   │
   ▼  scripts/build_index.py & src/embeddings.py
      Prepends `Title: <doc_title>\n\n` to every chunk before all-MiniLM-L6-v2 (384-dim float32) encoding
(44 × 384) NumPy Matrix   — Saved to index/embeddings.npz (61.5 KB) + index/chunks.json (74.2 KB)
   │
   ▼  src/retriever.py    — Pure-NumPy Cosine Similarity + Descending Top-K (default top_k=4)
Top-4 Evidence Chunks
   │
   ▼  src/llm.py          — Groq API (openai/gpt-oss-120b, max_completion_tokens=1024, reasoning_effort="low", temp=0.2)
Profile-Tailored Answer + Filename Citations + Safe Out-of-Scope Refusal + Educational Disclaimer
```

### Pure-NumPy Cosine Similarity & Top-K Retrieval (`src/retriever.py`)
No vector database or orchestration framework (no LangChain, LlamaIndex, FAISS, or Chroma) is used. All similarity math is implemented transparently in NumPy:

$$
\text{cosine similarity}(q, d_i) = \frac{q \cdot d_i}{\|q\|_2 \, \|d_i\|_2}
$$

```python
dot_products = np.dot(docs, q)
query_norm = float(np.linalg.norm(q))
doc_norms = np.linalg.norm(docs, axis=1)
denominators = np.maximum(doc_norms * query_norm, 1e-10)
similarities = np.clip(dot_products / denominators, -1.0, 1.0)
sorted_indices = np.argsort(similarities)[::-1]  # Descending Top-K
```

---

## 4. Project Structure

```text
.
├── .streamlit/
│   └── config.toml                # AurumAI dark-gold theme & server settings
├── data/
│   ├── 01_physical_gold_bullion_bars_and_coins.txt
│   ├── ... (20 clean .txt gold investment articles with 4-line headers)
│   └── 20_gold_investor_decision_matrix.txt
├── index/
│   ├── embeddings.npz             # Pre-computed (44 × 384) float32 NumPy embedding matrix
│   └── chunks.json                # Serialized chunk texts & provenance metadata
├── src/
│   ├── config.py                  # Paths, 220/40 word chunking, Top-4 & Groq defaults
│   ├── loader.py                  # .txt/.md reader & 4-line header parser
│   ├── chunker.py                 # Word-based overlapping chunker + `Title:` embedding prefix
│   ├── embeddings.py              # SentenceTransformers all-MiniLM-L6-v2 wrapper
│   ├── retriever.py               # Pure-NumPy cosine similarity & Top-K vector index
│   ├── llm.py                     # Groq client (With-RAG AurumAI & Without-RAG baseline)
│   └── rag_pipeline.py            # End-to-end RAG orchestrator
├── scripts/
│   └── build_index.py             # Offline data/*.txt -> Chunk -> Embed -> .npz index builder
├── eval/
│   ├── questions.json             # 10 test questions (6 fact, 2 multi-doc profile, 2 out-of-scope)
│   ├── run_evaluation.py          # Automated With-RAG vs. Without-RAG benchmark (5s Groq sleep)
│   ├── results.json               # Saved benchmark JSON results
│   └── evaluation_report.md       # Full quantitative & qualitative evaluation report
├── tests/
│   └── test_rag_core.py           # Unit tests for NumPy cosine sim, header parsing, 220/40 chunking
├── app.py                         # Multi-tab AurumAI Streamlit web application
├── requirements.txt               # Minimal Python dependencies
├── VIVA_GUIDE.md                  # Oral defense cheat sheet & live demo script
└── README.md
```

---

## 5. Quickstart (Run Locally)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Your Free Groq API Key
Get a free API key at [https://console.groq.com/keys](https://console.groq.com/keys), then copy `.env.example` to `.env`:
```bash
cp .env.example .env
# Edit .env and set: GROQ_API_KEY=gsk_...
```

### 3. Rebuild Index (Optional — Pre-Built Index Included)
```bash
python scripts/build_index.py
```

### 4. Launch AurumAI Streamlit App
```bash
streamlit run app.py
```

### 5. Run Unit Tests & Full Evaluation
```bash
python -m pytest -v
python eval/run_evaluation.py --model openai/gpt-oss-120b --top-k 4 --delay 5.0
```

---

## 6. Evaluation Results: With RAG (AurumAI) vs. Without RAG

We evaluated **AurumAI** across **10 test questions** ([`eval/questions.json`](eval/questions.json)) using Groq `openai/gpt-oss-120b` (`top_k=4`, `min_similarity=0.25`, 5-second sleep between calls):
- **6 Specific Gold / Egypt Fact Questions** (ETF expense ratios & spreads, 60/20/20 split, 2026 pullback catalysts, Egypt 21K gold pound specs, *masna'eya* & cashback rules, AZ-Gold redemption tiers)
- **2 Multi-Document Profile Recommendation Questions** (Egyptian student saving EGP 2,000/month for 4 years; short-term saver needing EGP 150,000 in 6 months)
- **2 Out-of-Scope Refusal Questions** (Ethereum Layer-2 / Bitcoin DeFi yields; Central Bank of Egypt residential real estate mortgages)

### Summary Comparison Table

| Metric | With RAG (AurumAI) | Without RAG (Baseline) | Improvement |
| :--- | :---: | :---: | :---: |
| **In-Scope Retrieval Hit@4** | **100.0% (8/8)** | N/A | — |
| **Mean Top-1 Cosine Similarity (In-Scope)** | **`0.7256`** | N/A | — |
| **Mean Key-Fact / Refusal Recall** | **98.0%** | 30.9% | **+67.1 pp** |
| **Out-of-Scope Safe Refusal Rate** | **100.0% (2/2)** | 0.0% (0/2) | **+100.0 pp** |
| **Mean Correctness Score (1–5, LLM Judge)** | **5.00 / 5** | 1.90 / 5 | **+3.10** |
| **Mean Relevance Score (1–5, LLM Judge)** | **5.00 / 5** | 4.00 / 5 | **+1.00** |
| **Hallucination / Boundary-Violation Rate** | **0.0%** | 20.0% | **-20.0 pp** |

### Per-Question Breakdown

| # | Type | Target File(s) | Hit@4 / Refusal | Top-1 Sim | Recall (RAG vs No-RAG) | Correctness (RAG vs No-RAG) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| Q1 | `fact` | `02_gold_etfs_and_spot_trackers.txt` | Yes | 0.658 | 100% vs 40% | 5/5 vs 2/5 |
| Q2 | `fact` | `07_60_20_20_balanced_portfolio_with_gold.txt` | Yes | 0.677 | 100% vs 80% | 5/5 vs 4/5 |
| Q3 | `fact` | `14_2026_gold_market_rally_and_pullback_dynamics.txt` | Yes | 0.707 | 100% vs 60% | 5/5 vs 2/5 |
| Q4 | `fact` | `16_egypt_24k_gold_bars_vs_21k_gold_pounds.txt` | Yes | 0.829 | 100% vs 60% | 5/5 vs 1/5 |
| Q5 | `fact` | `17_egypt_making_charges_masnaeya_and_cashback.txt` | Yes | 0.706 | 100% vs 0% | 5/5 vs 2/5 |
| Q6 | `fact` | `18_egypt_gold_mutual_funds_az_gold_and_regulated_options.txt` | Yes | 0.797 | 100% vs 29% | 5/5 vs 1/5 |
| Q7 | `multi_doc_profile` | `20_gold_investor_decision_matrix.txt`, `18_...`, `17_...` | Yes | 0.750 | 100% vs 20% | 5/5 vs 2/5 |
| Q8 | `multi_doc_profile` | `20_gold_investor_decision_matrix.txt`, `09_...`, `12_...` | Yes | 0.680 | 80% vs 20% | 5/5 vs 2/5 |
| Q9 | `out_of_scope` | _Out of Scope (Crypto / DeFi)_ | Refused | 0.349 | 100% vs 0% | 5/5 vs 2/5 |
| Q10 | `out_of_scope` | _Out of Scope (Egypt Mortgages)_ | Refused | 0.518 | 100% vs 0% | 5/5 vs 1/5 |

Full qualitative side-by-side transcripts are saved in [`eval/evaluation_report.md`](eval/evaluation_report.md) and rendered interactively inside the Streamlit **📈 Evaluation Results** tab.

---

## 7. Deploying Live Demo to Streamlit Community Cloud

The repository is lightweight (~200 KB total including all 20 `.txt` articles and the pre-built NumPy index) and ready for instant deployment on **Streamlit Community Cloud**:

1. **Push to GitHub:** Create a new GitHub repository and push this project (`data/`, `index/`, `eval/`, `src/`, `scripts/`, `tests/`, `app.py`, `requirements.txt`, `.streamlit/config.toml`). Secrets (`.env`) are automatically excluded by `.gitignore`.
2. **Deploy on Streamlit Cloud:**
   - Go to [https://share.streamlit.io](https://share.streamlit.io) → click **Create app** → **Deploy a public app from GitHub**.
   - Select your repository, branch (`main`), and main file path: `app.py`.
3. **Configure Secrets (Recommended for Live Demo):**
   - Open **Advanced settings → Secrets** (or **Settings → Secrets** after deploy) and add:
     ```toml
     GROQ_API_KEY = "gsk_your_groq_api_key_here"
     ```
   - *(Visitors can also paste their own free Groq API key directly in the sidebar, and Tab 3's Knowledge Base Explorer & Pure-NumPy Retrieval Debugger work even without an API key.)*
4. **Click Deploy:** Streamlit Cloud installs `requirements.txt` (using lightweight CPU PyTorch wheels), loads the pre-built `index/embeddings.npz` + `index/chunks.json` in seconds, and launches the live AurumAI web demo.

---

*Disclaimer: AurumAI is an educational AI project and does not provide licensed financial or investment advice.*
