# 🪙 AurumAI — Gold Investment RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers gold-investing questions using a curated knowledge base of 20 articles, pure-NumPy cosine similarity search, a pretrained Groq LLM, and a Streamlit web interface.

---

## 1. Bot Idea, Purpose & Target Users

**Problem:** Beginners and retail savers—especially in Egypt—face confusing, scattered advice about buying gold (24K bars vs. 21K gold pounds, making charges/*masna'eya*, Gold ETFs, local gold mutual funds like AZ-Gold, and 2026 market pullbacks). General LLMs often give generic or outdated numbers, fail to tailor advice to a user's budget and horizon, and answer out-of-scope financial questions instead of refusing.

**Solution:** **AurumAI** retrieves relevant passages from a curated 20-article gold-investing knowledge base and grounds a pretrained LLM (`openai/gpt-oss-120b` via Groq) strictly in those articles—citing exact source filenames, tailoring recommendations to budget/horizon, refusing out-of-scope topics (like crypto or mortgages), and adding an educational disclaimer.

**Target Users:**
- **Students & Small-Budget Savers** building monthly Dollar-Cost Averaging (DCA) plans.
- **Physical Gold Buyers (Egypt & Global)** comparing 24K bars vs. 21K gold pounds, *masna'eya* cashback rules, and invoice/hallmark checklists.
- **Portfolio Builders** planning 5%–15% or 60/20/20 gold allocations.

---

## 2. Knowledge Base (`data/`)

20 clean `.txt` articles (each starting with a 4-line header: `Title:`, `Category:`, `Date:`, `Source:`) covering 5 topics:
1. **Beginner Investment Vehicles (`01`–`05`):** Physical bullion, Gold ETFs (`GLD`, `IAU`, `GLDM`), digital gold platforms, mining/royalty stocks, and a vehicle comparison.
2. **Portfolio Allocation & Strategies (`06`–`10`):** 5%–15% rule, 60/20/20 split, monthly DCA vs. lump sum, rebalancing & 3–5+ year horizon, inflation/currency hedging.
3. **Risks, Costs & 2026 Market (`11`–`15`):** Opportunity cost, bid-ask spreads, storage/custody costs, 2026 rally & 8%–15% pullback dynamics, volatility management.
4. **Egypt Gold Buying Guides (`16`–`19`):** 24K bars (*sabeeka*) vs. 21K gold pounds, making charges (*masna'eya*) & cashback rules, FRA-regulated gold mutual funds (`AZ-Gold`), official hallmark & tax invoice buyback checklist.
5. **Investor Decision Matrix (`20`):** `20_gold_investor_decision_matrix.txt` mapping 4 investor profiles (student/small monthly budget, long-term physical saver, diversified portfolio builder, short-term cash holder) to concrete recommendations.

---

## 3. RAG Implementation

- **Document Loading & Chunking (`src/loader.py`, `src/chunker.py`):** Reads all 20 `.txt` files and splits them into **44 overlapping chunks** (`chunk_size = 220` words, `overlap = 40` words). Prepends `Title: <doc_title>\n\n` to every chunk before embedding so mid-article paragraphs stay tied to their topic.
- **Embeddings (`src/embeddings.py`):** Encodes chunks and queries into 384-dimensional vectors using `sentence-transformers/all-MiniLM-L6-v2` and saves them to `index/embeddings.npz` and `index/chunks.json`.
- **Pure-NumPy Cosine Similarity & Top-K Retrieval (`src/retriever.py`):** Implemented directly with NumPy (`np.dot`, `np.linalg.norm`, `np.argsort`) with no external vector database, returning the **Top-4** most relevant chunks:

$$
\text{cosine similarity}(q, d_i) = \frac{q \cdot d_i}{\|q\|_2 \, \|d_i\|_2}
$$

<<<<<<< HEAD
- **LLM Integration (`src/llm.py`):** Sends the retrieved Top-4 chunks to Groq (`openai/gpt-oss-120b`, `temperature = 0.2`, `max_completion_tokens = 1024`) and also provides a **Without-RAG** mode for side-by-side comparison.
- **User Interface (`app.py`):** Clean 3-tab Streamlit app (**Chatbot**, **With RAG vs. Without RAG**, and **Evaluation Results**).
=======
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
<<<<<<< HEAD
├── data/                       # 20 curated gold-investing .txt articles
├── index/                      # Pre-built NumPy embeddings (.npz) & chunks (.json)
=======
├── data/
│   ├── 01_physical_gold_bullion_bars_and_coins.txt
│   ├── ... (20 clean .txt gold investment articles with 4-line headers)
│   └── 20_gold_investor_decision_matrix.txt
├── index/
│   ├── embeddings.npz             # Pre-computed (44 × 384) float32 NumPy embedding matrix
│   └── chunks.json                # Serialized chunk texts & provenance metadata
=======
├── src/
│   ├── config.py               # Paths and RAG hyperparameters
│   ├── loader.py               # .txt loader & 4-line header parser
│   ├── chunker.py              # 220-word / 40-overlap chunker + Title prefix
│   ├── embeddings.py           # all-MiniLM-L6-v2 wrapper
│   ├── retriever.py            # Pure-NumPy cosine similarity & Top-K search
│   ├── llm.py                  # Groq client (With RAG & Without RAG)
│   └── rag_pipeline.py         # End-to-end RAG orchestrator
├── scripts/build_index.py      # Rebuilds index/ from data/
├── eval/
<<<<<<< HEAD
│   ├── questions.json          # 10 test questions (6 fact, 2 profile, 2 out-of-scope)
│   ├── run_evaluation.py       # Automated With-RAG vs. Without-RAG evaluation script
│   ├── results.json            # Saved evaluation metrics & responses
│   └── evaluation_report.md    # Full qualitative comparison report
├── app.py                      # Streamlit web application
└── requirements.txt            # Python dependencies
=======
│   ├── questions.json             # 10 test questions (6 fact, 2 multi-doc profile, 2 out-of-scope)
│   ├── run_evaluation.py          # Automated With-RAG vs. Without-RAG benchmark (5s Groq sleep)
│   ├── results.json               # Saved benchmark JSON results
│   └── evaluation_report.md       # Full quantitative & qualitative evaluation report
├── app.py                         # Multi-tab AurumAI Streamlit web application
├── requirements.txt               # Minimal Python dependencies
└── README.md
```

---

## 5. How to Run Locally & Deploy on Streamlit Cloud

### Run Locally
```bash
pip install -r requirements.txt
cp .env.example .env            # Add your free GROQ_API_KEY from https://console.groq.com/keys
streamlit run app.py
```
*(Optional: rebuild the vector index with `python scripts/build_index.py`, or re-run evaluation with `python eval/run_evaluation.py`.)*

<<<<<<< HEAD
### Deploy Live Demo on Streamlit Community Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app** → select your GitHub repo (`main` branch, `app.py`).
2. Under **Advanced settings**, choose **Python 3.12** and add your key in **Secrets**:
   ```toml
   GROQ_API_KEY = "gsk_your_groq_api_key_here"
   ```
3. Click **Deploy**. *(Users can also paste a free Groq API key directly inside the live app if Secrets are not set.)*
=======
### 5. Re-run Full 10-Question Evaluation
```bash
python eval/run_evaluation.py --model openai/gpt-oss-120b --top-k 4 --delay 5.0
```

---

## 6. Evaluation Results (10 Questions: With RAG vs. Without RAG)

Tested on **10 questions** ([`eval/questions.json`](eval/questions.json)): **6 gold/Egypt fact questions**, **2 multi-document profile recommendation questions**, and **2 out-of-scope refusal questions** (crypto & real estate mortgages).

| Metric | With RAG (AurumAI) | Without RAG (Baseline) | Improvement |
| :--- | :---: | :---: | :---: |
| **In-Scope Retrieval Hit@4** | **100.0% (8/8)** | N/A | — |
| **Mean Key-Fact / Refusal Recall** | **98.0%** | 30.9% | **+67.1 pp** |
| **Out-of-Scope Safe Refusal Rate** | **100.0% (2/2)** | 0.0% (0/2) | **+100.0 pp** |
| **Mean Correctness (1–5 Judge)** | **5.00 / 5** | 1.90 / 5 | **+3.10** |
| **Mean Relevance (1–5 Judge)** | **5.00 / 5** | 4.00 / 5 | **+1.00** |
| **Hallucination / Boundary Violations** | **0.0%** | 20.0% | **-20.0 pp** |

Full question-by-question answers and scores are available in [`eval/evaluation_report.md`](eval/evaluation_report.md) and inside the **📊 Evaluation Results** tab of the app.

---

<<<<<<< HEAD
*Disclaimer: AurumAI is a student educational project and does not provide licensed financial advice.*
=======
## 7. Deploying Live Demo to Streamlit Community Cloud

The repository is lightweight (~200 KB total including all 20 `.txt` articles and the pre-built NumPy index) and ready for instant deployment on **Streamlit Community Cloud**:

1. **Push to GitHub:** Create a new GitHub repository and push this project (`data/`, `index/`, `eval/`, `src/`, `scripts/`, `app.py`, `requirements.txt`). Secrets (`.env`) are automatically excluded by `.gitignore`.
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
