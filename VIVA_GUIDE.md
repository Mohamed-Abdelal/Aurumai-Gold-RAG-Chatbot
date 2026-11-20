# Oral Defense & Technical Study Guide (`VIVA_GUIDE.md`)

> **Course:** AIE417 — Selected Topics in AI 2 (Assignment 2)
> **Project:** AurumAI — Gold Investment Recommendation Assistant (Global & Egypt Markets)

This cheat sheet summarizes every architectural decision, formula, and evaluation result so you can confidently present **AurumAI** during your live demo.

---

## 1. End-to-End Architecture

```text
OFFLINE INDEXING (scripts/build_index.py)
  20 Curated Gold Investment Articles (data/*.txt, 4-line metadata headers)
        │
        ▼  Clean .txt/.md loader + header parser (src/loader.py)
  20 Structured Documents
        │
        ▼  Word-based overlapping chunker: 220 words, 40-word overlap (src/chunker.py)
  44 Text Chunks + Metadata (filename, title, category, date, source)
        │
        ▼  Title-anchored encoding: `Title: <doc_title>\n\n<chunk_text>`
        ▼  SentenceTransformer all-MiniLM-L6-v2 (384-dim float32 dense vectors) (src/embeddings.py)
  NumPy Matrix (44 × 384) saved to index/embeddings.npz (61.5 KB) + index/chunks.json (74.2 KB)

ONLINE QUERY PIPELINE (src/rag_pipeline.py + app.py)
  User Question (with budget, horizon, or Egypt/global specifics)
        │
        ▼  Embed query into 1D vector q ∈ R^384 (src/embeddings.py)
        ▼  Pure-NumPy Cosine Similarity against all 44 chunk vectors + Descending Top-4 (src/retriever.py)
  Top-4 Relevant Chunks (with similarity scores + filenames)
        │
        ▼  Strict AurumAI Prompt + Retrieved Context -> Groq API (openai/gpt-oss-120b) (src/llm.py)
  Tailored Recommendation + Exact Filename Citations + Safe Out-of-Scope Refusal + Disclaimer
```

---

## 2. Core Engineering Talking Points (Likely TA Questions)

### Q1: How does pure-NumPy cosine similarity work? (`src/retriever.py`)
Given query vector $q \in \mathbb{R}^{384}$ and chunk matrix $D \in \mathbb{R}^{44 \times 384}$:

$$\text{cosine\_similarity}(q, d_i) = \frac{q \cdot d_i}{\|q\|_2 \, \|d_i\|_2}$$

1. `dot_products = np.dot(docs, q)` computes all dot products in one vectorized call.
2. `np.linalg.norm(q)` and `np.linalg.norm(docs, axis=1)` compute L2 norms, guarded by `np.maximum(..., 1e-10)`.
3. `np.argsort(scores)[::-1]` ranks all chunks descending and returns the **Top-4** chunks passing `min_similarity = 0.25`.
4. Zero vector databases or wrappers (no LangChain, LlamaIndex, FAISS, or Chroma).

### Q2: Why prepend `Title: <doc_title>\n\n` before embedding? (`src/chunker.py`)
When an article is split into 2–3 word chunks (`chunk_size=220`, `overlap=40`), mid-article paragraphs (such as cashback percentages or AZ-Gold Year-1 redemption fees) may not repeat the article's main subject in every sentence. Prepending `Title: <doc_title>\n\n` before encoding with `all-MiniLM-L6-v2` keeps every mid-article chunk anchored to its topic, boosting in-scope Top-4 retrieval to **100.0% (8/8)** with a mean Top-1 cosine similarity of **`0.7256`**.

### Q3: Why `max_completion_tokens=1024`, `reasoning_effort="low"`, and `temperature=0.2`?
- `openai/gpt-oss-120b` uses internal reasoning tokens that count toward `max_completion_tokens`. Setting `reasoning_effort="low"` keeps reasoning concise so answers finish fast and never exhaust the output budget.
- `temperature=0.2` minimizes creative drift so exact percentages (e.g., GLDM `0.10%`, AZ-Gold `4.9% / 3.3% / 1.7% / 0%`, Egypt 21K pound `8g / 7g pure`) remain faithful to the retrieved articles.

### Q4: How do you prove RAG helps? (`eval/evaluation_report.md`)
Across our 10-question benchmark (6 fact + 2 multi-doc profile + 2 out-of-scope):
- **Key-Fact / Refusal Recall:** **98.0%** (With RAG) vs. **30.9%** (Without RAG)
- **Correctness (LLM Judge):** **5.00 / 5** (With RAG) vs. **1.90 / 5** (Without RAG)
- **Out-of-Scope Safe Refusal Rate:** **100.0%** (With RAG) vs. **0.0%** (Without RAG)
- **Hallucination / Boundary-Violation Rate:** **0.0%** (With RAG) vs. **20.0%** (Without RAG)

---

## 3. Recommended 3-Minute Live Demo Script

1. **Knowledge Base & Pure-NumPy Debugger (45s):** Open Tab 3 (**Gold Knowledge Base & NumPy Search**) to show all 20 `.txt` articles across 5 categories, then run an Egypt query (*"24K bullion bars vs 21K gold pounds masna'eya cashback"*) to show live cosine scores (`~0.75`) without calling an LLM.
2. **Profile-Tailored Chat (45s):** Switch to Tab 1 (**AurumAI Advisor**) and click the Egyptian student sample question (*EGP 2,000/month for 4 years*). Highlight how AurumAI recommends monthly DCA into AZ-Gold, warns against 1g bar *masna'eya* and Year-1 redemption fees, cites exact `.txt` filenames, and includes the educational disclaimer.
3. **Side-by-Side RAG vs. No-RAG + Out-of-Scope Refusal (60s):** Switch to Tab 2 (**Live RAG vs. No-RAG**) and test **Q6 (AZ-Gold)** or **Q9 (Ethereum/Bitcoin DeFi)** to demonstrate factual accuracy on Egypt rules and strict out-of-scope refusal vs. the ungrounded baseline.
4. **Evaluation Dashboard (30s):** Open Tab 4 (**Evaluation Results**) to present the 10-question benchmark (`98.0%` vs `30.9%` recall; `5.00/5` vs `1.90/5` correctness; `100%` vs `0%` out-of-scope refusal).
