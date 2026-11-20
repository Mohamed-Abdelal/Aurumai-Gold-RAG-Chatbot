"""
AurumAI — Gold Investment Recommendation Assistant (Streamlit Web Interface)

Features:
  1. Chat Assistant with live RAG ON / RAG OFF toggle & filename-cited retrieved evidence
  2. Side-by-Side Comparison Arena (With RAG vs. Without RAG on the exact same question)
  3. Gold Knowledge Base Explorer + Pure-NumPy Retrieval Debugger (works without API key)
  4. Evaluation Dashboard displaying the 10-question AurumAI benchmark results
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.config import (
    AVAILABLE_GROQ_MODELS,
    CHUNKS_PATH,
    DEFAULT_GROQ_MODEL,
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    EMBEDDINGS_PATH,
    EVAL_QUESTIONS_PATH,
    EVAL_RESULTS_PATH,
)
from src.embeddings import EmbeddingModel
from src.llm import GroqLLMClient
from src.rag_pipeline import RAGPipeline
from src.retriever import NumPyVectorIndex, RetrievedChunk

load_dotenv()

st.set_page_config(
    page_title="AurumAI | Gold Investment Recommendation Assistant",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded",
)

QUICK_SAMPLE_QUESTIONS = [
    "I'm a university student in Egypt saving EGP 2,000/month for 4 years. Should I buy 1g bars or AZ-Gold?",
    "In Egypt, should I buy 24K bullion bars (sabeeka) or 21K gold pounds, and how does masna'eya cashback work?",
    "How does the 60/20/20 balanced portfolio split with gold work, and what allocation rule fits high-inflation markets?",
    "I need EGP 150,000 in 6 months for tuition. Should I park it in gold during 2026 pullbacks?",
]


@st.cache_resource(show_spinner="Loading AurumAI NumPy vector index and embedding model...")
def load_rag_resources() -> Optional[RAGPipeline]:
    """Load the pre-built pure-NumPy index and SentenceTransformer model once per session."""
    if not EMBEDDINGS_PATH.exists() or not CHUNKS_PATH.exists():
        return None
    index = NumPyVectorIndex.load(EMBEDDINGS_PATH, CHUNKS_PATH)
    embedder = EmbeddingModel(model_name=EMBEDDING_MODEL_NAME)
    return RAGPipeline(index=index, embedding_model=embedder, llm_client=None)


@st.cache_data
def load_eval_questions() -> List[Dict]:
    if not EVAL_QUESTIONS_PATH.exists():
        return []
    with EVAL_QUESTIONS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_eval_results() -> Optional[Dict]:
    if not EVAL_RESULTS_PATH.exists():
        return None
    with EVAL_RESULTS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_groq_api_key() -> tuple[str, str]:
    """
    Resolve GROQ_API_KEY in priority order:
      1. In-app browser session key (st.session_state["session_groq_api_key"])
      2. Streamlit Community Cloud Secrets (st.secrets["GROQ_API_KEY"])
      3. Local environment variable / .env (os.getenv("GROQ_API_KEY"))

    Returns (api_key, source_label). Never writes user keys to disk.
    """
    session_key = str(st.session_state.get("session_groq_api_key", "")).strip()
    if session_key:
        return session_key, "Session Input"
    try:
        secret_key = st.secrets.get("GROQ_API_KEY", "")
        if secret_key:
            return str(secret_key).strip(), "Streamlit Secrets"
    except Exception:
        pass
    env_key = os.getenv("GROQ_API_KEY", "").strip()
    if env_key:
        return env_key, "Local .env"
    return "", ""


def validate_groq_api_key(api_key: str) -> tuple[bool, str]:
    """Verify a Groq API key by listing available models (lightweight, zero-token check)."""
    if not api_key.strip():
        return False, "API key cannot be empty."
    try:
        client = GroqLLMClient(api_key=api_key.strip())
        client.client.models.list()
        return True, "Connected! Groq API key is valid."
    except Exception as exc:
        return False, f"Key validation failed: {exc}"


def render_inline_api_key_setup(widget_prefix: str) -> None:
    """Show an in-page API key entry box so live-demo visitors can activate chat immediately."""
    with st.container(border=True):
        st.markdown("#### 🔑 Activate Live Chat — Add Your Groq API Key")
        st.caption(
            "Enter a free Groq API key (`gsk_...`) from [console.groq.com/keys](https://console.groq.com/keys) "
            "to enable live answers in this browser session (stored only in memory, never saved to GitHub). "
            "For automatic zero-click live demos, set `GROQ_API_KEY` in **Streamlit Cloud → Settings → Secrets**."
        )
        with st.form(key=f"{widget_prefix}_api_key_form", clear_on_submit=True):
            c_in, c_btn = st.columns([4, 1])
            with c_in:
                entered_key = st.text_input(
                    "Groq API Key",
                    type="password",
                    placeholder="gsk_...",
                    label_visibility="collapsed",
                )
            with c_btn:
                submitted = st.form_submit_button("Activate Key", type="primary", width="stretch")

            if submitted:
                if not entered_key.strip():
                    st.warning("Please paste a valid `gsk_...` key first.")
                else:
                    st.session_state["session_groq_api_key"] = entered_key.strip()
                    st.rerun()


def render_retrieved_sources(retrieved_chunks: List[RetrievedChunk]) -> None:
    """Render retrieved chunks with NumPy cosine similarity scores and article filenames."""
    if not retrieved_chunks:
        st.caption("No retrieved chunks (either RAG is disabled or no chunks passed the similarity threshold).")
        return

    with st.expander(f"📚 Retrieved Evidence ({len(retrieved_chunks)} chunks via Pure-NumPy Cosine Similarity)", expanded=False):
        for item in retrieved_chunks:
            c = item.chunk
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.markdown(f"**[Source {item.rank}] {c.title}**")
                st.caption(
                    f"File `{c.filename}` · Category `{c.category}` · "
                    f"Date `{c.date}` · Chunk `{c.chunk_id}` ({c.word_count} words)"
                )
            with col_b:
                st.metric("Cosine Sim", f"{item.score:.4f}")
            st.progress(max(0.0, min(1.0, float(item.score))))
            st.markdown(f"> {c.text}")
            st.divider()


def main() -> None:
    st.title("🪙 AurumAI — Gold Investment Recommendation Assistant")
    st.caption(
        "Evidence-Grounded Gold Investing Copilot (Global & Egypt Markets) "
        "| Pure-NumPy Cosine Similarity + Title-Anchored Embeddings + Groq LLM"
    )

    pipeline = load_rag_resources()
    if pipeline is None:
        st.error(
            "Pre-built NumPy index not found (`index/embeddings.npz` & `index/chunks.json`). "
            "Run `python scripts/build_index.py` first."
        )
        st.stop()

    stats = pipeline.get_stats()
    docs = pipeline.index.get_unique_documents()
    categories = stats["categories"]
    eval_questions = load_eval_questions()

    # ── Sidebar Controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ AurumAI Settings")

        groq_api_key, key_source = resolve_groq_api_key()
        if groq_api_key:
            st.success(f"Groq API key active (`{key_source}`)", icon="✅")
        else:
            st.warning(
                "No Groq API key active yet. Add a free key below (or set `GROQ_API_KEY` in Streamlit Secrets).",
                icon="🔑",
            )

        with st.expander("🔑 Manage Groq API Key", expanded=not bool(groq_api_key)):
            st.caption(
                "Get a free key at [console.groq.com/keys](https://console.groq.com/keys). "
                "Keys entered here live only in your current browser session."
            )
            with st.form("sidebar_api_key_form", clear_on_submit=True):
                new_key = st.text_input(
                    "Enter / Override Groq API Key",
                    type="password",
                    placeholder="gsk_...",
                )
                save_clicked = st.form_submit_button("💾 Save Key for Session", width="stretch")
                if save_clicked:
                    if new_key.strip():
                        st.session_state["session_groq_api_key"] = new_key.strip()
                        st.rerun()
                    else:
                        st.warning("Paste a non-empty key before saving.")

            b1, b2 = st.columns(2)
            with b1:
                if st.button("🧪 Test Key", width="stretch", disabled=not bool(groq_api_key)):
                    with st.spinner("Testing Groq connection..."):
                        ok, msg = validate_groq_api_key(groq_api_key)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
            with b2:
                if st.button(
                    "🧹 Clear Key",
                    width="stretch",
                    disabled=not bool(st.session_state.get("session_groq_api_key")),
                ):
                    st.session_state.pop("session_groq_api_key", None)
                    st.rerun()

        st.divider()
        st.subheader("🤖 LLM Settings")
        default_idx = (
            AVAILABLE_GROQ_MODELS.index(DEFAULT_GROQ_MODEL)
            if DEFAULT_GROQ_MODEL in AVAILABLE_GROQ_MODELS
            else 0
        )
        model_choice = st.selectbox("Groq Model", options=AVAILABLE_GROQ_MODELS, index=default_idx)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=DEFAULT_TEMPERATURE, step=0.05)

        st.divider()
        st.subheader("🔎 NumPy RAG Retrieval")
        use_rag = st.toggle("Enable RAG (Retrieval-Augmented)", value=True)
        top_k = st.slider("Top-K Chunks", min_value=1, max_value=8, value=DEFAULT_TOP_K)
        min_sim = st.slider(
            "Min Cosine Similarity",
            min_value=0.0,
            max_value=0.80,
            value=DEFAULT_MIN_SIMILARITY,
            step=0.05,
        )

        category_options = ["All Categories"] + categories
        selected_category = st.selectbox("Filter by Category (Optional)", options=category_options)
        category_filter = None if selected_category == "All Categories" else selected_category

        doc_options = {"All Articles (Full Knowledge Base)": None}
        for d in docs:
            if category_filter and d["category"] != category_filter:
                continue
            label = f"{d['filename']} — {d['title'][:36]}..."
            doc_options[label] = d["filename"]
        selected_doc_label = st.selectbox("Scope to Article (Optional)", options=list(doc_options.keys()))
        doc_filter = doc_options[selected_doc_label]

        st.divider()
        st.subheader("📊 Knowledge Base Index")
        m1, m2 = st.columns(2)
        m1.metric("Articles", stats["num_documents"])
        m2.metric("Chunks", stats["num_chunks"])
        st.caption(f"Embedder: `{stats['embedding_model']}` ({stats['embedding_dim']}d)")
        st.caption("Chunking: `220 words` / `40 overlap` + `Title:` prefix")
        st.caption("Similarity: Pure NumPy `dot / (||q||·||d||)`")

        if st.button("🗑️ Clear Chat History", width="stretch"):
            st.session_state["messages"] = []
            st.rerun()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_chat, tab_compare, tab_kb, tab_eval = st.tabs(
        [
            "💬 AurumAI Advisor",
            "⚖️ Live RAG vs. No-RAG",
            "📚 Gold Knowledge Base & NumPy Search",
            "📈 Evaluation Results",
        ]
    )

    # ── TAB 1: Chat Assistant ─────────────────────────────────────────────────
    with tab_chat:
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        if not groq_api_key:
            render_inline_api_key_setup("chat")

        if not st.session_state["messages"]:
            st.markdown("#### 💡 Try Asking AurumAI")
            cols = st.columns(2)
            for idx, sq in enumerate(QUICK_SAMPLE_QUESTIONS):
                with cols[idx % 2]:
                    if st.button(f"🪙 {sq}", key=f"quick_q_{idx}", width="stretch"):
                        st.session_state["pending_prompt"] = sq
                        st.rerun()

        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    badge = "🟢 With RAG" if msg.get("mode") == "rag" else "🟠 Without RAG (Baseline)"
                    meta_line = f"{badge} · `{msg.get('model', '')}` · {msg.get('latency_ms', 0):.0f} ms"
                    st.caption(meta_line)
                st.markdown(msg["content"])
                if msg.get("sources"):
                    render_retrieved_sources(msg["sources"])

        typed_prompt = st.chat_input(
            "Ask about gold bullion, ETFs, 60/20/20 allocation, 2026 pullbacks, or Egypt 24K bars vs. 21K pounds & AZ-Gold..."
        )
        prompt = st.session_state.pop("pending_prompt", None) or typed_prompt

        if prompt:
            if not groq_api_key:
                st.error("Please enter a free Groq API key in the sidebar to generate LLM responses.")
            else:
                st.session_state["messages"].append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                history_for_llm = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state["messages"][:-1]
                ]

                with st.chat_message("assistant"):
                    with st.spinner(
                        "Retrieving Top-K gold articles with pure NumPy & querying Groq..."
                        if use_rag
                        else "Querying Groq without RAG..."
                    ):
                        try:
                            client = GroqLLMClient(api_key=groq_api_key)
                            if use_rag:
                                result = pipeline.answer_with_rag(
                                    question=prompt,
                                    top_k=top_k,
                                    min_similarity=min_sim,
                                    model=model_choice,
                                    temperature=temperature,
                                    chat_history=history_for_llm,
                                    doc_filter=doc_filter,
                                    category_filter=category_filter,
                                    llm_client=client,
                                )
                            else:
                                result = pipeline.answer_without_rag(
                                    question=prompt,
                                    model=model_choice,
                                    temperature=temperature,
                                    chat_history=history_for_llm,
                                    llm_client=client,
                                )

                            resp = result.llm_response
                            badge = "🟢 With RAG" if resp.mode == "rag" else "🟠 Without RAG (Baseline)"
                            st.caption(f"{badge} · `{resp.model}` · {resp.latency_ms:.0f} ms")
                            st.markdown(resp.answer)
                            render_retrieved_sources(result.retrieved_chunks)

                            st.session_state["messages"].append(
                                {
                                    "role": "assistant",
                                    "content": resp.answer,
                                    "mode": resp.mode,
                                    "model": resp.model,
                                    "latency_ms": resp.latency_ms,
                                    "sources": result.retrieved_chunks,
                                }
                            )
                        except Exception as exc:
                            st.error(f"Groq API error: {exc}")

    # ── TAB 2: Side-by-Side Live Comparison ───────────────────────────────────
    with tab_compare:
        st.subheader("⚖️ Live Side-by-Side: With RAG vs. Without RAG")
        st.write(
            "Compare AurumAI's grounded RAG recommendations against the ungrounded Groq baseline on "
            "gold investing facts, Egypt-specific rules, investor profiles, and out-of-scope refusals."
        )

        if not groq_api_key:
            render_inline_api_key_setup("compare")

        preset_options = ["(Custom question)"] + [
            f"Q{q['id']} [{q.get('question_type', 'fact')}]: {q['question']}" for q in eval_questions
        ]
        chosen_preset = st.selectbox("Pick an evaluation question or enter your own:", options=preset_options)
        default_text = "" if chosen_preset == "(Custom question)" else chosen_preset.split("]: ", 1)[1]

        compare_question = st.text_area("Question to compare:", value=default_text, height=90)
        run_cmp = st.button("🚀 Run Side-by-Side Comparison", type="primary")

        if run_cmp:
            if not compare_question.strip():
                st.warning("Please enter a question first.")
            elif not groq_api_key:
                st.error("Please enter a free Groq API key in the sidebar.")
            else:
                with st.spinner("Running both With-RAG and Without-RAG on Groq..."):
                    try:
                        client = GroqLLMClient(api_key=groq_api_key)
                        cmp = pipeline.compare(
                            question=compare_question.strip(),
                            top_k=top_k,
                            min_similarity=min_sim,
                            model=model_choice,
                            temperature=temperature,
                            doc_filter=doc_filter,
                            category_filter=category_filter,
                            llm_client=client,
                        )
                        left, right = st.columns(2)
                        with left:
                            r_rag = cmp["with_rag"]
                            st.markdown("### 🟢 With RAG (AurumAI Grounded)")
                            st.caption(
                                f"Model: `{r_rag.llm_response.model}` · "
                                f"Latency: {r_rag.llm_response.latency_ms:.0f} ms · "
                                f"Chunks: {len(r_rag.retrieved_chunks)}"
                            )
                            st.markdown(r_rag.llm_response.answer)
                            render_retrieved_sources(r_rag.retrieved_chunks)

                        with right:
                            r_no = cmp["without_rag"]
                            st.markdown("### 🟠 Without RAG (Base LLM)")
                            st.caption(
                                f"Model: `{r_no.llm_response.model}` · "
                                f"Latency: {r_no.llm_response.latency_ms:.0f} ms · "
                                f"Chunks: 0"
                            )
                            st.markdown(r_no.llm_response.answer)
                    except Exception as exc:
                        st.error(f"Comparison failed: {exc}")

    # ── TAB 3: Knowledge Base Explorer & NumPy Retrieval Debugger ─────────────
    with tab_kb:
        st.subheader("📚 AurumAI Gold Investment Knowledge Base")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Articles", stats["num_documents"])
        c2.metric("Total Chunks", stats["num_chunks"])
        c3.metric("Vector Dim", stats["embedding_dim"])
        c4.metric("Matrix Dtype", str(pipeline.index.embeddings.dtype))

        st.markdown("#### 🔍 Pure-NumPy Retrieval Debugger (No API Key Needed)")
        debug_query = st.text_input(
            "Test query embedding & pure-NumPy cosine similarity Top-K retrieval directly:",
            placeholder="e.g., What is the cashback rule for sealed 24K gold bars in Egypt?",
        )
        if debug_query.strip():
            hits = pipeline.retrieve(
                question=debug_query.strip(),
                top_k=top_k,
                min_similarity=0.0,
                doc_filter=doc_filter,
                category_filter=category_filter,
            )
            if hits:
                for item in hits:
                    c = item.chunk
                    st.markdown(
                        f"**#{item.rank} · Cosine Similarity: `{item.score:.4f}`** — "
                        f"**{c.title}** (`{c.filename}`, chunk `{c.chunk_index}`)"
                    )
                    st.progress(max(0.0, min(1.0, float(item.score))))
                    st.caption(c.text)
            else:
                st.info("No chunks matched.")

        st.divider()
        st.markdown("#### 📄 Indexed Articles Catalog")
        if docs:
            df = pd.DataFrame(
                [
                    {
                        "Filename": d["filename"],
                        "Category": d["category"],
                        "Title": d["title"],
                        "Date": d["date"],
                        "Source": d["source"],
                        "Chunks": d["chunk_count"],
                        "Words": d["total_words"],
                    }
                    for d in docs
                ]
            )
            st.dataframe(df, width="stretch", hide_index=True)

    # ── TAB 4: Evaluation Results ─────────────────────────────────────────────
    with tab_eval:
        st.subheader("📈 AurumAI Evaluation: With RAG vs. Without RAG (10 Questions)")
        eval_data = load_eval_results()
        if not eval_data:
            st.info(
                "No saved evaluation run found yet (`eval/results.json`). "
                "Once your `GROQ_API_KEY` is configured in `.env`, run:\n\n"
                "```bash\npython eval/run_evaluation.py --delay 5.0\n```"
            )
            if eval_questions:
                st.markdown(f"#### Prepared Benchmark Questions ({len(eval_questions)} Questions)")
                q_df = pd.DataFrame(
                    [
                        {
                            "ID": f"Q{q['id']}",
                            "Type": q.get("question_type", "fact"),
                            "Target Files": ", ".join(q.get("target_files", [])) or "(Out of Scope)",
                            "Question": q["question"],
                        }
                        for q in eval_questions
                    ]
                )
                st.dataframe(q_df, width="stretch", hide_index=True)
        else:
            s = eval_data["summary"]
            st.caption(
                f"Evaluated on `{eval_data['evaluated_at_utc']}` · Model: `{eval_data['model']}` · "
                f"Top-K: `{eval_data['top_k']}` · Questions: `{s['num_questions']}` "
                f"({s['num_in_scope']} in-scope + {s['num_out_of_scope']} out-of-scope)"
            )
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("In-Scope Retrieval Hit@K", f"{s['retrieval_hit_rate']*100:.1f}%")
            k2.metric(
                "Key-Fact / Refusal Recall (RAG)",
                f"{s['avg_fact_recall_rag']*100:.1f}%",
                delta=f"+{(s['avg_fact_recall_rag'] - s['avg_fact_recall_no_rag'])*100:.1f}% vs No-RAG",
            )
            if s.get("judge_enabled"):
                k3.metric(
                    "Correctness (RAG)",
                    f"{s['avg_correctness_rag']:.2f}/5",
                    delta=f"+{s['avg_correctness_rag'] - s['avg_correctness_no_rag']:.2f}",
                )
                k4.metric(
                    "Out-of-Scope Refusal (RAG)",
                    f"{s['oos_refusal_rate_rag']*100:.0f}%",
                    delta=f"vs {s['oos_refusal_rate_no_rag']*100:.0f}% No-RAG",
                )
            else:
                k3.metric("Mean Top-1 Cosine Sim", f"{s['avg_top1_similarity']:.4f}")
                k4.metric("Avg Latency (RAG)", f"{s['avg_latency_rag_ms']:.0f} ms")

            chart_df = pd.DataFrame(
                {
                    "Metric": ["Key-Fact / Refusal Recall (%)"],
                    "With RAG": [round(s["avg_fact_recall_rag"] * 100, 1)],
                    "Without RAG": [round(s["avg_fact_recall_no_rag"] * 100, 1)],
                }
            )
            if s.get("judge_enabled"):
                chart_df = pd.concat(
                    [
                        chart_df,
                        pd.DataFrame(
                            {
                                "Metric": ["Correctness (% of 5)", "Relevance (% of 5)"],
                                "With RAG": [
                                    round(s["avg_correctness_rag"] * 20, 1),
                                    round(s["avg_relevance_rag"] * 20, 1),
                                ],
                                "Without RAG": [
                                    round(s["avg_correctness_no_rag"] * 20, 1),
                                    round(s["avg_relevance_no_rag"] * 20, 1),
                                ],
                            }
                        ),
                    ],
                    ignore_index=True,
                )
            st.bar_chart(chart_df.set_index("Metric"))

            st.markdown("#### Detailed Per-Question Results")
            for r in eval_data["results"]:
                qtype = r.get("question_type", "fact")
                if qtype == "out_of_scope":
                    hit_badge = f"Refused: {'✅' if r['rag']['fact_recall'] >= 1.0 else '❌'}"
                else:
                    hit_badge = f"Hit@{eval_data['top_k']}: {'✅' if r['retrieval_hit'] else '❌'}"
                title_line = (
                    f"Q{r['id']} [{qtype}]: {r['question']}  |  {hit_badge}  |  "
                    f"Score: {r['rag']['fact_recall']*100:.0f}% vs {r['no_rag']['fact_recall']*100:.0f}%"
                )
                with st.expander(title_line):
                    targets = ", ".join(r.get("target_files", [])) or "None (Out of Scope)"
                    st.caption(f"Target Files: `{targets}` · Top-1 Cosine Sim: `{r['top1_similarity']:.4f}`")
                    st.info(f"**Verified Reference:** {r['reference_answer']}")
                    c_left, c_right = st.columns(2)
                    with c_left:
                        st.markdown("**🟢 With RAG (AurumAI)**")
                        st.markdown(r["rag"]["answer"])
                    with c_right:
                        st.markdown("**🟠 Without RAG (Baseline)**")
                        st.markdown(r["no_rag"]["answer"])


if __name__ == "__main__":
    main()
