"""
AurumAI — Gold Investment Recommendation Assistant (Streamlit UI)

Simple, clean student RAG chatbot with 3 tabs:
  1. Chatbot (With RAG / Without RAG toggle + retrieved sources)
  2. Compare With RAG vs. Without RAG side by side
  3. Evaluation Results (10-question benchmark)
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.config import (
    CHUNKS_PATH,
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
    page_title="AurumAI | Gold Investment Chatbot",
    page_icon="🪙",
    layout="wide",
)

SAMPLE_QUESTIONS = [
    "I'm a student in Egypt saving EGP 2,000/month for 4 years. Should I buy 1g bars or AZ-Gold?",
    "In Egypt, should I buy 24K gold bars or 21K gold pounds, and how does masna'eya cashback work?",
    "How does the 60/20/20 portfolio split with gold work?",
    "I need EGP 150,000 in 6 months for tuition. Should I put it in gold?",
]


@st.cache_resource(show_spinner="Loading AurumAI knowledge base...")
def load_pipeline() -> Optional[RAGPipeline]:
    """Load the pre-built NumPy index and SentenceTransformer model once."""
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


def get_groq_api_key() -> tuple[str, str]:
    """
    Resolve GROQ_API_KEY in order:
      1. User-entered key in the current browser session
      2. Streamlit Community Cloud Secrets
      3. Local .env file
    """
    session_key = str(st.session_state.get("groq_api_key", "")).strip()
    if session_key:
        return session_key, "Session"
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


def render_api_key_prompt(prefix: str) -> None:
    """Clean inline box so live-demo users can paste a Groq API key if Secrets aren't set."""
    with st.container(border=True):
        st.markdown("#### 🔑 Enter a Free Groq API Key to Start Chatting")
        st.caption(
            "Get a free key at [console.groq.com/keys](https://console.groq.com/keys). "
            "Your key is kept only for this browser session."
        )
        with st.form(f"{prefix}_key_form", clear_on_submit=True):
            col1, col2 = st.columns([4, 1])
            with col1:
                entered = st.text_input(
                    "Groq API Key",
                    type="password",
                    placeholder="gsk_...",
                    label_visibility="collapsed",
                )
            with col2:
                submitted = st.form_submit_button("Activate", type="primary", width="stretch")
            if submitted and entered.strip():
                st.session_state["groq_api_key"] = entered.strip()
                st.rerun()


def render_sources(retrieved_chunks: List[RetrievedChunk]) -> None:
    """Show the retrieved knowledge-base chunks used to generate the answer."""
    if not retrieved_chunks:
        return
    with st.expander(f"📄 Sources Used ({len(retrieved_chunks)} retrieved chunks)", expanded=False):
        for item in retrieved_chunks:
            c = item.chunk
            st.markdown(f"**{item.rank}. {c.title}** (`{c.filename}`) — Similarity: `{item.score:.2f}`")
            st.caption(c.text)
            st.divider()


def main() -> None:
    st.title("🪙 AurumAI — Gold Investment Chatbot")
    st.caption(
        "RAG-powered Gold Investment Assistant (Global & Egypt Markets) built with Pure-NumPy Cosine Similarity and Groq LLM."
    )

    pipeline = load_pipeline()
    if pipeline is None:
        st.error("Vector index not found. Run `python scripts/build_index.py` first.")
        st.stop()

    stats = pipeline.get_stats()
    eval_questions = load_eval_questions()

    # ── Simple Sidebar ────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("🪙 AurumAI")
        st.write("Ask about physical gold bars, Gold ETFs, portfolio allocation, 2026 pullbacks, or buying gold in Egypt.")

        st.divider()
        groq_api_key, key_source = get_groq_api_key()
        if groq_api_key:
            st.success(f"API Key ready ({key_source})", icon="✅")
            if key_source == "Session" and st.button("Remove Session Key", width="stretch"):
                st.session_state.pop("groq_api_key", None)
                st.rerun()
        else:
            st.warning("No Groq API key set yet.", icon="🔑")
            with st.form("sidebar_key_form", clear_on_submit=True):
                key_in = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
                if st.form_submit_button("Save Key", width="stretch") and key_in.strip():
                    st.session_state["groq_api_key"] = key_in.strip()
                    st.rerun()

        st.divider()
        use_rag = st.toggle("Enable RAG (Use Knowledge Base)", value=True)
        st.caption(
            "Turn OFF to test the base LLM without retrieved gold articles."
        )

        st.divider()
        st.markdown("**Knowledge Base**")
        st.caption(f"• **{stats['num_documents']}** gold investment articles")
        st.caption(f"• **{stats['num_chunks']}** text chunks (pure-NumPy search)")

        if st.button("🗑️ Clear Chat", width="stretch"):
            st.session_state["messages"] = []
            st.rerun()

    # ── 3 Focused Tabs ────────────────────────────────────────────────────────
    tab_chat, tab_compare, tab_eval = st.tabs(
        [
            "💬 Chatbot",
            "⚖️ With RAG vs. Without RAG",
            "📊 Evaluation Results",
        ]
    )

    # ── TAB 1: Chatbot ────────────────────────────────────────────────────────
    with tab_chat:
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        if not groq_api_key:
            render_api_key_prompt("chat")

        if not st.session_state["messages"]:
            st.markdown("#### Try asking:")
            cols = st.columns(2)
            for idx, q in enumerate(SAMPLE_QUESTIONS):
                with cols[idx % 2]:
                    if st.button(q, key=f"sample_{idx}", width="stretch"):
                        st.session_state["pending_prompt"] = q
                        st.rerun()

        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant":
                    badge = "🟢 With RAG" if msg.get("mode") == "rag" else "🟠 Without RAG"
                    st.caption(badge)
                st.markdown(msg["content"])
                if msg.get("sources"):
                    render_sources(msg["sources"])

        typed_prompt = st.chat_input("Ask AurumAI a question about gold investing...")
        prompt = st.session_state.pop("pending_prompt", None) or typed_prompt

        if prompt:
            if not groq_api_key:
                st.error("Please enter a free Groq API key above to start chatting.")
            else:
                st.session_state["messages"].append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state["messages"][:-1]
                ]

                with st.chat_message("assistant"):
                    with st.spinner("Searching knowledge base and generating answer..." if use_rag else "Generating answer without RAG..."):
                        try:
                            client = GroqLLMClient(api_key=groq_api_key)
                            if use_rag:
                                result = pipeline.answer_with_rag(
                                    question=prompt,
                                    chat_history=history,
                                    llm_client=client,
                                )
                            else:
                                result = pipeline.answer_without_rag(
                                    question=prompt,
                                    chat_history=history,
                                    llm_client=client,
                                )

                            resp = result.llm_response
                            st.caption("🟢 With RAG" if resp.mode == "rag" else "🟠 Without RAG")
                            st.markdown(resp.answer)
                            render_sources(result.retrieved_chunks)

                            st.session_state["messages"].append(
                                {
                                    "role": "assistant",
                                    "content": resp.answer,
                                    "mode": resp.mode,
                                    "sources": result.retrieved_chunks,
                                }
                            )
                        except Exception as exc:
                            st.error(f"Error calling Groq API: {exc}")

    # ── TAB 2: Side-by-Side Comparison ────────────────────────────────────────
    with tab_compare:
        st.subheader("⚖️ Compare Answers: With RAG vs. Without RAG")
        st.write("Test the exact same question with and without the gold-investing knowledge base.")

        if not groq_api_key:
            render_api_key_prompt("compare")

        presets = ["(Type your own question)"] + [f"Q{q['id']}: {q['question']}" for q in eval_questions]
        chosen = st.selectbox("Choose a test question:", options=presets)
        default_q = "" if chosen == "(Type your own question)" else chosen.split(": ", 1)[1]

        question_input = st.text_area("Question:", value=default_q, height=85)
        if st.button("Compare Answers", type="primary"):
            if not question_input.strip():
                st.warning("Please enter a question.")
            elif not groq_api_key:
                st.error("Please enter a free Groq API key first.")
            else:
                with st.spinner("Generating With-RAG and Without-RAG answers..."):
                    try:
                        client = GroqLLMClient(api_key=groq_api_key)
                        cmp = pipeline.compare(question=question_input.strip(), llm_client=client)

                        left, right = st.columns(2)
                        with left:
                            st.markdown("### 🟢 With RAG (AurumAI)")
                            st.markdown(cmp["with_rag"].llm_response.answer)
                            render_sources(cmp["with_rag"].retrieved_chunks)
                        with right:
                            st.markdown("### 🟠 Without RAG (Base LLM)")
                            st.markdown(cmp["without_rag"].llm_response.answer)
                    except Exception as exc:
                        st.error(f"Comparison failed: {exc}")

    # ── TAB 3: Evaluation Results ─────────────────────────────────────────────
    with tab_eval:
        st.subheader("📊 10-Question Evaluation: With RAG vs. Without RAG")
        eval_data = load_eval_results()

        if not eval_data:
            st.info("Run `python eval/run_evaluation.py` to generate evaluation results.")
        else:
            s = eval_data["summary"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Retrieval Hit@4", f"{s['retrieval_hit_rate']*100:.0f}%")
            c2.metric("Key-Fact Recall (RAG)", f"{s['avg_fact_recall_rag']*100:.1f}%", f"vs {s['avg_fact_recall_no_rag']*100:.1f}% No-RAG")
            c3.metric("Correctness (RAG)", f"{s['avg_correctness_rag']:.2f} / 5", f"vs {s['avg_correctness_no_rag']:.2f} / 5 No-RAG")
            c4.metric("Out-of-Scope Refusal", f"{s['oos_refusal_rate_rag']*100:.0f}%", f"vs {s['oos_refusal_rate_no_rag']*100:.0f}% No-RAG")

            summary_df = pd.DataFrame(
                [
                    {
                        "Q#": f"Q{r['id']}",
                        "Category": r["question_type"],
                        "Question": r["question"],
                        "With RAG Recall": f"{r['rag']['fact_recall']*100:.0f}%",
                        "Without RAG Recall": f"{r['no_rag']['fact_recall']*100:.0f}%",
                        "Correctness (RAG vs No-RAG)": f"{r['rag']['judge']['correctness']}/5 vs {r['no_rag']['judge']['correctness']}/5",
                    }
                    for r in eval_data["results"]
                ]
            )
            st.dataframe(summary_df, width="stretch", hide_index=True)

            st.markdown("#### Detailed Responses")
            for r in eval_data["results"]:
                with st.expander(f"Q{r['id']}: {r['question']}"):
                    col_l, col_r = st.columns(2)
                    with col_l:
                        st.markdown("**🟢 With RAG**")
                        st.markdown(r["rag"]["answer"])
                    with col_r:
                        st.markdown("**🟠 Without RAG**")
                        st.markdown(r["no_rag"]["answer"])

        with st.expander(f"📚 View Knowledge Base Articles ({stats['num_documents']} files)"):
            docs_df = pd.DataFrame(pipeline.index.get_unique_documents())
            st.dataframe(docs_df, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
