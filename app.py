"""
AI Research Assistant — Streamlit Web App
------------------------------------------------
A web front-end for a four-agent pipeline: a question goes in, and
Research -> Summarizer -> Fact Checker -> Report Generator stages run
in sequence to produce a structured report.

BEFORE YOU RUN THIS:
  1. Install dependencies:
       pip install -r requirements.txt
  2. Get an API key — two options:
       FREE:  Groq (console.groq.com) — no credit card required.
              Create a ".env" file in this folder with:
                GROQ_API_KEY=your_key_here
       PAID:  OpenAI (platform.openai.com/api-keys) — pay-per-use.
              Create a ".env" file in this folder with:
                OPENAI_API_KEY=your_key_here
     If both are present, GROQ_API_KEY is used.
  3. Run:
       streamlit run app.py

COST NOTE: with a Groq key this app is free to run (subject to Groq's
free-tier rate limits). With an OpenAI key, each report makes four
separate paid API calls (one per agent).
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Research Assistant", page_icon="📚", layout="wide")


# ---------------------------------------------------------------------------
# Styling — editorial / academic theme
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;500;600;700&family=Source+Sans+3:wght@400;500;600&display=swap');

    .stApp { background-color: #FAF7F0; }
    * { font-family: 'Source Sans 3', sans-serif; }

    section[data-testid="stSidebar"] {
        background-color: #F3EEE3;
        border-right: 1px solid #E0D8C4;
    }
    section[data-testid="stSidebar"] * { color: #3A3226 !important; }

    h1 {
        font-family: 'Lora', serif !important;
        color: #6B2737 !important;
        font-weight: 700 !important;
    }
    h2, h3 {
        font-family: 'Lora', serif !important;
        color: #3A3226 !important;
    }
    .stApp p, .stApp span, .stApp label, .stApp div { color: #3A3226; }

    div.stButton > button {
        background-color: #6B2737;
        color: #FAF7F0;
        font-weight: 600;
        border: none;
        border-radius: 4px;
        padding: 0.55em 1.4em;
    }
    div.stButton > button:hover { background-color: #85333F; color: #FAF7F0; }

    .pipeline-step {
        border-left: 3px solid #D9CBAE;
        padding: 6px 0 6px 14px;
        margin-bottom: 4px;
        font-size: 13.5px;
        color: #6B6350;
    }
    .pipeline-step-num {
        display: inline-block;
        width: 18px; height: 18px;
        border-radius: 50%;
        background: #6B2737;
        color: #FAF7F0;
        font-size: 11px;
        font-weight: 700;
        text-align: center;
        line-height: 18px;
        margin-right: 8px;
    }

    .report-box {
        background-color: #FFFFFF;
        border: 1px solid #E0D8C4;
        border-radius: 8px;
        padding: 28px 32px;
    }

    .status-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 12.5px;
        font-weight: 600;
    }
    .status-ok { background-color: #E7EFE1; color: #3E6B2E; }
    .status-missing { background-color: #F7E3E1; color: #A13B2C; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Agent pipeline
# ---------------------------------------------------------------------------
# Each "agent" below is just a name + a system prompt. Rather than relying on
# automatic tool-calling handoffs (which several free/open models handle
# inconsistently), this pipeline calls each stage directly and passes its
# output into the next. Same four-agent structure, same behavior, just
# driven explicitly.

AGENTS = [
    (
        "Research Agent",
        "You are a meticulous researcher. Given the user's question, gather "
        "key facts, relevant context, and important details needed to answer "
        "it thoroughly. Be specific and organized.",
    ),
    (
        "Summarizer Agent",
        "You specialize in distilling research into clear, concise summaries. "
        "Take the research findings provided and condense them into 5-8 clear "
        "bullet points, organized by sub-topic.",
    ),
    (
        "Fact Checker Agent",
        "You are a skeptical editor. Review the summary provided against the "
        "original research also provided. Flag anything unsupported, "
        "contradictory, or exaggerated, and correct it if needed. Output the "
        "corrected, verified summary.",
    ),
    (
        "Report Generator Agent",
        "You are a professional technical writer. Using the fact-checked "
        "summary provided, write a final structured report in Markdown with: "
        "1) A short executive summary (2-3 sentences), 2) Key findings under "
        "clear headers, 3) A short conclusion.",
    ),
]


def get_client_and_model():
    from openai import OpenAI

    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if groq_key:
        return OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1"), "openai/gpt-oss-120b"
    elif openai_key:
        return OpenAI(api_key=openai_key), "gpt-4o-mini"
    else:
        raise RuntimeError("No API key found. Add GROQ_API_KEY or OPENAI_API_KEY to your .env file.")


def run_agent_stage(client, model, system_prompt, user_content):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    return response.choices[0].message.content


def run_research_assistant(user_question: str) -> str:
    client, model = get_client_and_model()

    # Stage 1: Research
    _, research_prompt = AGENTS[0]
    research_output = run_agent_stage(client, model, research_prompt, user_question)

    # Stage 2: Summarize
    _, summarizer_prompt = AGENTS[1]
    summary_output = run_agent_stage(
        client, model, summarizer_prompt,
        f"Original question: {user_question}\n\nResearch findings:\n{research_output}"
    )

    # Stage 3: Fact-check
    _, fact_checker_prompt = AGENTS[2]
    verified_summary = run_agent_stage(
        client, model, fact_checker_prompt,
        f"Original research:\n{research_output}\n\nSummary to review:\n{summary_output}"
    )

    # Stage 4: Report
    _, report_prompt = AGENTS[3]
    final_report = run_agent_stage(
        client, model, report_prompt,
        f"Original question: {user_question}\n\nFact-checked summary:\n{verified_summary}"
    )

    return final_report


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

groq_key_present = bool(os.getenv("GROQ_API_KEY"))
openai_key_present = bool(os.getenv("OPENAI_API_KEY"))
api_key_present = groq_key_present or openai_key_present

if groq_key_present:
    active_provider = "Groq — GPT-OSS 120B (free)"
elif openai_key_present:
    active_provider = "OpenAI — gpt-4o-mini (paid)"
else:
    active_provider = None

with st.sidebar:
    st.markdown("### API Status")
    if api_key_present:
        st.markdown('<span class="status-pill status-ok">● Key loaded</span>', unsafe_allow_html=True)
        st.caption(f"Provider: {active_provider}")
    else:
        st.markdown('<span class="status-pill status-missing">○ No key found</span>', unsafe_allow_html=True)
        st.caption("Add GROQ_API_KEY (free) or OPENAI_API_KEY (paid) to a .env file in this folder.")

    st.markdown("---")
    st.markdown("### Agent Pipeline")
    steps = [
        ("1", "Research Agent", "gathers facts and context"),
        ("2", "Summarizer Agent", "condenses into key bullet points"),
        ("3", "Fact Checker Agent", "flags unsupported claims"),
        ("4", "Report Generator", "writes the final structured report"),
    ]
    for num, name, desc in steps:
        st.markdown(
            f'<div class="pipeline-step"><span class="pipeline-step-num">{num}</span>'
            f'<strong>{name}</strong><br><span style="margin-left:26px;">{desc}</span></div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("### Model Info")
    if api_key_present:
        st.markdown(f"**Provider:** {active_provider}")
    else:
        st.markdown("**Provider:** not connected")
    st.markdown("**Method:** 4-stage sequential pipeline")
    st.caption("Each report makes 4 API calls (one per agent). Free with a Groq key, or uses paid OpenAI credits.")


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

st.title("📚 AI Research Assistant")
st.caption("Ask a question — four agents research, summarize, fact-check, and write a structured report.")

st.markdown("---")

question = st.text_area(
    "What would you like researched?",
    placeholder="e.g. What are the main approaches to carbon capture technology today?",
    height=110,
)

generate_clicked = st.button("🔎 Generate Report", type="primary", disabled=not api_key_present)

if not api_key_present:
    st.info("Add a free **GROQ_API_KEY** (or a paid **OPENAI_API_KEY**) to a `.env` file to enable report generation.")

if generate_clicked:
    if not question.strip():
        st.warning("Enter a question first — there's nothing to research yet.")
    else:
        with st.spinner("Agents are researching, summarizing, fact-checking, and drafting your report..."):
            try:
                report = run_research_assistant(question.strip())
                st.session_state["last_report"] = report
                st.session_state["last_question"] = question.strip()
            except Exception as e:
                st.error(f"Something went wrong while generating the report: {e}")

if "last_report" in st.session_state:
    st.markdown("---")
    st.markdown(f"### Report: *{st.session_state['last_question']}*")
    st.markdown(f'<div class="report-box">{st.session_state["last_report"]}</div>', unsafe_allow_html=True)

    st.download_button(
        "⬇ Download report as Markdown",
        data=st.session_state["last_report"],
        file_name="research_report.md",
        mime="text/markdown",
    )