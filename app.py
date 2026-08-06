"""
Autonomous Data Science Co-Pilot — Streamlit Web App
------------------------------------------------------
Upload a CSV / Excel / JSON file, ask a question in plain English, and this
app will autonomously write, execute, and self-correct Python/Pandas code to
answer it — producing a chart and a plain-English insight.

This file is a standalone implementation (independent from the Colab
notebook) so the deployed web app has no dependency on the research notebook.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import io
import contextlib
import traceback

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Data Science Copilot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main {
    padding-top: 1rem;
}
.block-container {
    padding-top: 2rem;
}
h1 {
    color:#2563eb;
}
div.stButton > button {
    width:100%;
    border-radius:12px;
    height:3rem;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

MAX_SELF_HEAL_RETRIES = 4
# "openrouter/free" auto-routes to whatever free model is currently available on OpenRouter —
# more robust than hardcoding a model ID since the free lineup rotates. To pin a specific
# free model instead, use e.g. "meta-llama/llama-3.3-70b-instruct:free" or "qwen/qwen3-coder:free".
LLM_MODEL = "openrouter/free"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
TRUSTED_DOC_SITES = ["pandas.pydata.org", "docs.python.org"]

SYSTEM_PROMPT = """You are a senior data analyst AI. You write short, correct, self-contained Python code
that analyzes a pandas DataFrame called `df` which is already loaded in memory.

Rules:
- Only use pandas, numpy, matplotlib.pyplot (as plt), and seaborn (as sns).
- Never read/write files and never re-load `df` — it already exists.
- Never use confidence intervals or error bars in charts.

- Always disable seaborn error bars (errorbar=None or ci=None).
- Create clean business-style charts with a white background.
- If you produce a chart, create it on a variable named `fig` (e.g. fig, ax = plt.subplots()) and do NOT call plt.show().
- Store any text findings in a variable named `insight_text` (a plain-English string).
- Return ONLY a Python code block — no explanations outside the code."""


# --------------------------------------------------------------------------
# Data loading & profiling
# --------------------------------------------------------------------------
def load_any_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    elif name.endswith(".json"):
        return pd.read_json(uploaded_file)
    raise ValueError("Unsupported file type. Please upload CSV, Excel, or JSON.")


def profile_dataframe(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.info(buf=buf)
    return f"""Shape: {df.shape[0]} rows x {df.shape[1]} columns

Columns and dtypes:
{buf.getvalue()}

Missing values per column:
{df.isnull().sum().to_string()}

First 5 rows:
{df.head().to_string()}"""


# --------------------------------------------------------------------------
# Prompting
# --------------------------------------------------------------------------
def build_generation_prompt(question: str, data_profile: str) -> str:
    return f"""Dataset profile:
{data_profile}

User question: "{question}"

Write Python code that answers this using the `df` DataFrame already in memory.
Produce a matplotlib/seaborn chart in `fig` where relevant, and always set `insight_text`
to a 2-4 sentence plain-English summary of what the data shows."""


def build_fix_prompt(question, data_profile, previous_code, error_message, doc_context) -> str:
    return f"""The following code failed.

Dataset profile:
{data_profile}

User question: "{question}"

Previous code:
```python
{previous_code}
```

Error raised:
{error_message}

Relevant excerpt from official Python/Pandas documentation (use this to fix the bug):
{doc_context}

Rewrite the FULL corrected code from scratch (same rules: use existing `df`, put chart in `fig`,
put summary text in `insight_text`). Return ONLY the corrected Python code."""


def extract_code_block(text: str) -> str:
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            if p.strip().startswith("python"):
                return p.strip()[len("python"):].strip()
        return parts[1].strip() if len(parts) > 1 else text
    return text


def call_llm_for_code(client: OpenAI, prompt: str) -> str:
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return extract_code_block(response.choices[0].message.content)


# --------------------------------------------------------------------------
# Sandbox execution
# --------------------------------------------------------------------------
ALLOWED_IMPORT_ROOTS = {"pandas", "numpy", "matplotlib", "seaborn"}


def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    """The LLM's generated code often re-imports pandas/numpy/matplotlib/seaborn even though
    they're already injected into the namespace. Python's `import` statement always calls
    __builtins__['__import__'], so without this, EVERY generated import statement raises
    ImportError: __import__ not found — before the actual analysis code ever runs.
    This restricted version allows only the four safe data-science libraries."""
    root = name.split(".")[0]
    if root not in ALLOWED_IMPORT_ROOTS:
        raise ImportError(f"Import of '{name}' is not permitted inside the sandbox.")
    return __import__(name, globals, locals, fromlist, level)


SAFE_BUILTINS = {
    "range": range, "len": len, "min": min, "max": max, "sum": sum, "sorted": sorted,
    "list": list, "dict": dict, "set": set, "tuple": tuple, "str": str, "int": int,
    "float": float, "bool": bool, "enumerate": enumerate, "zip": zip, "round": round,
    "print": print, "abs": abs, "__import__": _restricted_import,
}


def safe_exec(code_str: str, df: pd.DataFrame):
    local_ns = {"df": df, "pd": pd, "np": np, "plt": plt, "sns": sns}
    global_ns = {"__builtins__": SAFE_BUILTINS}
    stdout_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code_str, global_ns, local_ns)
            for ax in plt.gcf().axes:
                for patch in ax.patches:
                    patch.set_facecolor("#4472C4")
                    patch.set_alpha(0.95)
            
        return True, {
            "fig": local_ns.get("fig"),
            "insight_text": local_ns.get("insight_text", ""),
            "stdout": stdout_capture.getvalue(),
        }, ""
    except Exception:
        return False, {}, traceback.format_exc(limit=3)


# --------------------------------------------------------------------------
# RAG self-healing: live web search over official docs
# --------------------------------------------------------------------------
def web_search_docs(query: str, max_results: int = 5):
    resp = requests.get(
        "https://html.duckduckgo.com/html/",
        params={"q": query},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=10,
    )
    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    for a in soup.select("a.result__a"):
        href = a.get("href", "")
        if any(site in href for site in TRUSTED_DOC_SITES):
            links.append(href)
        if len(links) >= max_results:
            break
    return links


def fetch_page_text(url: str) -> str:
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text(separator=" ", strip=True)
    except Exception:
        return ""


def chunk_text(text: str, chunk_size: int = 400):
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]


def rerank_chunks(error_message: str, chunks: list, top_k: int = 2) -> str:
    if not chunks:
        return ""
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(chunks + [error_message])
    sims = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    top_idx = sims.argsort()[::-1][:top_k]
    return "\n---\n".join(chunks[i] for i in top_idx)


def fetch_relevant_docs(error_message: str) -> str:
    query = error_message.strip().splitlines()[-1] if error_message.strip() else "pandas error"
    query = f"{query} pandas python site:pandas.pydata.org OR site:docs.python.org"
    urls = web_search_docs(query)
    all_chunks = []
    for url in urls[:3]:
        all_chunks.extend(chunk_text(fetch_page_text(url)))
    return rerank_chunks(error_message, all_chunks)

# --------------------------------------------------------------------------
# Self-healing agent loop
# --------------------------------------------------------------------------
def autonomous_analyze(client, df, question, data_profile, max_retries=MAX_SELF_HEAL_RETRIES):
    heal_log = []
    code_str = call_llm_for_code(client, build_generation_prompt(question, data_profile))

    for attempt in range(1, max_retries + 1):
        success, result, error = safe_exec(code_str, df)
        if success:
            heal_log.append(f"Attempt {attempt}: succeeded")
            return {"success": True, "code": code_str, "result": result, "attempts": attempt, "heal_log": heal_log}

        heal_log.append(f"Attempt {attempt}: failed — {error.strip().splitlines()[-1]}")
        doc_context = fetch_relevant_docs(error)
        heal_log.append(f"   RAG fetched {len(doc_context)} chars of doc context to self-correct")
        code_str = call_llm_for_code(client, build_fix_prompt(question, data_profile, code_str, error, doc_context))

    return {"success": False, "code": code_str, "result": {}, "attempts": max_retries, "heal_log": heal_log}


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.markdown("""
<div style="padding:10px 0 0 0;">

<div style="
display:flex;
<div style="
display:flex;
justify-content:center;
align-items:center;
gap:20px;
margin-bottom:18px;
">

<div style="
font-size:58px;
line-height:1;
display:flex;
align-items:center;
justify-content:center;
margin-top:-6px;
">
📊
</div>

<h1 style="
margin:0;
font-size:64px;
font-weight:900;
letter-spacing:-2px;
line-height:1.15;
text-shadow:0 2px 8px rgba(37,99,235,.08);
background: linear-gradient(90deg,#0F172A 0%,#1D4ED8 60%,#4F46E5 100%);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
">
Autonomous Data Science Co-Pilot
</h1>

</div>

</div>

<p style="
margin-top:0px;
justify-content:center;
margin-bottom:10px;
font-size:20px;
font-weight:500;
color:#475569;
line-height:1.5;
max-width:850px;
margin:auto;
text-align:center;
">

Upload CSV, Excel or JSON datasets and instantly generate professional dashboards, business insights and AI-powered visualizations.

</p>

<div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:18px;">

<span style="background:#EFF6FF;color:#2563EB;padding:10px 18px;border-radius:999px;font-weight:700;">✨ AI Analytics</span>

<span style="background:#ECFDF5;color:#059669;padding:10px 18px;border-radius:999px;font-weight:700;">📊 Smart Charts</span>

<span style="background:#F5F3FF;color:#7C3AED;padding:10px 18px;border-radius:999px;font-weight:700;">🧠 Natural Language</span>

<span style="background:#FFF7ED;color:#EA580C;padding:10px 18px;border-radius:999px;font-weight:700;">⚡ Self-Healing Agent</span>

</div>

</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* ===========================
   Main Layout
=========================== */

.main{
    background:#FCFCFD;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
    max-width:1250px;
}

/* ===========================
   Headings
=========================== */

h1{
    color:#0F172A;
    font-weight:900;
    letter-spacing:-1.5px;
}

h2,h3{
    color:#1E293B;
    font-weight:700;
}

/* ===========================
   Analyze Button
=========================== */

.stButton > button{
    background:#2563EB;
    color:white;
    border:none;
    border-radius:14px;
    height:52px;
    font-weight:700;
    font-size:16px;
    transition:all .25s ease;
    box-shadow:0 8px 20px rgba(37,99,235,.18);
}

.stButton > button:hover{
    background:#1D4ED8;
    transform:translateY(-3px) scale(1.02);
    box-shadow:0 12px 24px rgba(37,99,235,.28);
}

/* ===========================
   Premium Text Input
=========================== */

.stTextInput input{
    background:#FFFFFF !important;
    border:2px solid #CBD5E1 !important;
    color:#0F172A !important;
    border-radius:14px !important;
    padding:12px 16px !important;
    font-size:17px !important;
    font-weight:500 !important;
    transition:all .25s ease !important;
}

.stTextInput input:hover{
    border-color:#94A3B8 !important;
}

.stTextInput input:focus{
    border:2px solid #2563EB !important;
    box-shadow:0 0 0 4px rgba(37,99,235,.15) !important;
}

/* ===========================
   File Uploader
=========================== */

.stFileUploader{
    border:1px solid #E2E8F0;
    border-radius:18px;
    padding:14px;
    background:#FFFFFF;
    box-shadow:0 12px 35px rgba(37,99,235,.10);
    transition:.25s;
}

.stFileUploader:hover{
    box-shadow:0 18px 40px rgba(37,99,235,.16);
}

/* ===========================
   Metric Cards
=========================== */

[data-testid="stMetric"]{
    background:#FFFFFF;
    border:1px solid #EEF2F7;
    border-radius:16px;
    padding:16px;
    box-shadow:0 10px 30px rgba(15,23,42,.08);
    transition:all .25s ease;
}

[data-testid="stMetric"]:hover{
    transform:translateY(-4px);
    box-shadow:0 16px 35px rgba(37,99,235,.12);
}

/* ===========================
   Expander
=========================== */

.streamlit-expanderHeader{
    font-weight:600;
    border-radius:10px;
}

/* ===========================
   Success / Info Boxes
=========================== */

[data-testid="stAlert"]{
    border-radius:14px;
}

/* ===========================
   DataFrame
=========================== */

[data-testid="stDataFrame"]{
    border-radius:16px;
    overflow:hidden;
}

/* ===========================
   Scrollbar
=========================== */

::-webkit-scrollbar{
    width:10px;
}

::-webkit-scrollbar-track{
    background:#F1F5F9;
}

::-webkit-scrollbar-thumb{
    background:#CBD5E1;
    border-radius:20px;
}

::-webkit-scrollbar-thumb:hover{
    background:#94A3B8;
}

</style>
""", unsafe_allow_html=True)

def get_secret_api_key() -> str:
    """Reads OPENROUTER_API_KEY from Streamlit secrets if configured. Returns "" if not set —
    st.secrets raises if no secrets.toml exists at all, so this must be wrapped in try/except."""
    try:
        return st.secrets.get("OPENROUTER_API_KEY", "")
    except Exception:
        return ""

with st.sidebar:
    st.header("⚙️ Configuration")
    secret_key = get_secret_api_key()
    if secret_key:
        st.success("✅ Server API Connected")
        api_key = st.text_input(
            "🔑 OpenRouter API Key (Optional)", type="password",
            help="Leave empty to use the server API.",
        ) or secret_key
        st.info("✅ Connected to the server API. Personal API key is optional.")
    else:
        api_key = st.text_input("🔑 OpenRouter API Key (Optional)", type="password", help="Leave blank to use the configured server API key.")
        st.warning("⚠️ No server API key configured. Enter your personal OpenRouter API key to run the analysis.")
        
    st.markdown("__")
    st.caption("🔒 Your API key is never stored and is only used for the current session.")

uploaded_file = st.file_uploader("📂 Upload Dataset", type=["csv", "xlsx", "xls", "json"])

if uploaded_file:
    try:
        df = load_any_file(uploaded_file)
        st.success(f"Loaded **{uploaded_file.name}** — {df.shape[0]} rows x {df.shape[1]} columns")

        # ================= Dataset Overview =================

        st.markdown("### 📊 Dataset Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("📄 Rows", df.shape[0])

        with col2:
            st.metric("📑 Columns", df.shape[1])

        with col3:
            st.metric("❌ Missing Values", int(df.isnull().sum().sum()))

        with col4:
            memory = round(df.memory_usage(deep=True).sum()/1024,2)
            st.metric("💾 Size (KB)", memory)

        # ================= Preview =================

        with st.expander("Preview data"):
            st.dataframe(df.head(20))

    except Exception as e:
        st.error(f"Could not read file: {e}")
        df = None
else:
    df = None
st.markdown("## ✨ AI Data Assistant")
st.markdown("""
<div style="
color:#334155;
font-size:19px;
font-weight:600;
letter-spacing:0.2px;
line-height:1.55;
margin-top:-8px;
margin-bottom:12px;
">
Ask questions in natural language and instantly generate professional charts, business insights and automated analysis.
</div>
""", unsafe_allow_html=True)

question = st.text_input("", placeholder="Example: Show revenue by region | Compare monthly sales | Top 5 products | Forecast sales trend")
run_clicked = st.button("🚀 Run Analysis", type="primary", disabled=not (df is not None and question and api_key))
if not api_key and (uploaded_file or question):
    st.info("🔑 Enter your OpenRouter API key from the sidebar to start the analysis.")
if run_clicked:
    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
    with st.spinner("Agent is writing and testing code..."):
        data_profile = profile_dataframe(df)
        outcome = autonomous_analyze(client, df, question, data_profile)

    if outcome["success"]:
        result = outcome["result"]
        st.success("✅ Analysis Completed Successfully")
        st.subheader("📊 Analysis Result")
        if result.get("fig") is not None:
            result["fig"].set_size_inches(16, 6)
            result["fig"].tight_layout()
            
            for ax in result["fig"].axes:
                ax.set_facecolor("white")
                result["fig"].patch.set_facecolor("white")

                # Only light horizontal grid
                ax.grid(axis="y", color="#EAEAEA", linestyle="-", linewidth=0.8)

                 # Remove unnecessary borders
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                 # Soft border colors
                ax.spines["left"].set_color("#CCCCCC")
                ax.spines["bottom"].set_color("#CCCCCC")

                # Font styling
                ax.tick_params(labelsize=11)

                ax.title.set_fontsize(16)
                ax.title.set_fontweight("bold")

                ax.xaxis.label.set_fontsize(12)
                ax.yaxis.label.set_fontsize(12)   
                
            st.pyplot(result["fig"], use_container_width=True)
            img = io.BytesIO()
            result["fig"].savefig(img, format="png", dpi=300)
            img.seek(0)
        if result.get("insight_text"):
            st.markdown(f"**Insight:** {result['insight_text']}")
        if result.get("stdout"):
            st.text(result["stdout"])
    else:
        st.error(f"Could not produce a working result after {outcome['attempts']} attempts.")

    with st.expander("Self-healing log"):
        for line in outcome["heal_log"]:
            st.text(line)

    with st.expander("Generated code (transparency / audit)"):
        st.code(outcome["code"], language="python")
        st.markdown("---")

st.markdown("""
<hr style="margin-top:60px;margin-bottom:25px;border:0;border-top:1px solid #E5E7EB;">
<div style="text-align:center;color:#6B7280;padding-bottom:25px;">
<h2 style="color:#374151;font-weight:700;margin-bottom:8px;">Autonomous Data Science Co-Pilot</h2>
<p style="font-size:17px;margin-bottom:8px;">AI Powered • Smart Visualization • Self-Healing Agent</p>
<p style="font-size:15px;">Powered by Python • Pandas • Matplotlib • Streamlit • OpenRouter AI</p>
</div>
""", unsafe_allow_html=True)
