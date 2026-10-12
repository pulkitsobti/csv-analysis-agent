"""Streamlit web app for the CSV analysis agent."""

import io
import json
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent import MODELS, DataAgent

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DATASETS = {
    "Students (6 rows)": os.path.join(BASE_DIR, "sample_data", "students.csv"),
    "Stationery sales (150 rows)": os.path.join(BASE_DIR, "sample_data", "sales.csv"),
}

st.set_page_config(page_title="CSV Analysis Agent", page_icon="📊", layout="wide")


@st.cache_data
def read_csv(data: bytes) -> pd.DataFrame:
    """Read CSV bytes, falling back to Latin-1 for files that aren't UTF-8."""
    try:
        return pd.read_csv(io.BytesIO(data))
    except UnicodeDecodeError:
        return pd.read_csv(io.BytesIO(data), encoding="latin-1")


def show_message(message):
    """Draw one chat message, plus the tool calls behind an answer."""
    with st.chat_message(message["role"]):
        # "$" would otherwise be read as the start of a maths formula.
        st.markdown(message["content"].replace("$", "\\$"))

        if message["role"] != "assistant":
            return
        steps = message.get("steps", [])
        if message.get("model"):
            st.caption(f"Answered by {message['model']}")
        if not steps:
            st.caption("⚠️ No tools were used for this answer.")
            return

        with st.expander(f"🔧 {len(steps)} tool call(s): how the agent got this answer"):
            for number, step in enumerate(steps, start=1):
                st.markdown(f"**{number}. `{step['tool']}`**")
                st.code(json.dumps(step["args"], indent=2), language="json")
                result = step["result"] or {}
                if "error" in result:
                    st.error(f"Tool error (sent back to the agent): {result['error']}")
                else:
                    text = json.dumps(result.get("result"), indent=2, default=str)
                    if len(text) > 3000:
                        text = text[:3000] + "\n... (truncated)"
                    st.code(text, language="json")


st.title("📊 CSV Analysis Agent")
st.write(
    "Upload a CSV and ask questions in plain English. The agent answers by "
    "calling real pandas tools, and every step it took is shown under its answer."
)

with st.sidebar:
    st.header("⚙️ Settings")
    user_key = st.text_input(
        "Your Gemini API key (optional)",
        type="password",
        help="Leave empty to use the app's key. Get a free key at aistudio.google.com.",
    )
    st.caption(f"Models (tried in order): {', '.join(MODELS)}")

# ---------- choose a dataset ----------

data, dataset_name = None, None
source = st.radio("Data source", ["Upload a CSV", "Use a sample dataset"], horizontal=True)
if source == "Upload a CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        data, dataset_name = uploaded_file.getvalue(), uploaded_file.name
else:
    choice = st.selectbox("Sample dataset", list(SAMPLE_DATASETS))
    with open(SAMPLE_DATASETS[choice], "rb") as file:
        data, dataset_name = file.read(), choice

if data is None:
    st.info("Upload a CSV file (or pick a sample dataset) to get started.")
    st.stop()

try:
    df = read_csv(data)
except Exception as error:
    st.error(f"Could not read the CSV file: {error}")
    st.stop()

# Start a fresh chat whenever a different dataset is loaded.
dataset_key = f"{dataset_name}-{len(data)}"
if st.session_state.get("dataset_key") != dataset_key:
    st.session_state.dataset_key = dataset_key
    st.session_state.messages = []

with st.expander(f"Preview: {dataset_name}  ({len(df)} rows × {len(df.columns)} columns)"):
    st.dataframe(df.head(20), width="stretch")

# ---------- chat ----------

for message in st.session_state.messages:
    show_message(message)

question = st.chat_input("Ask a question, e.g. Which group has the highest average?")
if question:
    api_key = user_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("No Gemini API key found. Add GEMINI_API_KEY to your .env file, or paste a key in the sidebar.")
        st.stop()

    history = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    show_message(st.session_state.messages[-1])

    with st.spinner("Analysing..."):
        try:
            result = DataAgent(df, api_key).ask(question, history)
        except Exception as error:
            result = {"answer": f"❌ The request failed: {error}", "steps": [], "model": None}

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "steps": result["steps"],
        "model": result["model"],
    })
    st.rerun()
