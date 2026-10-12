"""Streamlit web app for the CSV analysis agent."""

import io
import json
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google.genai import errors

from agent import MODELS, DataAgent
from memory import KINDS, MemoryStore

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEMORY_PATH = os.path.join(BASE_DIR, "memory.json")
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


def memory_sidebar(memory, columns):
    """Let the user see, add and delete what the agent remembers."""
    with st.sidebar:
        st.header("🧠 Memory")
        st.caption(
            "Facts the agent keeps between sessions (saved in memory.json). "
            "Ask it to remember or forget something, or manage them here."
        )

        active = memory.relevant(columns)
        if not active:
            st.info(
                "No memories for this dataset yet. Try asking: "
                "\"Remember that 'good attendance' means Attendance >= 90.\""
            )
        for item in active:
            with st.container(border=True):
                st.markdown(f"**{item['name']}** · `{item['kind']}` · {item['scope']}")
                st.write(item["content"])
                note = f"Updated {item['updated_at'][:10]}"
                if item["history"]:
                    note += f" · replaced {len(item['history'])} older version(s)"
                if memory.is_stale(item):
                    note += " · ⚠️ may be stale"
                st.caption(note)
                if item["history"]:
                    with st.expander("Older versions"):
                        for old in reversed(item["history"]):
                            st.caption(f"{old['replaced_at'][:10]}: {old['content']}")
                if st.button("Delete", key=f"delete-{item['scope_id']}-{item['name']}"):
                    memory.delete(item["name"], item["scope_id"])
                    st.rerun()

        hidden = len(memory.memories) - len(active)
        if hidden > 0:
            st.caption(f"{hidden} more memory/memories belong to other datasets.")

        with st.expander("➕ Add a memory"):
            with st.form("add-memory", clear_on_submit=True):
                name = st.text_input("Name", placeholder="good_attendance")
                kind = st.selectbox("Kind", KINDS)
                content = st.text_input("Fact", placeholder="Attendance >= 90")
                scopes = ["dataset", "global"] if columns else ["global"]
                scope = st.radio("Applies to", scopes, horizontal=True)
                if st.form_submit_button("Save"):
                    try:
                        memory.save(name, kind, content, scope, columns)
                        st.rerun()
                    except ValueError as error:
                        st.error(str(error))

        if memory.memories:
            with st.expander("🗑️ Clear all memory"):
                sure = st.checkbox("Yes, delete every saved memory")
                if st.button("Clear all", disabled=not sure):
                    memory.clear()
                    st.rerun()


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

memory = MemoryStore(MEMORY_PATH)

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
    memory_sidebar(memory, [])
    st.info("Upload a CSV file (or pick a sample dataset) to get started.")
    st.stop()

try:
    df = read_csv(data)
except Exception as error:
    st.error(f"Could not read the CSV file: {error}")
    st.stop()

memory_sidebar(memory, list(df.columns))

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

    # Earlier question/answer pairs, for follow-up questions. Failed ones are
    # skipped so error messages are never sent to Gemini as if it said them.
    messages = st.session_state.messages
    history = []
    for asked, answered in zip(messages[0::2], messages[1::2]):
        if not answered.get("failed"):
            history += [asked, answered]

    messages.append({"role": "user", "content": question})
    show_message(messages[-1])

    with st.spinner("Analysing..."):
        try:
            result = DataAgent(df, api_key, memory).ask(question, history)
        except errors.APIError as error:
            if error.code == 429:
                text = (
                    "⏳ The free Gemini quota for today is used up on every model. "
                    "Paste your own free API key in the sidebar "
                    "(get one at aistudio.google.com/apikey), or try again later."
                )
            elif error.code in (400, 401, 403) and "key" in str(error).lower():
                text = "🔑 That Gemini API key was rejected. Check the key and try again."
            else:
                text = f"❌ Gemini returned an error: {error}"
            result = {"answer": text, "steps": [], "model": None, "failed": True}
        except Exception as error:
            result = {"answer": f"❌ Something went wrong: {error}", "steps": [], "model": None, "failed": True}

    messages.append({
        "role": "assistant",
        "content": result["answer"],
        "steps": result["steps"],
        "model": result["model"],
        "failed": result.get("failed", False),
    })
    st.rerun()
