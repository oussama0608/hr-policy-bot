import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone

# 1. Load Secrets
load_dotenv()

# 2. Setup Page
st.set_page_config(page_title="HR Policy Bot", layout="wide")
st.title("🛡️ AI HR Assistant (with Citations)")

# Defaults (can be overridden via UI)
DEFAULT_PDF = os.path.expanduser(os.getenv("PDF_FILE", "handbook.pdf"))
DEFAULT_INDEX = os.getenv("PINECONE_INDEX_NAME", "")

# --- Sidebar: configuration ---
st.sidebar.header("Configuration")

openai_key = st.sidebar.text_input(
    "OpenAI API Key",
    value=os.getenv("OPENAI_API_KEY", ""),
    type="password",
    help="Used for embeddings and chat."
)
pinecone_key = st.sidebar.text_input(
    "Pinecone API Key",
    value=os.getenv("PINECONE_API_KEY", ""),
    type="password",
    help="Used to connect to your Pinecone index."
)
index_name = st.sidebar.text_input(
    "Pinecone Index Name",
    value=DEFAULT_INDEX,
    help="Existing index to read/write vectors."
).strip()

# Sync env for downstream libs
if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key
if pinecone_key:
    os.environ["PINECONE_API_KEY"] = pinecone_key
if index_name:
    os.environ["PINECONE_INDEX_NAME"] = index_name

pdf_path = st.sidebar.text_input(
    "Knowledge Base PDF",
    value=DEFAULT_PDF,
    help="Path to the handbook or policy PDF."
)
uploaded_pdf = st.sidebar.file_uploader("Upload PDF to use", type=["pdf"])
if uploaded_pdf:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    uploaded_path = uploads_dir / uploaded_pdf.name
    uploaded_path.write_bytes(uploaded_pdf.read())
    pdf_path = str(uploaded_path)
    st.sidebar.success(f"Uploaded and selected: {uploaded_path.name}")

chunk_size = st.sidebar.slider("Chunk size", min_value=500, max_value=4000, value=2000, step=100)
chunk_overlap = st.sidebar.slider("Chunk overlap", min_value=0, max_value=1000, value=500, step=50)
top_k = st.sidebar.slider("Top K (retrieval)", min_value=1, max_value=10, value=5, step=1)

default_samples = "\n".join(
    [
        "What is the vacation policy?",
        "What is the dress code?",
        "How do I report a workplace safety issue?",
        "What is the parental leave policy?",
    ]
)
sample_questions_raw = st.sidebar.text_area(
    "Sample questions (one per line)",
    value=default_samples,
    height=120,
)

recreate_clicked = st.sidebar.button("Recreate knowledge base", type="primary")
if recreate_clicked or uploaded_pdf:
    st.session_state["recreate_token"] = time.time()
recreate_token = st.session_state.get("recreate_token", 0)


def require_keys():
    """Quick guard to avoid confusing errors when keys are missing."""
    missing = []
    if not os.getenv("OPENAI_API_KEY"):
        missing.append("OpenAI API Key")
    if not os.getenv("PINECONE_API_KEY"):
        missing.append("Pinecone API Key")
    if not index_name:
        missing.append("Pinecone Index Name")
    if missing:
        st.warning("Please fill these before chatting: " + ", ".join(missing))
        return False
    return True


# 3. Initialize Logic (Cached so it doesn't re-run every click)
@st.cache_resource(show_spinner=False)
def setup_vector_store(pdf_file, chunk_sz, chunk_ov, index, recreate_marker):
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    index_client = pc.Index(index)
    stats = index_client.describe_index_stats()

    should_recreate = recreate_marker or stats.get("total_vector_count", 0) == 0
    if should_recreate:
        if not Path(pdf_file).exists():
            st.error(f"❌ PDF not found: {pdf_file}. Upload or point to a valid file.")
            return None

        try:
            index_client.delete(delete_all=True)
        except Exception:
            st.warning("Could not clear existing vectors; proceeding to upsert new chunks.")

        with st.spinner(f"📚 Reading {Path(pdf_file).name} and refreshing knowledge base..."):
            loader = PyPDFLoader(pdf_file)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_sz,
                chunk_overlap=chunk_ov,
            )
            splits = splitter.split_documents(docs)

            docsearch = PineconeVectorStore.from_documents(
                documents=splits,
                embedding=embeddings,
                index_name=index,
            )
            st.success("✅ Knowledge base updated.")
            return docsearch

    return PineconeVectorStore.from_existing_index(index_name=index, embedding=embeddings)


vector_store = setup_vector_store(pdf_path, chunk_size, chunk_overlap, index_name, recreate_token) if require_keys() else None
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 4. The Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []


def run_query(prompt_text: str):
    """Handle a user or sample prompt."""
    if not prompt_text.strip():
        return

    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    if vector_store is None:
        with st.chat_message("assistant"):
            st.error("Vector store is not ready. Add API keys and recreate the knowledge base.")
        return

    with st.chat_message("assistant"):
        results = vector_store.similarity_search(prompt_text, k=top_k)
        if not results:
            st.error("No results found in the knowledge base.")
            return

        context_text = "\n\n".join(doc.page_content for doc in results)
        system_prompt = f"""You are a helpful HR assistant.
Answer the question based ONLY on the context provided below.
If the answer is not in the context, say "I cannot find that information in the handbook."

Context:
{context_text}
"""

        response = llm.invoke([("system", system_prompt), ("user", prompt_text)])
        st.markdown(response.content)

        with st.expander("📚 View Source Citations"):
            for i, doc in enumerate(results):
                page_num = doc.metadata.get("page", "Unknown")
                page_label = page_num + 1 if isinstance(page_num, int) else page_num
                st.markdown(f"**Source {i + 1} (Page {page_label}):**")
                st.text(doc.page_content)
                st.divider()

    st.session_state.messages.append({"role": "assistant", "content": response.content})


# Display previous chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Sample questions
sample_questions = [q.strip() for q in sample_questions_raw.splitlines() if q.strip()]
if sample_questions:
    st.subheader("Try sample questions")
    cols = st.columns(2)
    for idx, question in enumerate(sample_questions):
        col = cols[idx % 2]
        if col.button(question, key=f"sample-{idx}"):
            run_query(question)

# 6. Handle User Input
if prompt := st.chat_input("Ask about vacation, safety, or dress code..."):
    run_query(prompt)
