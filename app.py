import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# 1. Load Secrets
load_dotenv()

# PDF to ingest (defaults to handbook.pdf). Override with env var PDF_FILE.
PDF_FILE = os.path.expanduser(os.getenv("PDF_FILE", "handbook.pdf"))

# 2. Setup Page
st.set_page_config(page_title="HR Policy Bot", layout="wide")
st.title("🛡️ AI HR Assistant (with Citations)")

# 3. Initialize Logic (Cached so it doesn't re-run every click)
@st.cache_resource
def setup_vector_store():
    # Initialize Pinecone Client
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # Define Embeddings (The translator from Text -> Numbers)
    # We use text-embedding-3-small to match your 1536 dimension index
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Check if index has data
    index_name = os.getenv("PINECONE_INDEX_NAME")
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    
    # If index is empty, we need to load the PDF
    if stats['total_vector_count'] == 0:
        st.warning(f"⚠️ Index is empty. Using PDF file: {PDF_FILE}")
        
        # Check if PDF exists
        if not os.path.exists(PDF_FILE):
            st.error(f"❌ PDF not found: {PDF_FILE}. Set PDF_FILE env var or place the file here.")
            return None

        with st.spinner(f"📚 Reading {os.path.basename(PDF_FILE)} and training AI... (This happens only once)"):
            # Load PDF
            loader = PyPDFLoader(PDF_FILE)
            docs = loader.load()
            
            # Split text into chunks 
            # Increased chunk size to 2000 to keep full paragraphs together
            splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
            splits = splitter.split_documents(docs)
            
            # Upload to Pinecone
            docsearch = PineconeVectorStore.from_documents(
                documents=splits, 
                embedding=embeddings, 
                index_name=index_name
            )
            st.success("✅ Knowledge Base Updated! The AI is ready.")
            time.sleep(2)
            st.rerun() # Refresh page to clear warnings
            return docsearch
    else:
        # Connect to existing index
        return PineconeVectorStore.from_existing_index(
            index_name=index_name, 
            embedding=embeddings
        )

# Initialize the database connection
vector_store = setup_vector_store()

# Initialize the Brain (GPT-4o-mini is cheap and fast)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 4. The Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. Handle User Input
if vector_store is not None:
    if prompt := st.chat_input("Ask about vacation, safety, or dress code..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 6. The "RAG" Logic (The Magic Part)
        with st.chat_message("assistant"):
            # A. Search Pinecone for the 3 most relevant paragraphs
            results = vector_store.similarity_search(prompt, k=5)
            
            # B. Construct the context
            context_text = "\n\n".join([doc.page_content for doc in results])
            
            # C. Ask AI to answer using ONLY that context
            system_prompt = f"""You are a helpful HR assistant. 
            Answer the question based ONLY on the context provided below.
            If the answer is not in the context, say "I cannot find that information in the handbook."
            
            Context:
            {context_text}
            """
            
            response = llm.invoke([
                ("system", system_prompt),
                ("user", prompt)
            ])
            
            # D. Display Answer
            st.markdown(response.content)
            
            # E. THE "PRO" FEATURE: Show Sources
            with st.expander("📚 View Source Citations"):
                for i, doc in enumerate(results):
                    page_num = doc.metadata.get('page', 'Unknown') + 1
                    st.markdown(f"**Source {i+1} (Page {page_num}):**")
                    # Show the FULL text chunk, no cutoff
                    st.text(doc.page_content) 
                    st.divider()
                    
        # Save assistant response
        st.session_state.messages.append({"role": "assistant", "content": response.content})
