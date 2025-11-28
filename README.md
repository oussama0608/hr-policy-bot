# Project 1: Cited-Sources HR/Legal Policy Bot (RAG)

## The Business Problem
Companies have hundreds of HR/Safety/Compliance PDFs. Teams don’t trust generic AI because it hallucinates. This bot answers questions **only from your documents** and cites the exact page so users can trust the response.

## The Solution
- Ask any question about your policies; the bot retrieves only the most relevant chunks and answers from them.
- Strict citations: “According to the Safety Manual (Page 12), you must wear a helmet.”
- Source view: toggle to see the exact text chunks used.
- Configurable: API keys, knowledge base PDF, Pinecone index, chunking, top-k, and sample test questions.

## Stack
- Orchestrator: LangChain (Python)
- Vector DB: Pinecone (cloud; free tier works). ChromaDB is an optional local alternative but the current code is wired for Pinecone.
- LLM: OpenAI `gpt-4o-mini` (cheap/fast)
- UI: Streamlit

## Setup
1) Clone this repo and create a virtualenv (recommended).  
2) Install deps: `pip install -r requirements.txt`  
3) Provide keys (either via `.env` or in the UI sidebar):
   - `OPENAI_API_KEY`
   - `PINECONE_API_KEY`
   - `PINECONE_INDEX_NAME` (existing index; 1536-dim for `text-embedding-3-small`)
4) Optional: set a default PDF path with `PDF_FILE` in `.env`; otherwise choose/upload via UI.

## Run
```
streamlit run app.py
```

## Using the App
1) Open the sidebar and fill in OpenAI + Pinecone keys and index name.  
2) Select or upload your policy PDF.  
3) Adjust chunk size/overlap and top-k as needed.  
4) Click “Recreate knowledge base” to ingest the PDF into Pinecone (or it will auto-rebuild when empty).  
5) Use the sample question buttons or type your own question in chat.  
6) Expand “📚 View Source Citations” to inspect the chunks and page numbers used.

## Why This Wins on Upwork
- Eliminates hallucinations with strict, page-level citations.
- Shows transparency with a “Source View” so clients can audit answers.
- Runs on commodity hardware (Streamlit + Pinecone cloud) and is easy to reconfigure per client (swap PDFs, change chunking, adjust k).***
