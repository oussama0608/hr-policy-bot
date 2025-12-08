# 🛡️ HR Policy Bot (RAG with Citations)

> **Eliminate hallucinations.** This AI assistant answers employee questions using *only* your verified HR handbooks, citing the exact page number for every response.

[[ Watch the 60-Second Demo Video ]] (Add your Loom/YouTube link here)

---

## 🚀 The Problem vs. The Solution

**The Problem:** Generic AI chatbots hallucinate. They invent policies that don't exist, creating legal liability.  
**The Solution:** This Retrieval-Augmented Generation (RAG) pipeline grounds every answer in your specific PDF documents. If the answer isn't in the handbook, the bot says "I don't know."

**Key Capabilities:**
*   **Strict Source Retrieval:** Uses LangChain & Pinecone to find the most relevant policy chunks.
*   **Page-Level Citations:** "According to Source 1 (Page 12)..." — verify answers instantly.
*   **Admin Dashboard:** Upload entirely new PDFs and adjust retrieval parameters (chunk size, overlap, top-k) without touching code.

## 🛠️ Tech Stack

*   **Core:** Python 3.10+
*   **Orchestration:** LangChain
*   **Vector Database:** Pinecone (Serverless)
*   **LLM:** OpenAI GPT-4o-mini
*   **UI:** Streamlit

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/hr-policy-bot.git
    cd hr-policy-bot
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file or enter these in the UI sidebar:
    ```bash
    OPENAI_API_KEY=sk-...
    PINECONE_API_KEY=...
    PINECONE_INDEX_NAME=hr-policy-index
    ```

## 🚦 Usage

1.  **Run the application:**
    ```bash
    streamlit run app.py
    ```
2.  **Open your browser:** Go to `http://localhost:8501`.
3.  **Upload a PDF:** Use the sample `handbook.pdf` or upload your own.
4.  **Ask a question:** "What is the policy on remote work?"

## 💡 How It Works

1.  **Ingestion:** The PDF is loaded and split into overlapping chunks (e.g., 2000 characters).
2.  **Embedding:** Each chunk is converted into a vector using OpenAI Embeddings and stored in Pinecone.
3.  **Retrieval:** When you ask a question, the system searches for the top K most similar chunks.
4.  **Generation:** The LLM receives your question + the retrieved chunks and generates an answer based *only* on that context.

## 📄 License
MIT
