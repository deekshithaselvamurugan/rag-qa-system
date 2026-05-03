# AI-Powered Document Q&A System (RAG Pipeline)

A production-ready Retrieval-Augmented Generation (RAG) application that allows users to upload documents and ask questions about them using AI.

## Features
- Upload PDF or TXT documents
- Automatic text chunking and embedding using HuggingFace
- Vector similarity search using FAISS
- AI-powered answers using Groq (Llama 3.1)
- Clean chat interface built with Streamlit

## Tech Stack
- **LangChain** — RAG pipeline orchestration
- **FAISS** — Vector database for similarity search
- **HuggingFace** — Local embedding model (all-MiniLM-L6-v2)
- **Groq + Llama 3.1** — LLM for answer generation
- **Streamlit** — Web UI

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/rag-qa-system.git
cd rag-qa-system
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API key
Create a `.env` file:
Get a free key at: https://console.groq.com

### 5. Run the app
```bash
streamlit run app.py
```

## How It Works
1. Upload a PDF or TXT document
2. Document is split into chunks and embedded locally
3. Embeddings stored in FAISS vector database
4. User question is embedded and similar chunks retrieved
5. Retrieved chunks + question sent to Llama 3.1 via Groq
6. Answer generated and displayed with source references
