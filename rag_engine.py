import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def load_document(file_path: str):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def build_vector_store(chunks):
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def load_vector_store():
    embeddings = get_embeddings()
    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

def build_qa_chain(vector_store):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = PromptTemplate.from_template("""You are a helpful assistant. \
Use the following context to answer the question. \
If the answer is not in the context, say \
"I don't have enough information in the provided documents to answer this."

Context:
{context}

Question: {question}

Answer:""")

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return {"chain": chain, "retriever": retriever}

def process_document_and_build_index(file_path: str):
    print("Loading document...")
    docs = load_document(file_path)
    print("Splitting into chunks...")
    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks")
    print("Building vector store...")
    vector_store = build_vector_store(chunks)
    print("Done! Index saved.")
    return vector_store

def ask_question(question: str, qa_chain: dict) -> dict:
    chain = qa_chain["chain"]
    retriever = qa_chain["retriever"]
    answer = chain.invoke(question)
    source_docs = retriever.invoke(question)
    return {
        "answer": answer,
        "sources": [doc.page_content[:200] for doc in source_docs]
    }