import os
from dotenv import load_dotenv

# Document loading
from langchain_community.document_loaders import PyPDFLoader

# Text splitting  (CORRECT PACKAGE)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings + LLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Vector store
from langchain_community.vectorstores import FAISS

# LangChain core
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


# ---------------------------------------------------
# 1) Load Environment
# ---------------------------------------------------
load_dotenv()   # expects GROQ_API_KEY in .env

PDF_PATH = "islr.pdf"   # <-- put your PDF in same folder


# ---------------------------------------------------
# 2) Load PDF
# ---------------------------------------------------
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()


# ---------------------------------------------------
# 3) Chunk Text
# ---------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

splits = splitter.split_documents(docs)


# ---------------------------------------------------
# 4) Embeddings + Vector DB
# ---------------------------------------------------
emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vs = FAISS.from_documents(splits, emb)

retriever = vs.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)


# ---------------------------------------------------
# 5) Prompt Template
# ---------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer ONLY from the provided context. "
        "If the answer is not found, say you don't know."
    ),
    (
        "human",
        "Question: {question}\n\nContext:\n{context}"
    )
])


# ---------------------------------------------------
# 6) Groq LLM
# ---------------------------------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",   # you can change to mixtral-8x7b-32768
    temperature=0
)


# ---------------------------------------------------
# 7) Helper to format docs
# ---------------------------------------------------
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


# ---------------------------------------------------
# 8) Build Chain
# ---------------------------------------------------
parallel = RunnableParallel({
    "context": retriever | RunnableLambda(format_docs),
    "question": RunnablePassthrough()
})

chain = parallel | prompt | llm | StrOutputParser()


# ---------------------------------------------------
# 9) Run Q&A Loop
# ---------------------------------------------------
print("PDF RAG ready with GROQ. Ask a question (Ctrl+C to exit).")

while True:
    q = input("\nQ: ").strip()

    if not q:
        continue

    ans = chain.invoke(q)
    print("\nA:", ans)
