# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from uuid import uuid4

from utils.gemini import rewrite_text
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = None
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


# def embed_and_store(documents):
#     global vectorstore
#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     docs = splitter.split_documents(documents)
#     vectorstore = FAISS.from_documents(docs, embedding)


# 2. Embed and store documents using FAISS
def embed_and_store(documents):
    global vectorstore
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    # Assign unique ID in metadata
    for doc in docs:
        doc.metadata["id"] = str(uuid4())

    # Create FAISS vectorstore
    vectorstore = FAISS.from_documents(docs, embedding)

def query_knowledge_base(question):
    global vectorstore
    if not vectorstore:
        return "No documents indexed yet.", 0.0
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in docs])
   
    return context[:2000], 0.7 if context.strip() else 0.0



