# RAG Genie - Ask Your Files/URLs!

# Required Libraries
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
# from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import (
    PyPDFLoader, UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader, WebBaseLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import os   
from langchain_community.llms import HuggingFaceHub
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import ChatOpenAI

# Title
st.title("📄 RAG Genie - Ask Your Files/URLs!")

# File Uploader
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf", "docx", "pptx"])
url_input = st.text_input("Or enter a website URL")

# Ensure only one input is provided
if uploaded_file and url_input:
    st.warning("Please provide only one input: either upload a file or enter a URL.")
    st.stop()

if not uploaded_file and not url_input:
    st.info("Upload a file or enter a URL to start.")
    st.stop()

# ------------------------ Document Loading ------------------------
documents = []
if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1]
    file_path = f"temp.{file_type}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
    elif file_type == "docx":
        loader = UnstructuredWordDocumentLoader(file_path)
    elif file_type == "pptx":
        loader = UnstructuredPowerPointLoader(file_path)
    else:
        st.error("Unsupported file type.")
        st.stop()

    documents = loader.load()

elif url_input:
    loader = WebBaseLoader(url_input)
    documents = loader.load()

# ------------------------ Document Loading ------------------------

# Load and Process PDF
if uploaded_file is not None or url_input is not None:
    # with open("temp.pdf", "wb") as f:
    #     f.write(uploaded_file.read())

    # loader = PyPDFLoader("temp.pdf")
    # documents = loader.load()

    # ---------------------------- create  embedding for vectore ----------------------------
    # Split Text
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Vector Store
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()
    # ---------------------------- end create  embedding for vectore ----------------------------
   
    # ---------------------------- LLM ----------------------------
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile"
    )
    # ---------------------------- Prompt Template for groq----------------------------

    # ---------------------------- Prompt Template for langchain chain----------------------------
    prompt_template = """
    You are a helpful assistant. Use the following context to answer the question at the end.
    If the answer is not found in the context, say "The answer is not available in the documents."

    Context:
    {context}

    Question:
    {question}
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    # ---------------------------- End Prompt Template for langchain chain----------------------------
    
    # ----------------------------Call llm and langchain----------------------------
    llm_router = ChatOpenAI(
        model_name="mistralai/mistral-7b-instruct",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.getenv("OPENAI_ROUTER_API_KEY"),  # <- added here
        temperature=0.3
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_router,
        retriever=retriever,  # your vector store retriever
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    # ---------------------------- end Call llm and langchain----------------------------


    # Input Question
    question = st.text_input("Ask a question based on the PDF")
    if question and qa_chain:
        response = qa_chain(question)
        docs = response["source_documents"]

        st.markdown("### 📌 Answer:")
        st.write(response["result"])
        

        st.markdown("### 📄 Source Documents:")
        for i, doc in enumerate(response["source_documents"]):
            st.write(f"**Source {i+1}:** {doc.metadata['source']}")
            st.write(doc.page_content[:300] + "...")
    

else:
    st.info("Please upload a PDF to begin.")
