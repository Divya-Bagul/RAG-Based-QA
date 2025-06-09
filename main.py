import streamlit as st
from utils.gemini import rewrite_text
from utils.loader import load_documents
from utils.embedder import embed_and_store, query_knowledge_base
from utils.fallback_llm import fallback_response

st.set_page_config(page_title="RAG Genie", layout="wide")
st.title("🧠 RAG Genie - Ask Your Files/URLs!")

uploaded_files = st.file_uploader("Upload PDF, DOCX, PPTX files", type=['pdf', 'docx', 'pptx'], accept_multiple_files=True)
url_input = st.text_input("Or paste a website URL to extract content:")

if st.button("Process Documents"):
    if uploaded_files or url_input:
        with st.spinner("Processing..."):
            docs = load_documents(uploaded_files, url_input)
            embed_and_store(docs)
        st.success("Documents processed and indexed.")
    else:
        st.warning("Upload at least one file or enter a URL.")

question = st.text_input("Ask a question:")
if st.button("Get Answer") and question:
    with st.spinner("Answering..."):
        answer, confidence = query_knowledge_base(question)
        if confidence < 0.3:
            answer = fallback_response(question)
            st.warning("Fallback LLM used via Groq.")
        st.markdown(f"**Answer:** {rewrite_text(answer)}")
        # st.markdown(f"**Answer:** {answer}")

