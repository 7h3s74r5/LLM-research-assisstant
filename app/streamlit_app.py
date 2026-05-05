import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
from src.preprocess import extract_text_from_pdf
from src.summarizer import summarize_text
from src.vector_store import create_vector_store, model
from src.qa_system import answer_question, get_relevant_context

st.title("📄 Research Paper Assistant")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    text = extract_text_from_pdf("temp.pdf")

    st.subheader("🔹 Summary")
    summary = summarize_text(text)
    st.write(summary)

    text_chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    index, _ = create_vector_store(text_chunks)

    question = st.text_input("Ask a question:")

    if question:
        context = get_relevant_context(question, text_chunks, index, model)
        answer = answer_question(question, context)

        st.subheader("💡 Answer")
        st.write(answer)
