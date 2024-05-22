import streamlit as st
from src.embedder import search, embed_resume_texts

st.title('Resume Semantic Search')

query = st.text_input('Enter your query:')
k = st.number_input("Enter the value of k:", min_value = 1, value=5)
if query:
    if embed_resume_texts():
        top_k_output = search(query, k)
    for idx, text in top_k_output:
        st.write(f"Resume {idx}:")
        st.write(text)

