import streamlit as st
from main import get_news

st.title("Welcome to the news finder and summarizer")

question = st.text_input("Topic Name:")

if question:
    answer = get_news(question)
    st.text("Answer:")
    st.write(answer)