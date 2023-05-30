import streamlit as st
from components import common
st.set_page_config(
    page_title="Hello 👋",
    page_icon="👋",
)
common.github_logo()
# show success message for 5 seconds
# st.sidebar.success("API Token saved")
st.write("# Welcome 👋")
st.write(
    "This application contains a demo for using Embeddings with OpenAI.")
st.write("## How to use this app")
st.markdown("""
1. Select a page from the sidebar 👈
2. Enter your API token 🔑
3. Follow the instructions on the page 📄
3. Have fun! 🤪
""")

st.write("## About this app")
st.write(
    "This app was created by [Streamlit](https://streamlit.io) and [OpenAI](https://openai.com) to demonstrate how to use Embeddings with OpenAI")
st.write(
    "The source code for this app is available on [GitHub](https://github.com/garethpaul/embeddings-app)")
st.write(
    "The app is hosted on [Streamlit Sharing](https://share.streamlit.io/garethpaul/embeddings-app/main/app.py)")

st.write("## Contact")
st.write(
    "If you have any questions, please contact [@gpj](https://twitter.com/gpj)")
