import streamlit as st
from components import common
st.set_page_config(
    page_title="Hello 👋",
    page_icon="👋",
)
common.github_logo()
# show success message for 5 seconds
# st.sidebar.success("API Token saved")
st.write("# Welcome to OpenAI 102👋")
st.write(
    "This application contains a demo for OpenAI 102. This workshop is designed to help you learn more about OpenAI and how to use it")
st.write("## How to use this app")
st.markdown("""
1. Select a page from the sidebar 👈
2. Enter your API token 🔑
3. Follow the instructions on the page 📄
3. Have fun! 🤪
""")

st.write("## About this app")

st.write(
    "The source code for this app is available on [GitHub](https://github.com/garethpaul/embeddings-app)")

st.write("## Contact")
st.write(
    "If you have any questions, please contact [@gpj](https://twitter.com/gpj)")

common.add_logo()
