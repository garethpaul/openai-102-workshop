import streamlit as st
from components import common
st.set_page_config(
    page_title="Hello 👋",
    page_icon="👋",
)
common.github_logo()
# show success message for 5 seconds
# st.sidebar.success("API Token saved")
st.write("# OpenAI Workshop👋")

st.code("""
- 👋 Welcome! We're all new here! 👨‍💻👩‍💻
- 🤗🧠 Let's apply some Emotional Intelligence to Artificial Intelligence
- 🔨 Workshop
    - 🧐 Getting Started with OpenAI APIs
    - 💊 LLMs - Large Language Models
    - 📝 Embeddings
    - 🔎 Text Search
    - 🤞 Recommendations
    - 🤹‍♀️ Clustering
    - 🦾 Fine Tuning (time permitting)
    - ⛓️ Langchain (time permitting)
""")
st.write("# Follow along!")
st.write(
    "The source code for this app is available on [GitHub](https://github.com/garethpaul/openai-102-workshop)")



common.add_logo()
