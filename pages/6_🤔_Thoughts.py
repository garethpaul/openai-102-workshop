import streamlit as st

st.set_page_config(
    page_title="Hello 👋",
    page_icon="👋",
)

st.write("# Thoughts")
st.markdown("""
Right now, the options for the LLM are very limited. As I write this OpenAI is clearly in the lead, and there is little reason to use anything besides GPT4 or GPT3.5. However, there are many more options for embeddings. Unlike the GPT models, OpenAI’s embedding are not clearly superior. If you look at [benchmarks](https://huggingface.co/spaces/mteb/leaderboard), you will find models that score higher than ada-002. In particular the Instructor models (xl and large) do very well. Of course benchmarks don’t mean much in isolation. What matters to you is the right compromise between variables such as cost, performance, or speed.


""")
