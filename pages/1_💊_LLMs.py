import streamlit as st

st.markdown("""
# How we got to LLMs?
The history of large language models can be traced back to the development of neural networks and natural language processing (NLP) techniques. 

GenAI models use NNs (Neural Networks) to identify patterns and structures within existing data to generate new and original content
- Unsupervised training
- Semi-supervised learning

Breakthrough in leveraging large amount of unlabeled data to create foundation models (LLMs)
From millions to 10 Trillion words
i.e GPT-3 and Stable Diffusion
""")
# add image from url
st.markdown("## Broad range of applications and capabilities")
st.image("https://www.sequoiacap.com/wp-content/uploads/sites/6/2022/09/genai-landscape-8.png", width=700)

st.markdown("  # The Basics of Large Language Models")
st.markdown("## What is a Language Model?")
st.markdown("A language model is a statistical model that can be used to predict the next word in a sequence of words. Language models are used in a variety of applications, including speech recognition, machine translation, and text generation.")

st.markdown("## Concepts")
st.markdown("### Prompts")
st.markdown("""Prompts are how you “program” the model by providing some instructions or a few examples.
- Show and Tell
- Provide Quality Data
- Settings
""")
st.markdown(
    "[Example Prompts](https://gist.github.com/garethpaul/50dc3e31eacf0707e59d6035b70b8f6a)")
st.image("https://storage.cloud.google.com/artifacts.gjones-webinar.appspot.com/llm_prompts.png", width=700)

st.markdown("### Models")
st.markdown("GPT-4 is a language model that was trained on a large corpus of text. It is a transformer-based model that was trained on a dataset of 175 billion parameters. It is a large language model that can be used to generate text, images, and more. It is a powerful tool that can be used to generate text for a variety of use cases")
st.markdown("Models have parameters that can be used to tune the model")
st.markdown("""
**Temperature**
Controls the output of the model - creative vs deterministic

**Top-k and Top-p**
Top-k: Pick next token from top ‘k’ tokens by probability
Top-p: Pick from top tokens based on sum of their probabilities
Top-p more dynamic - used to exclude outputs with lower probabilities

*Stop Sequences*
String that tells the model to stop generating more content


*Frequency and Presence Penalties*
Frequency penalizes tokens already appearing (including prompt)
Presence applies regardless of frequency
""")
st.markdown("### Tokens")
st.image("https://storage.cloud.google.com/artifacts.gjones-webinar.appspot.com/tokens.png", width=700)
st.image("https://storage.cloud.google.com/artifacts.gjones-webinar.appspot.com/token_workflow.png", width=700)

st.markdown("### Embeddings")
st.markdown("""
A set of models that can convert text into a numerical form

Embeddings are commonly used for:
- Search (where results are ranked by relevance to a query string)
- Clustering (where text strings are grouped by similarity)
- Recommendations (where items with related text strings are recommended)
- Anomaly detection (where outliers with little relatedness are identified)
- Diversity measurement (where similarity distributions are analyzed)
- Classification (where text strings are classified by their most similar label)
""")
