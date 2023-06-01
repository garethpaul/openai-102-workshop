import streamlit as st


st.markdown("# Getting Started with OpenAI APIs")
st.markdown("""
[OpenAI](https://openai.com) is an AI research and deployment company who are building safe and beneficial (AGI). Read more on [OpenAI API Docs](https://platform.openai.com/docs/api-reference).
""")

st.markdown("""## Concepts
- [Models](#models)
- [Prompts](#prompts)
- [Tokens](#tokens)
""")

st.markdown("## Models")
st.markdown("""
- GPT-4
- **GPT-3.5**
- DALL·E 
- Whisper
- **Embeddings**
- Moderation
- GPT-3
""")

st.markdown("## Prompts")
st.markdown("""
A prompt can be a sentence, a question, a phrase, or even just a word. It serves as a starting point for the AI to generate text based on its training data. For example, if you provide the prompt "Once upon a time", the AI might continue with a fairy tale-like story.

There are also different stratgies you may consider to think about prompts:
""")
with st.expander("zero-shot"):
    st.markdown("""
### Zero-shot
These prompts involve providing the model with a single prompt and asking it to generate a response based on that. In other words, you provide some input to the model once, and the model generates an output based on that input. This is the simplest way of using the model and can often produce good results, especially if your prompt is clear and specific.

For instance, if you're using the model to generate a story, a one-shot prompt might look like this:

"Translate English to French:
cheese =>"
""")
with st.expander("one-shot"):
    st.markdown("""
### Few-shot
These prompts involve providing the model with a prompt and single example and asking it to generate a response based on that. In other words, you provide some input to the model once, and the model generates an output based on that input. This is the simplest way of using the model and can often produce good results, especially if your prompt is clear and specific.

For example:

"Translate English to French:
sea otter > loutre de mer
cheese = >"
""")

with st.expander("few-shot"):
    st.markdown("""     
### Few-shot

In addition to the task description the model sees a few examples of the task.

"Translate English to French:
sea otter > loutre de mer
cheese > fromage
car > voiture
phone > téléphone
cheese = >"
    """)

st.markdown("## Tokens")
st.markdown("""
Tokens can be thought of as pieces of words and currency. Before the API processes the prompts, the input is broken down into tokens. These tokens are not cut up exactly where the words start or end - tokens can include trailing spaces and even sub-words.

### Token Limits
Depending on the model used, requests can use up to 4097 tokens shared between prompt and completion. If your prompt is 4000 tokens, your completion can be 97 tokens at most. 

### Token Pricing
The API offers multiple model types at different price points. Each model has a spectrum of capabilities, with davinci being the most capable and ada the fastest. Requests to these different models are priced differently. You can find details on token pricing [here](https://openai.com/pricing).  

The limit is currently a technical limitation, but there are often creative ways to solve problems within the limit, e.g. condensing your prompt, breaking the text into smaller pieces, etc.
""")
with st.expander("Token Tips & Tricks"):
    st.markdown("""
- Tokens can include trailing spaces so it's generally recommended to avoid trailing spaces e.g. "How do I bake a nice cake ?" vs "How do I bake a nice cake?". Trailing space characters may result in lower-quality output. This is because the API already incorporates trailing spaces in its dictionary of tokens.
- Biases for specific tokens can be set in the logit_bias parameter to modify the likelihood of the specified tokens appearing in the completion. Consider, for example, that we are building an AI Baking Assistant that is sensitive about its user’s egg allergies. To view completion probabilities in [Playground](https://platform.openai.com/playground?mode=complete) select Full Spectrum from the Show Probabilities dropdown. 
- Let's say we have a user using an AI Baking Assistant that is sensitive about egg allergies, we can use our knowledge of tokens to set biases in the logit_bias parameter in order to discourage the model from generating a response that includes any variation of the word ‘egg’ using [tokenizer](https://platform.openai.com/tokenizer) also see [tiktoken](https://github.com/openai/tiktoken).
- In the response payload for API requests you'll get details on the tokens used for every request. It's important to pay attention to these so you can control costs. 
- You can also control costs with something like this.
""")
    st.code("""
import tiktoken

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

num_tokens_from_string("this workshop is great!", "cl100k_base")""")

st.markdown("# Appendix")

st.markdown("""## Embedding
OpenAI’s text embeddings measure the relatedness of text strings. They can be used for a variety of tasks, including semantic search, semantic similarity, and zero-shot classification.

An embedding is a vector (list) of floating point numbers. The distance between two vectors measures their relatedness. Small distances suggest high relatedness and large distances suggest low relatedness.

To learn more see the section on [Embeddings](./Embeddings).
""")

st.markdown("""## Fine Tuning
Fine-tuning lets you get more out of the models available through the API by providing:

- Higher quality results than prompt design
- Ability to train on more examples than can fit in a prompt
- Token savings due to shorter prompts
- Lower latency requests

Fine-tuning improves on few-shot learning by training on many more examples than can fit in the prompt, letting you achieve better results on a wide number of tasks. Once a model has been fine-tuned, you won't need to provide examples in the prompt anymore. This saves costs and enables lower-latency requests.

See more in the [Fine Tuning](./FineTuning) section.
""")

st.markdown("""## Tooling

There are a number of tools available to help you get the most out of the OpenAI and other related APIs.

These tools include AutoGPT, Langchain, and more. See the [Langchain](./Langchain) section as an example.

""")

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
