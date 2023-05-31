import streamlit.components.v1 as components
import streamlit as st
from utils import crawler, token, generate
from components import common
from uuid import uuid4
import pandas as pd

PICKLE_FILE_PATH = 'embeddings.pkl'


def page():
    st.set_page_config(
        page_title="Text Search",
        page_icon="👋",
    )
    common.sidebar()
    st.write("# 🔍 Text Search")
    st.write("## Text Search using Embeddings")
    # get the user input
    openai_api_key = st.text_input(
        "OpenAI API Key", type="password")
    # save the API key in the session state
    st.session_state.openai_api_key = openai_api_key
    import os
    os.environ["OPENAI_API_KEY"] = openai_api_key

    st.write("Now we can crawl a list of URLs and generate embeddings for each page. We can use these embeddings to compare a query to most similar text from crawled pages.")
    # User input for text list and new text
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Question**")
        new_text = st.text_input("Query:", "What is A2P 10DLC?")
        new_text = new_text.strip()

    with col2:
        st.write("**Enter a list of urls to crawl**")
        default_links = "https://www.twilio.com/docs/sms/a2p-10dlc\nhttps://www.twilio.com/docs/sms/a2p-10dlc/onboarding-isv\nhttps://support.twilio.com/hc/en-us/articles/1260800720410-What-is-A2P-10DLC-\nhttps://www.twilio.com/en-us/support-plans"
        text_input = st.text_area("URLs (one per line)", value=default_links)

    url_list = []

    if st.button("Find similar text"):
        chunks = []
        url_list = [url.strip() for url in text_input.split("\n")]
        for url in url_list:
            text = crawler.get_text(url)

            texts = token.text_splitter.split_text(text)
            chunks.extend([{
                'id': str(uuid4()),
                'text': texts[i],
                'chunk': i,
                'url': url
            } for i in range(len(texts))])
            # get the embeddings for the text
        text_embeddings = [generate.get_embeddings(
            chunk['text'], embedding_type='url') for chunk in chunks]
        # get the embeddings for the new text
        new_text_embeddings = generate.get_embeddings(new_text)
        # calculate the similarity scores
        results = []
        for i, chunk in enumerate(chunks):
            embedding_new = new_text_embeddings[0]['embedding']
            embedding_text = text_embeddings[i][0]['embedding']
            similarity_score = generate.cosine_similarity(
                embedding_new, embedding_text)
            euclidean_dist = generate.euclidean_distance(
                embedding_new, embedding_text)
            manhattan_dist = generate.manhattan_distance(
                embedding_new, embedding_text)
            results.append((chunk['text'], similarity_score,
                           euclidean_dist, manhattan_dist))

        # sort the results by similarity score
        results = sorted(results, key=lambda x: x[1], reverse=True)

        # only show top 10 results
        results = results[:10]
        # create a dataframe for the results
        df_results = pd.DataFrame(
            results, columns=["Text", "Cosine Similarity", "Euclidean Distance", "Manhattan Distance"])
        # display the results in a table
        st.table(df_results)

    with st.expander("Show code for crawling, chunking and getting embeddings"):
        if url_list == []:
            url_list = [url.strip() for url in default_links.split("\n")]
        st.code("""
import openai
import tiktoken
import requests
from bs4 import BeautifulSoup

from langchain.text_splitter import RecursiveCharacterTextSplitter

urls = {url_list}
chunks = []
for item in urls:
    text = crawler_get_text(item)
    texts = token.text_splitter.split_text(text)
    chunks.extend([{{'id': str(uuid4()), 'text': texts[i],
                  'chunk': i, 'url': item}} for i in range(len(texts))])
text_embeddings = [get_embeddings(
    chunk['text'], embedding_type='url') for chunk in chunks]

def get_embeddings(query):
    res = openai.Embedding.create(
            input=[query],
            engine="text-embedding-ada-002"
    )
    return res.data[0]['embedding']

def crawler_get_text(url):
    # crawl the url with requests and get the text back
    # Crawl the URL with requests and get the text back
    # create fake browser for chrome to send the request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)...`
    }

    response = requests.get(url, headers=headers)
    html_content = response.text

    # Parse HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Extract the text from the parsed HTML
    text = soup.get_text()
    return text


tokenizer = tiktoken.get_encoding('p50k_base')

def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["/n/n", "/n", " ", ""] # replace with new line
)
    """)

    with st.expander("Show code for calculating similarity scores"):
        st.code(f"""
results = []
query = '{new_text}'
query_embeddings = get_embeddings(query)
for i, chunk in enumerate(chunks):
    embedding_new = query_embeddings[0]['embedding']
    embedding_text = text_embeddings[i][0]['embedding']
    similarity_score = cosine_similarity(
        embedding_new, embedding_text)
    results.append((chunk['text'], similarity_score))
print(results)

def cosine_similarity(embedding1, embedding2):
    dot_product = sum(val1 * val2 for val1,
                    val2 in zip(embedding1, embedding2))
    norm_vector1 = math.sqrt(sum(val * val for val in embedding1))
    norm_vector2 = math.sqrt(sum(val * val for val in embedding2))
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity
    """)
    st.write("## Augmented Text Search w/Twilio Docs (local)")
    st.markdown("""
    Recently semantic search has exploded. Everyone is chatting with their data, using the same basic recipe:
1. Take a set of documents and split them into paragraphs.
2. Convert each paragraph into a vector of embeddings that represent its coordinates in the semantic space.
3. Given a question, embed it in the same space. Find a few relevant paragraphs via semantic similarity between the vectors.
4. Tell GPT to answer the question using the information contained in them, with a prompt like “You are an expert on the Twilio and helpful Q&A bot. The user just asked you a question about this subject. Answer it using only the information contained in the following paragraphs.”
    """)
    st.write(
        "We can use a similar technique to compare a query to most similar text from crawled pages")

    nn_model, metadata = generate.load_embeddings_and_train_model(
        PICKLE_FILE_PATH)
    query = st.text_input("Ask me a question about Twilio",
                          "What is Twilio Verify?")
    use_embeddings = st.checkbox("Use Embeddings")

    if use_embeddings:
        xq = generate.get_embeddings(query)[0]['embedding']
        top_k_metadata = generate.get_top_k_metadata(xq, nn_model, metadata)
        with st.expander("Preview Top K Metadata"):
            st.write(top_k_metadata)
        augmented_query = generate.create_augmented_query(
            top_k_metadata, query)
        with st.expander("Show Augmented Query"):
            st.write(top_k_metadata)

        response = generate.get_model_response(augmented_query)
        st.write(response['message']['content'])

        st.write("Sources:")
        for i, item in enumerate(top_k_metadata):
            st.write(f"URL for item {i+1}: {item['url']}")
    else:
        if query != "Ask me a question about Twilio":
            response = generate.get_generic_response(query)
            st.write(response)

    with st.expander("Show code for augmented text search"):
        st.code("""GIST""")


if __name__ == "__main__":
    page()
