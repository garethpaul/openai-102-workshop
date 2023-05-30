import streamlit as st
from sklearn.cluster import DBSCAN
from utils import generate

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from components import common
import numpy as np


def page_text_clustering():
    st.set_page_config(
        page_title="Clustering",
        page_icon="👋",
    )

    common.api_token()
    common.github_logo()
    # Generate example text
    # show this as a textarea box with a button to generate
    st.write("# 🤹‍♂️ Clustering")
    st.write("## Simple Clustering using Embeddings")
    st.write("We can also use embeddings to cluster text like reviews, emails etc")

    text_input = st.text_area("Text Input", "I enjoyed the movie. It was fantastic!\nThe restaurant service was terrible. I wouldn't recommend it.\nThe concert was amazing. The band played all my favorite songs.\nThe hotel room was spacious and clean. I had a comfortable stay.")
    if st.button("Cluster"):

        text_data = text_input.split("\n")
        # create a json object for the text data
        text_payload = [{'text': text} for text in text_data]

        # Get embeddings for the text data using OpenAI API
        embeddings = []
        for text in text_data:
            response = generate.get_embeddings(text, embedding_type='text')
            embedding = response[0]['embedding']
            embeddings.append(embedding)

        # Convert embeddings to NumPy array
        embeddings = np.array(embeddings)

        # Perform clustering using K-means
        n_clusters = 2
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(embeddings)
        labels = kmeans.labels_

        # Visualize cluster centroids
        centroids = kmeans.cluster_centers_

        # Scatter plot with cluster colors
        fig, ax = plt.subplots()
        for i, label in enumerate(set(labels)):
            indices = np.where(labels == label)
            ax.scatter(embeddings[indices, 0],
                       embeddings[indices, 1], label=f'Cluster {label + 1}')

        ax.scatter(centroids[:, 0], centroids[:, 1], marker='x',
                   color='red', s=100, label='Centroids')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title('Clustering Results')
        ax.legend()

        # Convert Matplotlib plot to Streamlit plot
        st.pyplot(fig)

        # Perform clustering using DBSCAN
        #dbscan = DBSCAN(eps=0.5, min_samples=5)
        # dbscan.fit(embeddings)
        #labels = dbscan.labels_

        # Visualize DBSCAN clusters
        unique_labels = set(labels)

        # Assign cluster labels to customers
        n_clusters = 3
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(embeddings)
        labels = kmeans.labels_
        for i, review in enumerate(text_payload):
            review['cluster'] = int(labels[i])

        st.write(text_payload)

        # Perform hierarchical clustering
        hierarchical = AgglomerativeClustering(n_clusters=3)
        labels = hierarchical.fit_predict(embeddings)

        # Visualize hierarchical clustering
        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        fig, ax = plt.subplots()
        for label, color in zip(unique_labels, colors):
            indices = np.where(labels == label)
            ax.scatter(embeddings[indices, 0], embeddings[indices,
                                                          1], color=color, label=f'Cluster {label}')

        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_title('Hierarchical Clustering Results')
        ax.legend()
        st.pyplot(fig)

    # import cluster from parent director
    # ImportError: attempted relative import with no known parent package
    import customer_cluster
    customer_cluster.main()


if __name__ == "__main__":
    page_text_clustering()
