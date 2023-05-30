import streamlit as st
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Load customer data from JSON file
with open("customer_profiles.json", "r") as file:
    customer_data = json.load(file)

# Load industry embeddings from JSON file
with open("industry_embeddings.json", "r") as file:
    industry_embeddings = json.load(file)


def calculate_similarity(embedding1, embedding2):
    # Use cosine similarity to compare the embeddings
    similarity = np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )
    return similarity


def recommend_product(customer_id):
    customer = next(
        (c for c in customer_data if c["customer_id"] == customer_id), None
    )
    if customer:
        customer_industry = customer.get("industry")
        customer_embedding = np.array(
            industry_embeddings[customer_industry][0].get("embedding")
        )

        # Calculate similarity scores between customer embedding and industry embeddings
        similarity_scores = {}
        for industry1, embeddings1 in industry_embeddings.items():
            similarity_scores[industry1] = {}
            for industry2, embeddings2 in industry_embeddings.items():
                embedding1 = np.array(embeddings1[0].get("embedding"))
                embedding2 = np.array(embeddings2[0].get("embedding"))
                similarity = calculate_similarity(embedding1, embedding2)
                similarity_scores[industry1][industry2] = similarity

        # Sort industries by similarity score in descending order
        sorted_scores = {
            industry: {
                k: v for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True)
            }
            for industry, scores in similarity_scores.items()
        }

        # Retrieve top matching industry
        top_industry = list(sorted_scores.keys())[0]

        # Retrieve products associated with the top industry
        products = get_products_by_industry(top_industry)

        # Return a random product from the list
        return np.random.choice(products), sorted_scores

    return None, None


def get_products_by_industry(industry):
    # Define product mappings for each industry
    industry_products = {
        "E-commerce": ["Twilio Engage", "Marketing Campaigns"],
        "Telecommunications": ["Programmable Messaging", "Programmable Voice", "Programmable Video"],
        # Add more industry-product mappings here
    }
    return industry_products.get(industry, [])


def main():
    st.title("Recommendations")

    # Select customer
    customer_ids = [customer["customer_id"] for customer in customer_data]
    selected_customer_id = st.selectbox("Select Customer ID", customer_ids)

    # Display customer information
    customer = next(
        (c for c in customer_data if c["customer_id"]
         == selected_customer_id), None
    )
    if customer:
        st.subheader("Customer Information")
        st.write(f"Customer ID: {customer['customer_id']}")
        st.write(f"Name: {customer['name']}")
        st.write(f"Industry: {customer['industry']}")

    # Generate product recommendation
    recommended_product, similarity_scores = recommend_product(
        selected_customer_id)

    # Display recommendation
    if recommended_product:
        st.subheader("Recommended Product")
        st.info(recommended_product)
    else:
        st.info("No recommendation available.")

    # Visualize similarity scores as a bar chart
    if similarity_scores:
        st.subheader("Similarity Scores")
        industries = list(similarity_scores.keys())
        similarities = [
            list(scores.values()) for scores in similarity_scores.values()
        ]

        fig, ax = plt.subplots()
        for i, industry in enumerate(industries):
            ax.barh(industry, similarities[i])

        ax.set_ylabel("Industry")
        ax.set_xlabel("Similarity Score")
        ax.set_title("Similarity Scores for Industries")

        st.pyplot(fig)

    # Visualize similarity scores as a heatmap
    if similarity_scores:
        st.subheader("Similarity Scores Heatmap")
        scores_array = np.array(
            [
                [similarity_scores[industry1][industry2]
                    for industry2 in industries]
                for industry1 in industries
            ]
        )

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            scores_array,
            annot=False,
            fmt=".2f",
            cmap="YlGnBu",
            xticklabels=industries,
            yticklabels=industries,
            cbar_kws={"shrink": 0.4},  # Reduce the size of the colorbar
            ax=ax,
        )

        for i in range(len(industries)):
            for j in range(len(industries)):
                text = ax.text(
                    j + 0.5, i + 0.5, f"{scores_array[i, j]:.2f}", ha="center", va="center", fontsize=7)

        ax.tick_params(axis="both", which="both", labelsize=8)

        ax.set_xlabel("Industry")
        ax.set_ylabel("Industry")
        ax.set_title("Pairwise Similarity Scores between Industries")

        st.pyplot(fig)


if __name__ == "__main__":
    main()
