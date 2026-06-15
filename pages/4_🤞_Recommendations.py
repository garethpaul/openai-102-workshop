import streamlit as st
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

from components.recommendations import recommend_product

# Load customer data from JSON file
with open("customer_profiles.json", "r") as file:
    customer_data = json.load(file)

# Load industry embeddings from JSON file
with open("industry_embeddings.json", "r") as file:
    industry_embeddings = json.load(file)


INDUSTRY_PRODUCTS = {
    "E-commerce": ["Twilio Engage", "Marketing Campaigns"],
    "Telecommunications": [
        "Programmable Messaging",
        "Programmable Voice",
        "Programmable Video",
    ],
}


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
        selected_customer_id,
        customer_data,
        industry_embeddings,
        INDUSTRY_PRODUCTS,
    )

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
