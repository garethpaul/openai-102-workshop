from collections.abc import Iterable, Mapping
import random

from utils.generate import cosine_similarity


def _embedding_for(industry_embeddings, industry):
    entries = industry_embeddings.get(industry)
    if not isinstance(entries, list) or not entries:
        return None
    first_entry = entries[0]
    if not isinstance(first_entry, dict):
        return None
    return first_entry.get("embedding")


def build_similarity_scores(industry_embeddings):
    scores = {}
    for industry1 in industry_embeddings:
        embedding1 = _embedding_for(industry_embeddings, industry1)
        if embedding1 is None:
            continue
        scores[industry1] = {}
        for industry2 in industry_embeddings:
            embedding2 = _embedding_for(industry_embeddings, industry2)
            if embedding2 is None:
                continue
            scores[industry1][industry2] = cosine_similarity(
                embedding1, embedding2
            )
    return scores


def recommend_product(
    customer_id,
    customer_data,
    industry_embeddings,
    industry_products,
    choose_product=random.choice,
):
    if not isinstance(industry_embeddings, Mapping):
        return None, {}
    similarity_scores = build_similarity_scores(industry_embeddings)
    if (
        not isinstance(customer_data, Iterable)
        or isinstance(customer_data, (str, bytes, Mapping))
        or not isinstance(industry_products, Mapping)
    ):
        return None, similarity_scores

    customer = next(
        (
            item
            for item in customer_data
            if isinstance(item, Mapping)
            and item.get("customer_id") == customer_id
        ),
        None,
    )
    if customer is None:
        return None, similarity_scores

    customer_industry = customer.get("industry")
    if not isinstance(customer_industry, str) or not customer_industry.strip():
        return None, similarity_scores
    if _embedding_for(industry_embeddings, customer_industry) is None:
        return None, similarity_scores

    customer_scores = similarity_scores.get(customer_industry, {})
    if not customer_scores:
        return None, similarity_scores

    product_scores = {}
    validated_products = {}
    for industry, score in customer_scores.items():
        products = industry_products.get(industry)
        if not isinstance(products, list):
            continue
        product_names = [
            product.strip()
            for product in products
            if isinstance(product, str) and product.strip()
        ]
        if product_names:
            product_scores[industry] = score
            validated_products[industry] = product_names
    if not product_scores:
        return None, similarity_scores
    top_industry = max(product_scores, key=product_scores.get)
    return choose_product(validated_products[top_industry]), similarity_scores
