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
    customer = next(
        (item for item in customer_data if item.get("customer_id") == customer_id),
        None,
    )
    similarity_scores = build_similarity_scores(industry_embeddings)
    if customer is None:
        return None, similarity_scores

    customer_industry = customer.get("industry")
    if _embedding_for(industry_embeddings, customer_industry) is None:
        return None, similarity_scores

    customer_scores = similarity_scores.get(customer_industry, {})
    if not customer_scores:
        return None, similarity_scores

    top_industry = max(customer_scores, key=customer_scores.get)
    products = industry_products.get(top_industry, [])
    if not products:
        return None, similarity_scores
    return choose_product(products), similarity_scores
