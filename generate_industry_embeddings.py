import json
import openai

# Set up OpenAI API credentials
#openai.api_key = "YOUR_API_KEY"
# Define the list of industries
industries = [
    "E-commerce",
    "Telecommunications",
    "Healthcare",
    "Finance",
    "Technology",
    "Education",
    "Retail",
    "Automotive",
    "Hospitality",
    "Entertainment",
    "Media",
    "Travel",
    "Food & Beverage",
    "Real Estate",
    "Manufacturing",
    "Transportation",
    "Logistics",
    "Energy",
    "Nonprofit",
    "Government"
]

# Create a dictionary to store the industry embeddings
industry_embeddings = {}

# Generate embeddings for each industry
for industry in industries:
    response = openai.Embedding.create(
        input=[industry],
        engine="text-embedding-ada-002"
    )
    embedding = response["data"]
    industry_embeddings[industry] = embedding

# Save the industry embeddings to a JSON file
with open("industry_embeddings.json", "w") as file:
    json.dump(industry_embeddings, file)
