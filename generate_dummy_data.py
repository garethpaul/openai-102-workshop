import random
import json

# Generate Twilio customer purchase data
customer_data = []

for customer_id in range(1, 301):
    # Increase the probability of selecting "sms" by repeating it multiple times in the product SKUs list
    product_skus = random.choices(
        ["sms", "voice", "flex", "segment", "sendgrid", "sms", "sms"], k=random.randint(1, 5))
    programming_language = random.choice(
        ["Python", "JavaScript", "Ruby", "Java", "C#"])
    average_monthly_spend = random.randint(100, 1000)

    customer = {
        "customer_id": customer_id,
        "product_skus": product_skus,
        "programming_language": programming_language,
        "average_monthly_spend": average_monthly_spend
    }

    customer_data.append(customer)

# Write customer data to a JSON file
with open("customer_data.json", "w") as file:
    json.dump(customer_data, file)
