import json
import random

customer_data = []

products = [
    "Programmable Messaging",
    "Programmable Voice",
    "Programmable Video",
    "WhatsApp Business API",
    "Twilio Flex",
    "Twilio Conversations",
    "Twilio Engage",
    "Sync",
    "Twilio Frontline",
    "Marketing Campaigns",
    "Lookup",
    "Verify",
    "Intelligence",
    "Studio",
    "TaskRouter",
    "Trust Hub",
    "Event Streams",
    "Channel APIs",
    "Super Network",
    "Phone Numbers",
    "Short Codes",
    "Elastic SIP Trunking",
    "Interconnect",
    "Programmable Wireless",
    "Super SIM",
    "Narrowband",
    "Serverless",
    "Sync"
]

industries = [
    "E-commerce",
    "Telecommunications",
    "Marketing",
    "Healthcare",
    "Finance",
    "Travel",
    "Education",
    "Hospitality",
    "Retail",
    "Real Estate",
    "Automotive",
    "Technology",
    "Entertainment",
    "Transportation",
    "Logistics",
    "Media",
    "Food & Beverage",
    "Nonprofit",
    "Gaming",
    "Insurance"
]

for customer_id in range(1, 21):
    industry = random.choice(industries)
    products_list = []

    if random.random() < 0.8:
        products_list.append("Programmable Messaging")

    if random.random() < 0.4:
        random_product = random.choice(products)
        if random_product != "Programmable Messaging":
            products_list.append(random_product)

    customer = {
        "customer_id": customer_id,
        "name": f"Customer {customer_id}",
        "industry": industry,
        "products": products_list
    }

    customer_data.append(customer)

# Save the customer data to a JSON file
with open("customer_profiles.json", "w") as file:
    json.dump(customer_data, file, indent=2)
