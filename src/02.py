# added/edited
import os

import numpy as np
import openai
from dotenv import load_dotenv
from openai import OpenAI
from scipy.spatial import distance  # type: ignore

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# Create an OpenAI client and set your API key
client = OpenAI()


# added/edited
products = [
    { "title": "Smartphone X1", "short_description": "The latest flagship smartphone with AI-powered features and 5G connectivity.", "price": 799.99, "category": "Electronics", "features": [ "6.5-inch AMOLED display", "Quad-camera system with 48MP main sensor", "Face recognition and fingerprint sensor", "Fast wireless charging", ], },
    { "title": "Luxury Diamond Necklace", "short_description": "Elegant necklace featuring genuine diamonds, perfect for special occasions.", "price": 1499.99, "category": "Beauty", "features": [ "18k white gold chain", "0.5 carat diamond pendant", "Adjustable chain length", "Gift box included", ], },
    { "title": "RC Racing Car", "short_description": "High-speed remote-controlled racing car for adrenaline-packed fun.", "price": 89.99, "category": "Toys", "features": [ "Top speed of 30 mph", "Responsive remote control", "Rechargeable battery", "Durable construction", ], },
    { "title": "Ultra HD 4K TV", "short_description": "Immerse yourself in stunning visuals with this 65-inch 4K TV.", "price": 1299.99, "category": "Electronics", "features": [ "65-inch 4K UHD display", "Dolby Vision and HDR10+ support", "Smart TV with streaming apps", "Voice remote included", ], },
    { "title": "Glowing Skin Serum", "short_description": "Revitalize your skin with this nourishing serum for a radiant glow.", "price": 39.99, "category": "Beauty", "features": [ "Hyaluronic acid and vitamin C", "Hydrates and reduces fine lines", "Suitable for all skin types", "Cruelty-free", ], },
    { "title": "LEGO Space Shuttle", "short_description": "Build your own space adventure with this LEGO space shuttle set.", "price": 49.99, "category": "Toys", "features": [ "359 pieces for creative building", "Astronaut minifigure included", "Compatible with other LEGO sets", "For ages 7+", ], },
    { "title": "Wireless Noise-Canceling Headphones", "short_description": "Enjoy immersive audio and block out distractions with these headphones.", "price": 199.99, "category": "Electronics", "features": [ "Active noise cancellation", "Bluetooth 5.0 connectivity", "Long-lasting battery life", "Foldable design for portability", ], },
    { "title": "Luxury Perfume Gift Set", "short_description": "Indulge in a collection of premium fragrances with this gift set.", "price": 129.99, "category": "Beauty", "features": [ "Five unique scents", "Elegant packaging", "Perfect gift for fragrance enthusiasts", "Variety of fragrance notes", ], },
    { "title": "Remote-Controlled Drone", "short_description": "Take to the skies and capture stunning aerial footage with this drone.", "price": 299.99, "category": "Electronics", "features": [ "4K camera with gimbal stabilization", "GPS-assisted flight", "Remote control with smartphone app", "Return-to-home function", ], },
    { "title": "Luxurious Spa Gift Basket", "short_description": "Pamper yourself or a loved one with this spa gift basket full of relaxation goodies.", "price": 79.99, "category": "Beauty", "features": [ "Bath bombs, body lotion, and more", "Aromatherapy candles", "Reusable wicker basket", "Great for self-care", ], },
    { "title": "Robot Building Kit", "short_description": "Learn robotics and coding with this educational robot building kit.", "price": 59.99, "category": "Toys", "features": [ "Build and program your own robot", "STEM learning tool", "Compatible with Scratch and Python", "Ideal for young inventors", ], },
    { "title": "High-Performance Gaming Laptop", "short_description": "Dominate the gaming world with this powerful gaming laptop.", "price": 1499.99, "category": "Electronics", "features": [ "Intel Core i7 processor", "NVIDIA RTX graphics", "144Hz refresh rate display", "RGB backlit keyboard", ], },
    { "title": "Natural Mineral Makeup Set", "short_description": "Enhance your beauty with this mineral makeup set for a flawless look.", "price": 34.99, "category": "Beauty", "features": [ "Mineral foundation and eyeshadows", "Non-comedogenic and paraben-free", "Cruelty-free and vegan", "Includes makeup brushes", ], },
    { "title": "Interactive Robot Pet", "short_description": "Adopt your own robot pet that responds to your voice and touch.", "price": 79.99, "category": "Toys", "features": [ "Realistic pet behaviors", "Voice recognition and touch sensors", "Teaches responsibility and empathy", "Rechargeable battery", ], },
    { "title": "Smart Thermostat", "short_description": "Control your home's temperature and save energy with this smart thermostat.", "price": 129.99, "category": "Electronics", "features": [ "Wi-Fi connectivity", "Energy-saving features", "Compatible with voice assistants", "Easy installation", ], },
    { "title": "Designer Makeup Brush Set", "short_description": "Upgrade your makeup routine with this premium designer brush set.", "price": 59.99, "category": "Beauty", "features": [ "High-quality synthetic bristles", "Chic designer brush handles", "Complete set for all makeup needs", "Includes stylish carrying case", ], },
    { "title": "Remote-Controlled Dinosaur Toy", "short_description": "Roar into action with this remote-controlled dinosaur toy with lifelike movements.", "price": 49.99, "category": "Toys", "features": [ "Realistic dinosaur sound effects", "Walks and roars like a real dinosaur", "Remote control included", "Educational and entertaining", ], },
    { "title": "Wireless Charging Dock", "short_description": "Charge your devices conveniently with this sleek wireless charging dock.", "price": 39.99, "category": "Electronics", "features": [ "Qi wireless charging technology", "Supports multiple devices", "LED charging indicators", "Compact and stylish design", ], },
    { "title": "Luxury Skincare Set", "short_description": "Elevate your skincare routine with this luxurious skincare set.", "price": 179.99, "category": "Beauty", "features": [ "Premium anti-aging ingredients", "Hydrating and rejuvenating formulas", "Complete skincare regimen", "Elegant packaging", ], },
]


# Define a function to combine the relevant features into a single string
def create_product_text(product):
  return f"""Title: {product['title']}
Description: {product['short_description']}
Category: {product['category']}
Features: {'; '.join(product['features'])}"""

# Combine the features for each product
product_texts = [create_product_text(product) for product in products]

# Create the embeddings from product_texts
product_embeddings = create_embeddings(product_texts)



def create_embeddings(texts):
    response = client.embeddings.create(model="text-embedding-ada-002", input=texts)
    response_dict = response.model_dump()

    return [data["embedding"] for data in response_dict["data"]]


# Create the embeddings from product_texts
product_embeddings = create_embeddings(product_texts)


def find_n_closest(query_vector, embeddings, n=3):
    distances = []
    for index, embedding in enumerate(embeddings):
        # Calculate the cosine distance between the query vector and embedding
        dist = distance.cosine(query_vector, embedding)
        # Append the distance and index to distances
        distances.append({"distance": dist, "index": index})
    # Sort distances by the distance key
    distances_sorted = sorted(distances, key=lambda x: x["distance"])
    # Return the first n elements in distances_sorted
    return distances_sorted[0:n]


# Set your API key
client = OpenAI()

# Create the query vector from query_text
query_text = "computer"
query_vector = create_embeddings(query_text)[0]

# Find the five closest distances
hits = find_n_closest(query_vector, product_embeddings, n=5)

print(f'Search results for "{query_text}"')
for hit in hits:
    # Extract the product at each index in hits
    product = products[hit["index"]]
    print(product["title"])


# added/edited
products = [
    {
        "title": "Smartphone X1",
        "short_description": "The latest flagship smartphone with AI-powered features and 5G connectivity.",
        "price": 799.99,
        "category": "Electronics",
        "features": [
            "6.5-inch AMOLED display",
            "Quad-camera system with 48MP main sensor",
            "Face recognition and fingerprint sensor",
            "Fast wireless charging",
        ],
    },
    {
        "title": "Luxury Diamond Necklace",
        "short_description": "Elegant necklace featuring genuine diamonds, perfect for special occasions.",
        "price": 1499.99,
        "category": "Beauty",
        "features": [
            "18k white gold chain",
            "0.5 carat diamond pendant",
            "Adjustable chain length",
            "Gift box included",
        ],
    },
    {
        "title": "RC Racing Car",
        "short_description": "High-speed remote-controlled racing car for adrenaline-packed fun.",
        "price": 89.99,
        "category": "Toys",
        "features": [
            "Top speed of 30 mph",
            "Responsive remote control",
            "Rechargeable battery",
            "Durable construction",
        ],
    },
    {
        "title": "Ultra HD 4K TV",
        "short_description": "Immerse yourself in stunning visuals with this 65-inch 4K TV.",
        "price": 1299.99,
        "category": "Electronics",
        "features": [
            "65-inch 4K UHD display",
            "Dolby Vision and HDR10+ support",
            "Smart TV with streaming apps",
            "Voice remote included",
        ],
    },
    {
        "title": "Glowing Skin Serum",
        "short_description": "Revitalize your skin with this nourishing serum for a radiant glow.",
        "price": 39.99,
        "category": "Beauty",
        "features": [
            "Hyaluronic acid and vitamin C",
            "Hydrates and reduces fine lines",
            "Suitable for all skin types",
            "Cruelty-free",
        ],
    },
    {
        "title": "LEGO Space Shuttle",
        "short_description": "Build your own space adventure with this LEGO space shuttle set.",
        "price": 49.99,
        "category": "Toys",
        "features": [
            "359 pieces for creative building",
            "Astronaut minifigure included",
            "Compatible with other LEGO sets",
            "For ages 7+",
        ],
    },
    {
        "title": "Wireless Noise-Canceling Headphones",
        "short_description": "Enjoy immersive audio and block out distractions with these headphones.",
        "price": 199.99,
        "category": "Electronics",
        "features": [
            "Active noise cancellation",
            "Bluetooth 5.0 connectivity",
            "Long-lasting battery life",
            "Foldable design for portability",
        ],
    },
    {
        "title": "Luxury Perfume Gift Set",
        "short_description": "Indulge in a collection of premium fragrances with this gift set.",
        "price": 129.99,
        "category": "Beauty",
        "features": [
            "Five unique scents",
            "Elegant packaging",
            "Perfect gift for fragrance enthusiasts",
            "Variety of fragrance notes",
        ],
    },
    {
        "title": "Remote-Controlled Drone",
        "short_description": "Take to the skies and capture stunning aerial footage with this drone.",
        "price": 299.99,
        "category": "Electronics",
        "features": [
            "4K camera with gimbal stabilization",
            "GPS-assisted flight",
            "Remote control with smartphone app",
            "Return-to-home function",
        ],
    },
    {
        "title": "Luxurious Spa Gift Basket",
        "short_description": "Pamper yourself or a loved one with this spa gift basket full of relaxation goodies.",
        "price": 79.99,
        "category": "Beauty",
        "features": [
            "Bath bombs, body lotion, and more",
            "Aromatherapy candles",
            "Reusable wicker basket",
            "Great for self-care",
        ],
    },
    {
        "title": "Robot Building Kit",
        "short_description": "Learn robotics and coding with this educational robot building kit.",
        "price": 59.99,
        "category": "Toys",
        "features": [
            "Build and program your own robot",
            "STEM learning tool",
            "Compatible with Scratch and Python",
            "Ideal for young inventors",
        ],
    },
    {
        "title": "High-Performance Gaming Laptop",
        "short_description": "Dominate the gaming world with this powerful gaming laptop.",
        "price": 1499.99,
        "category": "Electronics",
        "features": [
            "Intel Core i7 processor",
            "NVIDIA RTX graphics",
            "144Hz refresh rate display",
            "RGB backlit keyboard",
        ],
    },
    {
        "title": "Natural Mineral Makeup Set",
        "short_description": "Enhance your beauty with this mineral makeup set for a flawless look.",
        "price": 34.99,
        "category": "Beauty",
        "features": [
            "Mineral foundation and eyeshadows",
            "Non-comedogenic and paraben-free",
            "Cruelty-free and vegan",
            "Includes makeup brushes",
        ],
    },
    {
        "title": "Interactive Robot Pet",
        "short_description": "Adopt your own robot pet that responds to your voice and touch.",
        "price": 79.99,
        "category": "Toys",
        "features": [
            "Realistic pet behaviors",
            "Voice recognition and touch sensors",
            "Teaches responsibility and empathy",
            "Rechargeable battery",
        ],
    },
    {
        "title": "Smart Thermostat",
        "short_description": "Control your home's temperature and save energy with this smart thermostat.",
        "price": 129.99,
        "category": "Electronics",
        "features": [
            "Wi-Fi connectivity",
            "Energy-saving features",
            "Compatible with voice assistants",
            "Easy installation",
        ],
    },
    {
        "title": "Designer Makeup Brush Set",
        "short_description": "Upgrade your makeup routine with this premium designer brush set.",
        "price": 59.99,
        "category": "Beauty",
        "features": [
            "High-quality synthetic bristles",
            "Chic designer brush handles",
            "Complete set for all makeup needs",
            "Includes stylish carrying case",
        ],
    },
    {
        "title": "Remote-Controlled Dinosaur Toy",
        "short_description": "Roar into action with this remote-controlled dinosaur toy with lifelike movements.",
        "price": 49.99,
        "category": "Toys",
        "features": [
            "Realistic dinosaur sound effects",
            "Walks and roars like a real dinosaur",
            "Remote control included",
            "Educational and entertaining",
        ],
    },
    {
        "title": "Wireless Charging Dock",
        "short_description": "Charge your devices conveniently with this sleek wireless charging dock.",
        "price": 39.99,
        "category": "Electronics",
        "features": [
            "Qi wireless charging technology",
            "Supports multiple devices",
            "LED charging indicators",
            "Compact and stylish design",
        ],
    },
    {
        "title": "Luxury Skincare Set",
        "short_description": "Elevate your skincare routine with this luxurious skincare set.",
        "price": 179.99,
        "category": "Beauty",
        "features": [
            "Premium anti-aging ingredients",
            "Hydrating and rejuvenating formulas",
            "Complete skincare regimen",
            "Elegant packaging",
        ],
    },
]
last_product = {
    "title": "Building Blocks Deluxe Set",
    "short_description": "Unleash your creativity with this deluxe set of building blocks for endless fun.",
    "price": 34.99,
    "category": "Toys",
    "features": [
        "Includes 500+ colorful building blocks",
        "Promotes STEM learning and creativity",
        "Compatible with other major brick brands",
        "Comes with a durable storage container",
        "Ideal for children ages 3 and up",
    ],
}

# Set your API key
client = OpenAI()

# Combine the features for last_product and each product in products
last_product_text = create_product_text(last_product)
product_texts = [create_product_text(product) for product in products]

# Embed last_product_text and product_texts
last_product_embeddings = create_embeddings(last_product_text)[0]
product_embeddings = create_embeddings(product_texts)

# Find the three smallest cosine distances and their indexes
hits = find_n_closest(last_product_embeddings, product_embeddings)

for hit in hits:
    product = products[hit["index"]]
    print(product["title"])


# added/edited
user_history = [
    {
        "title": "Remote-Controlled Dinosaur Toy",
        "short_description": "Roar into action with this remote-controlled dinosaur toy with lifelike movements.",
        "price": 49.99,
        "category": "Toys",
        "features": [
            "Realistic dinosaur sound effects",
            "Walks and roars like a real dinosaur",
            "Remote control included",
            "Educational and entertaining",
        ],
    },
    {
        "title": "Building Blocks Deluxe Set",
        "short_description": "Unleash your creativity with this deluxe set of building blocks for endless fun.",
        "price": 34.99,
        "category": "Toys",
        "features": [
            "Includes 500+ colorful building blocks",
            "Promotes STEM learning and creativity",
            "Compatible with other major brick brands",
            "Comes with a durable storage container",
            "Ideal for children ages 3 and up",
        ],
    },
]


# Set your API key
client = OpenAI()

# Prepare and embed the user_history, and calculate the mean embeddings
history_texts = [create_product_text(product) for product in user_history]
history_embeddings = create_embeddings(history_texts)
mean_history_embeddings = np.mean(history_embeddings, axis=0)

# Filter products to remove any in user_history
products_filtered = [product for product in products if product not in user_history]

# Combine product features and embed the resulting texts
product_texts = [create_product_text(product) for product in products_filtered]
product_embeddings = create_embeddings(product_texts)

hits = find_n_closest(mean_history_embeddings, product_embeddings)

for hit in hits:
    product = products_filtered[hit["index"]]
    print(product["title"])


# added/edited
sentiments = [{"label": "Positive"}, {"label": "Neutral"}, {"label": "Negative"}]
reviews = [
    "The food was delicious!",
    "The service was a bit slow but the food was good",
    "Never going back!",
]


# Set your API key
client = OpenAI()

# Create a list of class descriptions from the sentiment labels
class_descriptions = [sentiment["label"] for sentiment in sentiments]

# Embed the class_descriptions and reviews
class_embeddings = create_embeddings(class_descriptions)
review_embeddings = create_embeddings(reviews)


# Define a function to return the minimum distance and its index
def find_closest(query_vector, embeddings):
    distances = []
    for index, embedding in enumerate(embeddings):
        dist = distance.cosine(query_vector, embedding)
        distances.append({"distance": dist, "index": index})
    return min(distances, key=lambda x: x["distance"])


for index, review in enumerate(reviews):
    # Find the closest distance and its index using find_closest()
    closest = find_closest(review_embeddings[index], class_embeddings)
    # Subset sentiments using the index from closest
    label = sentiments[closest["index"]]["label"]
    print(f'"{review}" was classified as {label}')
