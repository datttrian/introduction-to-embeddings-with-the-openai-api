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


def create_embeddings(texts):
    response = client.embeddings.create(model="text-embedding-ada-002", input=texts)
    response_dict = response.model_dump()

    return [data["embedding"] for data in response_dict["data"]]


# Combine the features for each product
product_texts = [create_product_text(product) for product in products]

# Create the embeddings from product_texts
product_embeddings = create_embeddings(product_texts)
