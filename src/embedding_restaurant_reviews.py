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


def create_embeddings(texts):
    response = client.embeddings.create(model="text-embedding-ada-002", input=texts)
    response_dict = response.model_dump()

    return [data["embedding"] for data in response_dict["data"]]

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
