# added/edited
import os

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


# Define a function to return the minimum distance and its index
def find_closest(query_vector, embeddings):
    distances = []
    for index, embedding in enumerate(embeddings):
        dist = distance.cosine(query_vector, embedding)
        distances.append({"distance": dist, "index": index})
    return min(distances, key=lambda x: x["distance"])


# added/edited
sentiments = [
    {"label": "Positive", "description": "A positive restaurant review"},
    {"label": "Neutral", "description": "A neutral restaurant review"},
    {"label": "Negative", "description": "A negative restaurant review"},
]
reviews = [
    "The food was delicious!",
    "The service was a bit slow but the food was good",
    "Never going back!",
]

# Extract and embed the descriptions from sentiments
class_descriptions = [sentiment["description"] for sentiment in sentiments]
class_embeddings = create_embeddings(class_descriptions)
review_embeddings = create_embeddings(reviews)


for index, review in enumerate(reviews):
    closest = find_closest(review_embeddings[index], class_embeddings)
    label = sentiments[closest["index"]]["label"]
    print(f'"{review}" was classified as {label}')
