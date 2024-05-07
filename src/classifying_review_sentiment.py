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
