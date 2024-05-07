# added/edited
import os

import openai
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# Create an OpenAI client and set your API key
client = OpenAI()

# added/edited
short_description = (
    "The latest flagship smartphone with AI-powered features and 5G connectivity."
)
list_of_descriptions = [
    "Charge your devices conveniently with this sleek wireless charging dock.",
    "Elevate your skincare routine with this luxurious skincare set.",
]


# Define a create_embeddings function
def create_embeddings(texts):
    response = client.embeddings.create(model="text-embedding-ada-002", input=texts)
    response_dict = response.model_dump()

    return [data["embedding"] for data in response_dict["data"]]


# Embed short_description and print
print(create_embeddings(short_description)[0])

# Embed list_of_descriptions and print
print(create_embeddings(list_of_descriptions))
