# added/edited
import os

import numpy as np
import openai
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from openai import OpenAI
from scipy.spatial import distance  # type: ignore
from sklearn.manifold import TSNE  # type: ignore

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# Create an OpenAI client and set your API key
client = OpenAI()

# Create a request to obtain embeddings
response = client.embeddings.create(
    model="text-embedding-ada-002", input="This can contain any text."
)

# Convert the response into a dictionary
response_dict = response.model_dump()

print(response_dict)


# Extract the total_tokens from response_dict
print(response_dict["usage"]["total_tokens"])

# Extract the embeddings from response_dict
print(response_dict["data"][0]["embedding"])
