# added/edited
import os

import openai
from dotenv import load_dotenv
from openai import OpenAI

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
