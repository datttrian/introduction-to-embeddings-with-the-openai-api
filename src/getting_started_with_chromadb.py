# added/edited
import os

import chromadb
import openai
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]


# Create a persistant client
client = chromadb.PersistentClient()

# Create a netflix_title collection using the OpenAI Embedding function
collection = client.create_collection(
    name="netflix_titles", embedding_function=OpenAIEmbeddingFunction()  # type: ignore
)

# List the collections
print(client.list_collections())
