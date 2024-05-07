import os

import chromadb
import openai
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]


# Create a persistant client
client = chromadb.PersistentClient()

# Retrieve the netflix_titles collection
collection = client.get_collection(
    name="netflix_titles", embedding_function=OpenAIEmbeddingFunction()  # type: ignore
)

# Query the collection for "films about dogs"
result = collection.query(query_texts=["films about dogs"], n_results=3)

print(result)
