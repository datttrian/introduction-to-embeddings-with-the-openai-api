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

reference_ids = ["s999", "s1000"]

# Retrieve the documents for the reference_ids
reference_texts = collection.get(ids=reference_ids)["documents"]

# Query using reference_texts
result = collection.query(query_texts=reference_texts, n_results=3)

print(result["documents"])
