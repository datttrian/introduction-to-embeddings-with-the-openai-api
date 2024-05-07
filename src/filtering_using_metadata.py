# added/edited
import csv
import os

import chromadb
import openai
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]


# Create a persistant client
client = chromadb.PersistentClient()

# added/edited
ids = []
metadatas = []
documents = []

with open("netflix_titles_1000.csv") as csvfile:
    reader = csv.DictReader(csvfile)
    for i, row in enumerate(reader):
        # ids
        ids.append(row["show_id"])

        # metadatas
        metadata = {"rating": row["rating"], "release_year": int(row["release_year"])}
        metadatas.append(metadata)

        # documents
        text = f"Title: {row['title']} ({row['type']})\nDescription: {row['description']}\nCategories: {row['listed_in']}"
        documents.append(text)

# Recreate the netflix_titles collection
client.delete_collection("netflix_titles")
collection = client.create_collection(
    name="netflix_titles", embedding_function=OpenAIEmbeddingFunction()  # type: ignore
)

# Add the documents and IDs to the collection
collection.add(ids=ids, documents=documents, metadatas=metadatas)  # type: ignore

# Retrieve the netflix_titles collection
collection = client.get_collection(
    name="netflix_titles", embedding_function=OpenAIEmbeddingFunction()  # type: ignore
)

reference_texts = ["children's story about a car", "lions"]

# Query two results using reference_texts
result = collection.query(
    query_texts=reference_texts,
    n_results=2,
    # Filter for titles with a G rating released before 2019
    where={"$and": [{"rating": {"$eq": "G"}}, {"release_year": {"$lt": 2019}}]},  # type: ignore
)

print(result["documents"])
