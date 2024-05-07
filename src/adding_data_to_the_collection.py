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

ids = []
documents = []

with open("netflix_titles_1000.csv") as csvfile:
    reader = csv.DictReader(csvfile)
    for i, row in enumerate(reader):
        ids.append(row["show_id"])
        text = f"Title: {row['title']} ({row['type']})\nDescription: {row['description']}\nCategories: {row['listed_in']}"
        documents.append(text)


# added/edited
client.delete_collection("netflix_titles")


# Recreate the netflix_titles collection
collection = client.create_collection(
    name="netflix_titles", embedding_function=OpenAIEmbeddingFunction()  # type: ignore
)

# Add the documents and IDs to the collection
collection.add(ids=ids, documents=documents)

# Print the collection size and first ten items
print(f"No. of documents: {collection.count()}")
print(f"First ten documents: {collection.peek()}")
