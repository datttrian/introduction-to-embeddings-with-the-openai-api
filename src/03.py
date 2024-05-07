# added/edited
import csv
import os

import chromadb
import openai
import tiktoken
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

# Load the encoder for the OpenAI text-embedding-ada-002 model
enc = tiktoken.encoding_for_model("text-embedding-ada-002")

# Encode each text in documents and calculate the total tokens
total_tokens = sum(len(enc.encode(text)) for text in documents)

cost_per_1k_tokens = 0.0001

# Display number of tokens and cost
print("Total tokens:", total_tokens)
print("Cost:", cost_per_1k_tokens * total_tokens / 1000)


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


# Retrieve the netflix_titles collection
collection = client.get_collection(
    name="netflix_titles", embedding_function=OpenAIEmbeddingFunction()  # type: ignore
)

# Query the collection for "films about dogs"
result = collection.query(query_texts=["films about dogs"], n_results=3)

print(result)


# added/edited
new_data = [
    {
        "id": "s1001",
        "document": "Title: Cats & Dogs (Movie)\nDescription: A look at the top-secret, high-tech espionage war going on between cats and dogs, of which their human owners are blissfully unaware.",
    },
    {
        "id": "s6884",
        "document": 'Title: Goosebumps 2: Haunted Halloween (Movie)\nDescription: Three teens spend their Halloween trying to stop a magical book, which brings characters from the "Goosebumps" novels to life.\nCategories: Children & Family Movies, Comedies',
    },
]


# Retrieve the netflix_titles collection
collection = client.get_collection(
    name="netflix_titles", embedding_function=OpenAIEmbeddingFunction()  # type: ignore
)

# Update or add the new documents
collection.upsert(
    ids=[doc["id"] for doc in new_data], documents=[doc["document"] for doc in new_data]
)

# Delete the item with ID "s95" and re-run the query
collection.delete(ids=["s95"])

result = collection.query(query_texts=["films about dogs"], n_results=3)
print(result)


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
