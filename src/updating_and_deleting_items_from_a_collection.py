import os

import chromadb
import openai
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]


# Create a persistant client
client = chromadb.PersistentClient()

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
