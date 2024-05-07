# added/edited
import csv
import tiktoken


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
