# prepare_index.py
import pandas as pd
import openai
import json
import numpy as np
import faiss

openai.api_key = "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"
#openai.api_key = "Key"
# Load Excel file
#df = pd.read_excel("games_data.xlsx")
df = pd.read_excel("./NewInspired.xlsx")

# Combine fields for better embeddings
def combine_fields(row):
    return f"{row['Game Name']} by {row['Publisher']} - {row['Inspiration']}"

df['combined'] = df.apply(combine_fields, axis=1)

# Generate embeddings
def get_embedding(text):
    response = openai.Embedding.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

print("Generating embeddings...")
embeddings = [get_embedding(text) for text in df['combined']]
embedding_matrix = np.array(embeddings).astype("float32")

# Create FAISS index
print("Saving FAISS index...")
index = faiss.IndexFlatL2(embedding_matrix.shape[1])
index.add(embedding_matrix)
faiss.write_index(index, "games_index.faiss")

# Save game data as JSONL
print("Saving game metadata as JSONL...")
df[['Game Name', 'Publisher', 'Inspiration']].to_json("games_data.jsonl", orient="records", lines=True)

print("✅ Index and metadata saved successfully.")
