import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the dataset
with open("data/jsons/HealthCareMagic-100k.json", "r") as f:
    data = json.load(f)

# Create a corpus of inputs (questions and symptoms descriptions)
corpus = [entry["input"] for entry in data]
answers = [entry["output"] for entry in data]

# Embed the corpus using SentenceTransformer for retrieval
retriever_model = SentenceTransformer("all-MiniLM-L6-v2")
corpus_embeddings = retriever_model.encode(corpus, convert_to_tensor=True)

# Save the embeddings and corpus for later use
np.save("corpus_embeddings.npy", corpus_embeddings.numpy())
with open("corpus.json", "w") as f:
    json.dump(corpus, f)
