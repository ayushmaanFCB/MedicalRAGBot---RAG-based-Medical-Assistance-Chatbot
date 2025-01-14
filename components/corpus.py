import faiss
import numpy as np
import json

# Load embeddings and corpus
corpus_embeddings = np.load("data/embeddings/corpus_embeddings.npy")
with open("data/embeddings/corpus.json", "r") as f:
    corpus = json.load(f)

# Build FAISS index
index = faiss.IndexFlatL2(corpus_embeddings.shape[1])
index.add(corpus_embeddings)
faiss.write_index(index, "data/FAISS/faiss_index.bin")
