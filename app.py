import tensorflow as tf
from sentence_transformers import SentenceTransformer
import json
from components.response import retrieve_context, generate_response, embed_question
from transformers import T5Tokenizer, TFT5ForConditionalGeneration

# Load the fine-tuned model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("original_model")
model = TFT5ForConditionalGeneration.from_pretrained("original_model")

# Load the retriever model
retriever_model = SentenceTransformer("all-MiniLM-L6-v2")

with open("data/embeddings/corpus.json", "r") as f:
    corpus = json.load(f)


def rag_pipeline(question, corpus, k=3):
    # Step 1: Encode the question
    question_embedding = embed_question(question, retriever_model).astype("float32")

    # Step 2: Retrieve top-k contexts
    indices, distances = retrieve_context(question_embedding, k)
    retrieved_context = " ".join([corpus[i] for i in indices[0]])

    print(f"Retrieved Context: {retrieved_context}")

    # Step 3: Generate response using fine-tuned T5
    response = generate_response(question, retrieved_context, tokenizer, model)
    return response


if __name__ == "__main__":
    question = "Stomach Issue Symptoms?"

    response = rag_pipeline(question, corpus)
    print("Generated Response:", response)
