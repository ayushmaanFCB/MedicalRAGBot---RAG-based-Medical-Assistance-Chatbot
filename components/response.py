import faiss
import numpy as np
import json

# Load the saved FAISS index
index = faiss.read_index("faiss_index.bin")


# Encode the question to get its embedding
def embed_question(question, retriever_model):
    question_embedding = retriever_model.encode(question, convert_to_tensor=True)
    return question_embedding.cpu().numpy()


# Retrieve the top context
def retrieve_context(question_embedding, k=3):
    question_embedding = np.array([question_embedding]).astype(
        "float32"
    )  # Ensure it's 2D
    distances, indices = index.search(question_embedding, k)

    print(f"Distances: {distances}")
    print(f"Indices: {indices}")

    return indices, distances


def generate_response(question, retrieved_context, tokenizer, model):
    # Prepare the input
    input_text = f"Context: {retrieved_context} Question: {question}"

    # Tokenize and pass to the model
    input_ids = tokenizer(
        input_text, return_tensors="tf", padding=True, truncation=True, max_length=512
    ).input_ids
    outputs = model.generate(
        input_ids, max_length=200, num_beams=5, early_stopping=True
    )
    print(f"Model Output (Raw Tokens): {outputs}")

    print(f"Input Text: {input_text}")
    print(f"Input IDs: {input_ids}")

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def rag_pipeline(question, corpus, k=3):
    # Step 1: Encode the question
    question_embedding = embed_question(question).astype("float32")

    # Step 2: Retrieve top-k contexts
    indices, distances = retrieve_context(question_embedding, k)
    retrieved_context = " ".join([corpus[i] for i in indices[0]])

    print(f"Retrieved Context: {retrieved_context}")

    # Step 3: Generate response using fine-tuned T5
    response = generate_response(question, retrieved_context)
    return response
