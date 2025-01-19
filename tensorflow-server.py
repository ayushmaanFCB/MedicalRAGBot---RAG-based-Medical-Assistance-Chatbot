import tensorflow as tf
from sentence_transformers import SentenceTransformer
import json
import numpy as np
import requests
from components.response import retrieve_context, generate_response, embed_question
from transformers import T5Tokenizer, TFT5ForConditionalGeneration

# Load the fine-tuned model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("original_model")
# model = TFT5ForConditionalGeneration.from_pretrained("original_model")

# Load the retriever model
retriever_model = SentenceTransformer("all-MiniLM-L6-v2")

with open("data/embeddings/corpus.json", "r") as f:
    corpus = json.load(f)


def generate_response(question, retrieved_context):
    # Prepare the input
    input_text = f"Context: {retrieved_context} Question: {question}"

    # Tokenize the input text
    encoding = tokenizer(
        input_text, return_tensors="tf", padding=True, truncation=True, max_length=512
    )

    # Convert to lists for JSON serialization
    input_ids = encoding["input_ids"].numpy().tolist()
    attention_mask = encoding["attention_mask"].numpy().tolist()

    # For T5, decoder_input_ids can be the same as input_ids for inference
    decoder_input_ids = input_ids  # This is typically the same for inference
    decoder_attention_mask = attention_mask  # This is typically the same for inference

    # Prepare the payload for TensorFlow Serving
    payload = {
        "signature_name": "serving_default",
        "instances": [
            {
                "input_ids": input_ids[0],  # Use the first (and only) input
                "attention_mask": attention_mask[0],  # Include attention mask
                "decoder_input_ids": decoder_input_ids[0],  # Include decoder input ids
                "decoder_attention_mask": decoder_attention_mask[
                    0
                ],  # Include decoder attention mask
            }
        ],
    }

    # Send the request to TensorFlow Serving
    url = "http://localhost:8501/v1/models/fine_tuned_t5:predict"
    response = requests.post(url, json=payload)

    # Check for response errors
    if response.status_code != 200:
        raise Exception(
            f"Request failed with status code {response.status_code}: {response.text}"
        )

    # Get the output from the response
    output = response.json()

    # Print all keys in the output
    print("Response Keys:", output.keys())  # Print the keys in the response

    # Extract the predictions
    if "predictions" in output:
        predictions = output["predictions"]
        print("Predictions Keys:")

        # Iterate over each prediction and print its keys
        for i, prediction in enumerate(predictions):
            print(f"Keys in prediction {i}:", prediction.keys())

        # Assuming we want to decode the first prediction
        if predictions:
            first_prediction = predictions[0]

            # Check if 'logits' is present
            if "logits" in first_prediction:
                logits = first_prediction["logits"]

                # Get the top-k predicted token IDs for each position
                top_k = 5  # Change this to the number of responses you want
                top_k_indices = np.argsort(logits, axis=-1)[
                    :, -top_k:
                ]  # Get the indices of the top-k logits

                # Decode the top-k predicted token IDs into text
                all_responses = []
                for k in range(top_k):
                    predicted_ids = top_k_indices[:, k]  # Get the k-th top prediction
                    response_text = tokenizer.decode(
                        predicted_ids[0], skip_special_tokens=True
                    )
                    all_responses.append(response_text)

                # Print all generated responses
                print("Generated Responses:")
                for idx, response in enumerate(all_responses):
                    print(f"Response {idx + 1}: {response}")
            else:
                raise ValueError("Expected 'logits' key not found in predictions.")
    else:
        raise ValueError("Unexpected output format: 'predictions' key not found.")


def rag_pipeline(question, corpus, k=3):
    predictions = None

    # ncode the question
    question_embedding = embed_question(question, retriever_model).astype("float32")

    # Retrieve top-k contexts
    indices, distances = retrieve_context(question_embedding=question_embedding, k=k)
    retrieved_context = " ".join([corpus[i] for i in indices[0]])

    print(f"Retrieved Context: {retrieved_context}")

    generate_response(question=question, retrieved_context=retrieved_context)


if __name__ == "__main__":
    question = "I have mild Fever"
    rag_pipeline(question=question, corpus=corpus)
