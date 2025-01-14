import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM

# Load the model from .h5 file
model = TFAutoModelForSeq2SeqLM.from_pretrained("model")

try:
    # Save the model in the SavedModel format
    model.save("model/tf_server_deployment")
    print("Model Converted and Saved Successfully")
except Exception as e:
    print("Model Conversion Error: ", e)
