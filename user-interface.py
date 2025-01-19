from tensorflow_server import generate_response, rag_pipeline, corpus
import gradio as gr


def medical_response(message, history):
    response = rag_pipeline(question=message, corpus=corpus)
    return response


gr.ChatInterface(fn=medical_response, type="messages").launch(server_port=1111)
