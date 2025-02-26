from tensorflow_server import generate_response, rag_pipeline, corpus
import gradio as gr


# Define the function for generating responses
def medical_response(message, history):
    response = rag_pipeline(question=message, corpus=corpus)
    return response


theme = gr.themes.Base(
    primary_hue="fuchsia",
    secondary_hue="indigo",
    neutral_hue="stone",
)

title = "Medical Chat Assistant"
description = """
This assistant helps answer medical questions. 
Please enter your query in the chatbox below. 
For urgent concerns, always consult a professional.
"""

chat_interface = gr.ChatInterface(
    fn=medical_response,  # Function to handle the chat
    type="messages",  # Type of messages (e.g., text)
    title=title,  # Page title
    description=description,  # Subtitle or description
    theme=theme,  # Make the chatbox smaller using the "compact" theme
)

# Launch the app
chat_interface.launch(
    server_port=1111, share=True  # Custom server port  # Allow public sharing
)
