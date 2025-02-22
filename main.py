import os
import gradio as gr
import google.generativeai as genai
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Configure generation settings
generation_config = {
    "temperature": 0.15,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Initialize chatbot model
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
    system_instruction="You are a highly knowledgeable medical chatbot specializing in brain tumors. \n"
                       "You must provide detailed, factual, and to-the-point answers based on verified medical knowledge. \n"
                       "If you do not know the answer, simply respond with: \"I don't know the answer yet.\"\n"
)

# Store chat history
chat_history = [
    {"role": "user", "parts": ["hello"]},
    {"role": "model", "parts": ["Hello! How can I help you today with your questions about brain tumors?"]},
]

# Start chat session
chat_session = model.start_chat(history=chat_history)

# Function to generate chatbot response
def chatbot_response(user_input, history):
    global chat_session

    # Send message to chatbot
    response = chat_session.send_message(user_input)

    # Append to history
    history.append((user_input, response.text))

    return "", history

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("#  Brain Tumor Chatbot")
    gr.Markdown("Ask me anything about brain tumors!")

    chatbot = gr.Chatbot()
    user_input = gr.Textbox(label="Your question:")
    
    with gr.Row():
        submit_button = gr.Button("Ask")
        clear_button = gr.Button("Clear")

    submit_button.click(chatbot_response, inputs=[user_input, chatbot], outputs=[user_input, chatbot])
    clear_button.click(lambda: [], outputs=[chatbot])

# Run Gradio app
import os

def start():
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))

