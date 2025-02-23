import streamlit as st
import os
from groq import Groq

# Create a Groq client
def create_client():
    api_key = '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'
    return Groq(api_key=api_key)

# Get chat completion
def get_chat_completion(client, model, message):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": message,
            }
        ],
        model=model,
    )
    return chat_completion.choices[0].message.content

# Streamlit app
def main():
    st.title("Groq AI Chat Completion")
    st.write("Get answers from fast language models")

    # Create a Groq client
    client = create_client()

    # Get user input
    st.header("Enter a question")
    message = st.text_area("")

    # Get model selection
    st.header("Select a model")
    model_options = ["llama3-70b-8192","llama-3.3-70b-versatile"]
    model = st.selectbox("Model", options=model_options)

    # Get chat completion
    if st.button("Get Answer"):
        answer = get_chat_completion(client, model, message)
        st.write("Answer:")
        st.write(answer)

if __name__ == "__main__":
    main()