import streamlit as st
from google import genai
from google.genai import types
import tempfile

def generate_response(image_file, prompt):
    try:
        client = genai.Client(
            api_key='AIzaSyDBvuL_-rHm8M9Vi-YOYqnbSs0Wcj3gVLA',
        )

        # Create a temporary file to upload
        with tempfile.NamedTemporaryFile(suffix='.jpeg') as tmp:
            tmp.write(image_file.getbuffer())
            tmp.seek(0)
            uploaded_file = client.files.upload(file=tmp.name)
            files = [uploaded_file]

        model = "gemini-2.0-flash"
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(
                        file_uri=files[0].uri,
                        mime_type=files[0].mime_type,
                    ),
                    types.Part.from_text(
                        text=prompt
                    ),
                ],
            ),
        ]

        generate_content_config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            response_mime_type="text/plain",
        )

        response = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            response += chunk.text

        return response

    except Exception as e:
        return f"An error occurred: {e}"

def main():
    st.set_page_config(layout="wide")
    st.title("Gemini Image-to-Text Generator")

    st.header("Upload an Image and Generate Text")

    with st.form("image_form"):
        image_file = st.file_uploader(
            "Upload an image file",
            type=['jpeg', 'jpg', 'png'],
            help="Upload an image file to generate text based on its content."
        )

        prompt = st.text_area(
            "Enter your prompt or description",
            placeholder="Describe the content of the image and explain its significance...",
            height=100
        )

        submit_button = st.form_submit_button(label='Generate')

    if submit_button and image_file:
        response = generate_response(image_file, prompt)
        st.header("Generated Response:")
        st.code(response)

if __name__ == "__main__":
    main()