import streamlit as st
import google.generativeai as genai
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage

Google_API_Key='AIzaSyDbsl4puS8xddMH0avmIpQqkbJuB1RAhUs'
genai.configure(api_key=Google_API_Key)

model_name = 'gemini-1.5-flash'
model = genai.GenerativeModel(model_name=model_name)

def main():

    spacer, col = st.columns([5, 1])  
    with col:  
        st.image('groqcloud_darkmode.png')

    # The title and greeting message of the Streamlit application
    st.title("Welcome to my AI tool!")
    st.write("Let's start our conversation!")

    st.sidebar.title('Customization')
    system_prompt = st.sidebar.text_input("System prompt:")
    model_name = st.sidebar.selectbox(
        'Choose a model',
        [ 'llama-3.1-70b-versatile', 'llama-3.1-8b-instant', 'llama3-70b-8192', 'llama3-8b-8192']
    )
    conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value = 5)

    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

    user_question = st.text_area("Please ask a question:", height=200)

    # If the user has asked a question,
    if user_question:
        # session state variable
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history=[]
        else:
            for message in st.session_state.chat_history:
                memory.save_context(
                    {'input':message['human']},
                    {'output':message['AI']}
                    )

        # Construct a chat prompt template using various components
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate(
                    content=system_prompt
                ),  # This is the persistent system prompt that is always included at the start of the chat.

                MessagesPlaceholder(
                    variable_name="chat_history"
                ),  # This placeholder will be replaced by the actual chat history during the conversation. It helps in maintaining context.

                HumanMessagePromptTemplate.from_template(
                    "{human_input}"
                ),  # This template is where the user's current input will be injected into the prompt.
            ]
        )

        model = genai.GenerativeModel(model_name=model_name)

        try:
            response_google = model.generate_content(user_question,prompt=prompt).text
            message = {'human': user_question, 'AI': response_google}
            st.session_state.chat_history.append(message)

            # Display the conversation history in reverse order
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                if i == 0:
                    st.write(f"**Chatbot:** {chat['AI']}")
                else:
                    st.write(f"**User:** {chat['human']}")
                    st.write(f"**Chatbot:** {chat['AI']}")
                st.markdown("---")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()