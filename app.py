import streamlit as st
import os
from groq import Groq
import random

from langchain.chains import ConversationChain, LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate


def get_groq_api_key():
    return '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'

def display_groq_logo():
    spacer, col = st.columns([5, 1])  
    with col:  
        st.image('groqcloud_darkmode.png')

def display_title_and_greeting():
    st.write("Hello! Let's start our conversation!")

def add_customization_options():
    st.sidebar.title('Customization')
    system_prompt = st.sidebar.text_input("System prompt:")
    model = st.sidebar.selectbox(
        'Choose a model',
        ['llama3-8b-8192','llama3-70b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']
    )
    conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value = 5)
    return system_prompt, model, conversational_memory_length

def create_memory(conversational_memory_length):
    return ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

def get_user_question():
    return st.text_area("Ask a question:",height=200)

def initialize_groq_chat(groq_api_key, model_name):
    return ChatGroq(groq_api_key=groq_api_key, model_name=model_name)

def construct_chat_prompt_template(system_prompt):
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=system_prompt
            ),
            MessagesPlaceholder(
                variable_name="chat_history"
            ),
            HumanMessagePromptTemplate.from_template(
                "{human_input}"
            ),
        ]
    )

def create_conversation_chain(llm, prompt, memory):
    return LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)

def handle_user_input(user_question, conversation):
    response = conversation.predict(human_input=user_question)
    message = {'human':user_question,'AI':response}
    st.session_state.chat_history.append(message)
    st.write("Chatbot:", response)

def main():
    groq_api_key = get_groq_api_key()
    display_groq_logo()
    display_title_and_greeting()
    system_prompt, model, conversational_memory_length = add_customization_options()
    memory = create_memory(conversational_memory_length)
    user_question = get_user_question()
    groq_chat = initialize_groq_chat(groq_api_key, model)
    prompt = construct_chat_prompt_template(system_prompt)
    conversation = create_conversation_chain(groq_chat, prompt, memory)
    if user_question:
        handle_user_input(user_question, conversation)

if __name__ == "__main__":
    main()




