import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
import time
import re
import random
from langchain.chains import ConversationChain, LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# 配置
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
DEFAULT_URL = 'https://www.phirda.com/artilce_37852.html'
SEQUENCE_FILE = 'sequence.txt'
MAX_EMPTY_RETRIES = 2  # 设置最大重试次数为5次

def get_sequence():
    if os.path.exists(SEQUENCE_FILE):
        with open(SEQUENCE_FILE, 'r') as f:
            return int(f.read())
    else:
        return 0

def save_sequence(number):
    with open(SEQUENCE_FILE, 'w') as f:
        f.write(str(number - MAX_EMPTY_RETRIES + 1))

def send_request(url, headers):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = 'utf-8'
        return response
    except requests.RequestException as e:
        print(f"请求错误：{e}")
        return None

def parse_page(response):
    soup = BeautifulSoup(response.text, 'html.parser')
    news_items = soup.find_all('div', class_='news_main')
    if not news_items:
        return None
    return news_items

def extract_info(news_items):
    news_list = []
    for item in news_items:
        title_tag = item.find('h2', class_='news-title')
        title = title_tag.text.strip() if title_tag and title_tag.text else "无标题"

        content_tag = item.find('p', class_='news-content')
        content = content_tag.text.strip() if content_tag and content_tag.text else "无正文"

        news_list.append({
            'title': title,
            'content': content
        })
    return news_list

def main():
    """
    This function is the main entry point of the application. It sets up the Groq client, the Streamlit interface, and handles the chat interaction.
    """

    # Get Groq API key
    groq_api_key = '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'

    # Display the Groq logo
    spacer, col = st.columns([5, 1])  
    with col:  
        st.image('groqcloud_darkmode.png')

    # The title and greeting message of the Streamlit application
    st.title("Welcome to my AI tool!")
    st.write("Let's start our conversation!")

    # Add customization options to the sidebar
    st.sidebar.title('Customization')
    system_prompt = st.sidebar.text_input("System prompt:")
    model = st.sidebar.selectbox(
        'Choose a model',
        [ 'deepseek-r1-distill-llama-70b','gemma2-9b-it', 'llama-3.1-8b-instant', 'llama3-70b-8192', 'llama3-8b-8192']
    )

    # User input area
    st.header("News Crawler and Summarizer")
    user_question = st.text_area("Please ask a question:", height=200)

    # Initialize Groq Langchain chat object and conversation
    groq_chat = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name=model
    )

    # Create a form for the crawl button
    with st.form("crawler_form"):
        crawl_button = st.form_submit_button("Start Crawling and Summarizing")

    if crawl_button:
        st.info("Starting to crawl news articles...")
        current_number = get_sequence()
        url = re.sub(r'\d+', str(current_number), DEFAULT_URL)
        empty_retries = 0  # 初始化空页面重试计数器

        while True:
            st.info(f"Requesting: {url}")
            headers = {'User-Agent': USER_AGENT}
            response = send_request(url, headers)

            if not response or response.status_code != 200:
                st.warning(f"Failed to get page: {url}, status code: {response.status_code if response else 'No response'}")
                empty_retries += 1
                if empty_retries >= MAX_EMPTY_RETRIES:
                    st.error(f"Stopped after {MAX_EMPTY_RETRIES} consecutive empty pages.")
                    save_sequence(current_number)
                    break
                else:
                    st.info(f"Retrying with next sequence number: {current_number + 1}")
                    current_number += 1
                    url = re.sub(r'\d+', str(current_number), DEFAULT_URL)
                    continue

            news_items = parse_page(response)
            if not news_items:
                st.warning(f"No news items found on page: {url}")
                empty_retries += 1
                if empty_retries >= MAX_EMPTY_RETRIES:
                    st.error(f"Stopped after {MAX_EMPTY_RETRIES} consecutive empty pages.")
                    save_sequence(current_number)
                    break
                else:
                    current_number += 1
                    url = re.sub(r'\d+', str(current_number), DEFAULT_URL)
                    continue
            else:
                empty_retries = 0

            news_list = extract_info(news_items)
            total = len(news_list)

            if total == 0:
                st.warning("No news articles found. Trying next sequence number.")
                empty_retries += 1
                if empty_retries >= MAX_EMPTY_RETRIES:
                    st.error(f"Stopped after {MAX_EMPTY_RETRIES} consecutive empty pages.")
                    save_sequence(current_number)
                    break
                else:
                    current_number += 1
                    url = re.sub(r'\d+', str(current_number), DEFAULT_URL)
                    continue
            else:
                empty_retries = 0

            # Display each article and its summary
            st.subheader(f"Found {total} articles:")
            for i, article in enumerate(news_list):
                st.subheader(f"Article {i+1}: {article['title']}")
                st.write(f"Content: {article['content'][:300]}...")  # Show first 300 characters

                # Prepare the prompt for summarization
                prompt = ChatPromptTemplate.from_messages([
                    SystemMessage(content=system_prompt),
                    HumanMessagePromptTemplate.from_template("{human_input}")
                ])

                # Create conversation chain
                conversation = LLMChain(
                    llm=groq_chat,
                    prompt=prompt,
                    verbose=True
                )

                # Generate summary
                summary = conversation.predict(human_input=article['content'])
                st.write(f"Summary: {summary}")
                st.markdown("---")

            # Save current number and prepare next URL
            save_sequence(current_number)
            current_number += 1
            url = re.sub(r'\d+', str(current_number), DEFAULT_URL)
            st.info("Moving to next sequence number...")
            time.sleep(1)

if __name__ == "__main__":
    main()