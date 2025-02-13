import streamlit as st
import os
import requests
from bs4 import BeautifulSoup
import time
import re
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from transformers import AutoTokenizer

# 配置
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
DEFAULT_URL = 'https://www.phirda.com/artilce_37852.html'
SEQUENCE_FILE = 'sequence.txt'
MAX_EMPTY_RETRIES = 2  # 设置最大重试次数为2次

# Import config after defining constants that config might use
import config

# Initialize tokenizer globally
try:
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
except Exception as e:
    st.error(f"Error loading tokenizer: {e}")
    tokenizer = None # Make sure tokenizer is None if it fails to load

def get_sequence():
    if os.path.exists(SEQUENCE_FILE):
        with open(SEQUENCE_FILE, 'r') as f:
            try:
                return int(f.read())
            except ValueError:
                st.error(f"Invalid sequence number in {SEQUENCE_FILE}. Resetting to 0.")
                return 0
    else:
        return 0

def save_sequence(number):
    with open(SEQUENCE_FILE, 'w') as f:
        f.write(str(number))

def send_request(url, headers):
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = 'utf-8'
        return response
    except requests.RequestException as e:
        st.error(f"请求错误：{e}")
        return None

def parse_page(response):
    try:
        soup = BeautifulSoup(response.text, 'html.parser')
        news_items = soup.find_all('div', class_='news_main')
        if not news_items:
            return None
        return news_items
    except Exception as e:
        st.error(f"Error parsing page: {e}")
        return None

def extract_info(news_items):
    news_list = []
    for item in news_items:
        try:
            # 提取标题
            title = item.find('dt').text.strip() if item.find('dt') else "无标题"

            # 提取发布时间
            publish_time_img = item.find_next('img', src="/images/ico4.png")  # Find the img tag
            if publish_time_img:
                publish_time_i = publish_time_img.find_parent('i')  # Go to the parent <i> tag
                if publish_time_i:
                    publish_time = publish_time_i.text.replace(publish_time_img.decode(), '').strip() # Extract text AND remove the image
                else:
                    publish_time = "无发布时间 (<i> tag not found)"
            else:
                publish_time = "无发布时间 (<img tag not found)"

            # 提取文章内容
            content = item.find_next('div', class_='xq_con')
            if content:
                content_text = content.get_text(strip=True)
            else:
                content_text = "无正文"

            news_list.append({
                'title': title,
                'publish_time': publish_time,
                'content': content_text
            })
        except Exception as e:
            st.error(f"Error extracting info from news item: {e}")
            continue
    return news_list

def chunk_text(text, tokenizer, max_tokens):
    """Splits text into chunks based on token count."""
    tokens = tokenizer.encode(text, return_attention_mask=False)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i + max_tokens]
        chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))  # Decode each chunk
    return chunks

def query_llm(user_question, groq_chat, system_prompt, memory):
    """Queries the LLM with the given question and returns the response."""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{human_input}")
    ])

    conversation = LLMChain(
        llm=groq_chat,
        prompt=prompt,
        verbose=False,
        memory=memory
    )

    try:
        answer = conversation.predict(human_input=user_question)
        return answer
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return "Error generating answer."

def summarize_article(article_content, groq_chat, system_prompt, tokenizer, max_chunk_tokens=4000):
    """Summarizes a long article by chunking and combining summaries."""

    if not tokenizer:
        st.error("Tokenizer failed to load. Cannot summarize article.")
        return "Error: Tokenizer unavailable."

    # Chunk the article
    chunks = chunk_text(article_content, tokenizer, max_chunk_tokens)
    chunk_summaries = []

    # Summarize each chunk
    for i, chunk in enumerate(chunks):
        prompt = f"Summarize this section of the article: {chunk}"
        try:
            chunk_summary = query_llm(prompt, groq_chat, system_prompt, None)  # Use temporary memory
            chunk_summaries.append(chunk_summary)
        except Exception as e:
            st.error(f"Error summarizing chunk {i+1}: {e}")
            return "Error summarizing article"

    # Combine the summaries
    combined_summary_prompt = f"Combine these summaries into a single, concise summary of the entire article:\n" + "\n".join(chunk_summaries)
    try:
        final_summary = query_llm(combined_summary_prompt, groq_chat, system_prompt, None)  # Or memory if needed
        return final_summary
    except Exception as e:
        st.error(f"Error creating final summary: {e}")
        return "Error summarizing article"


def main():
    """Main function to run the Streamlit app."""

    # Get Groq API key
    groq_api_key = config.GROQ_API_KEY

    # Display the Groq logo
    spacer, col = st.columns([5, 1])
    with col:
        st.image('groqcloud_darkmode.png')

    # The title and greeting message of the Streamlit application
    st.title("Welcome to my AI tool!")  # Added Title
    st.header("Let's start our conversation!")

    # Add customization options to the sidebar
    st.sidebar.title('Customization')
    system_prompt = st.sidebar.text_area("System prompt:", value="You are a helpful assistant.")
    model = st.sidebar.selectbox(
        'Choose a model',
        ['deepseek-r1-distill-llama-70b', 'gemma2-9b-it', 'llama-3.1-8b-instant', 'llama3-70b-8192', 'llama3-8b-8192']
    )

    # Initialize Groq Langchain chat object
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model
    )

    # Initialize conversation memory in Streamlit's session state
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # User input area
    user_question = st.text_area("Please ask a question or enter your query here:", height=200)

    # LLM Query Functionality - Immediate Response
    if user_question:
        with st.spinner("Generating response..."):
            llm_response = query_llm(user_question, groq_chat, system_prompt, st.session_state.memory)
            st.subheader("LLM Response:")
            st.write(llm_response)

    # Crawler Functionality - Button Activated
    with st.sidebar:
        crawl_button = st.button("News Summary")  # Outside of the form

    if crawl_button:
        headers = {'User-Agent': USER_AGENT}
        st.info("Starting to crawl news articles...")
        current_number = get_sequence()
        url = re.sub(r'\d+', str(current_number), DEFAULT_URL)
        empty_retries = 0

        while True:
            st.info(f"Requesting: {url}")
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

            if not news_list:
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

            for article in news_list:
                st.write(f"title: {article['title']}")
                st.write(f"content: {article['content']}")
                with st.spinner("Summarizing articles..."):
                    final_summary = summarize_article(article['content'], groq_chat, system_prompt, tokenizer)
                    st.subheader("Article Summary:")
                    st.write(final_summary)

            # Save current number and prepare next URL
            save_sequence(current_number + 1)
            current_number += 1
            url = re.sub(r'\d+', str(current_number), DEFAULT_URL)
            st.info("Moving to next sequence number...")
            time.sleep(1)

if __name__ == "__main__":
    main()