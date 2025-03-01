import streamlit as st
from agno.agent import Agent
from agno.models.mistral import MistralChat
from agno.tools.googlesearch import GoogleSearchTools

# Set page configuration
st.set_page_config(page_title="News Agent", page_icon="📰")

# Add title
st.title("News Search Agent")

# Store API key securely (in production, use st.secrets)
mistral_api_key = 'lL5l72aGJwQcUdc5EDRocZ3ts4SiWpuv'

# Initialize the agent
@st.cache_resource
def get_agent():
    return Agent(
        model=MistralChat(
            id="mistral-large-latest",
            api_key=mistral_api_key,
        ),
        tools=[GoogleSearchTools()],
        description="You are a news agent that helps users find the latest news.",
        instructions=[
            "Given a topic by the user, respond with 4 latest news items about that topic.",
            "Search for 10 news items and select the top 4 unique items.",
            "Search in English and in French.",
        ],
        show_tool_calls=True,
        debug_mode=True
    )

agent = get_agent()

# Create input field
topic = st.text_input("Enter a topic to search news about:", "")

# Add search button
if st.button("Search News"):
    if topic:
        with st.spinner('Searching for news...'):
            response = agent.get_response(topic)
            st.markdown(response)
    else:
        st.warning("Please enter a topic to search.")