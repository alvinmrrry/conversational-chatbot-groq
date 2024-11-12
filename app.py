import streamlit as st
import google.generativeai as genai

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

    # Add customization options to the sidebar
    st.sidebar.title('Customization')
    conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value = 5)

    # session state variable
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history=[]
    else:
        pass

    user_question = st.text_area("Please ask a question:",height=200)

    # If the user has asked a question,
    if user_question:

        try:
            response_google = model.generate_content(user_question)
            message = {'human':user_question,'AI':response_google}
            st.session_state.chat_history.append(message)

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