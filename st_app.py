import streamlit as st
import requests

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'system_prompt' not in st.session_state:
    st.session_state.system_prompt = "You are a helpful assistant."

# Function to call the language model API
def call_language_model(prompt, messages):
    api_url = "https://127.0.0.1:8000/chat/invoke"
    headers = {
        #"Authorization": "Bearer YOUR_API_KEY",
        "Content-Type": "application/json"
    }
    data = {
        "system_prompt": prompt,
        "messages": messages
    }
    response = requests.post(api_url, headers=headers, json=data)
    return response.json()

# Title of the Streamlit app
st.title("Streamlit Chatbot")

# System prompt input
st.text_area("Edit System Prompt:", value=st.session_state.system_prompt, key="system_prompt")

# Display chat history
for msg in st.session_state.messages:
    st.write(f"{msg['role']}: {msg['content']}")

# User input
user_input = st.text_input("You:")

# Handle user input
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    response = call_language_model(st.session_state.system_prompt, st.session_state.messages)
    st.session_state.messages.append({"role": "assistant", "content": response['assistant_reply']})

    # Clear user input after sending
    st.text_input("You:", value="", key="user_input")

# Run the Streamlit app with:
# streamlit run your_script_name.py
