import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

with_message_history = RunnableWithMessageHistory(model, get_session_history)

st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")
st.title("🤖Chat with Urmi: Your Personal AI Companion")

st.markdown("""
    <style>
        .stTextInput>div>div>input {
            border-radius: 15px;
            padding: 10px;
            font-size: 1rem;
        }
        .stTextInput>div>div>input:focus {
            border-color: #4caf50;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 1rem;
        }
        .chat-bubble-user {
            background-color: #f1f1f1;
            border-radius: 15px;
            padding: 10px;
            margin-bottom: 10px;
        }
        .chat-bubble-ai {
            background-color: #007bff;
            color: white;
            border-radius: 15px;
            padding: 10px;
            margin-bottom: 10px;
        }
        .chat-container {
            max-height: 500px;
            overflow-y: scroll;
            margin-bottom: 20px;
            padding: 20px;
            background-color: #fafafa;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("Type your message here...", "")

if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))
    
    config = {"configurable": {"session_id": "chat1"}}
    
    response = with_message_history.invoke(
        [HumanMessage(content=user_input)],
        config=config
    )
    
    st.session_state.messages.append(AIMessage(content=response.content))

st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        st.markdown(f'<div class="chat-bubble-user">{message.content}</div>', unsafe_allow_html=True)
    elif isinstance(message, AIMessage):
        st.markdown(f'<div class="chat-bubble-ai">{message.content}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
