import os
import streamlit as st
from utils.streaming_chain import get_response
from langchain_core.messages import AIMessage, HumanMessage


# let's  create the streamlit app
st.set_page_config(page_title=" Conversational Bot!")
st.title("Gemini Chatbot 💬")

# initialize the messages key in streamlit session to store message history
if "messages" not in st.session_state:
    # add greeting message to user
    st.session_state.messages = [
        AIMessage(content="Hello, I am a bot. How can I help you?")
    ]

# if there are messages already in session, write them on app
for message in st.session_state.messages:
    if isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)

prompt = st.chat_input("Say Something")

if prompt is not None and prompt != "":
    # add the message to chat message container
    if not isinstance(st.session_state.messages[-1], HumanMessage):
        st.session_state.messages.append(HumanMessage(content=prompt))
        # display to the streamlit application
        message = st.chat_message("user")
        message.write(f"{prompt}")

    if not isinstance(st.session_state.messages[-1], AIMessage):
        with st.chat_message("assistant"):
            # use .write() method for non-streaming, which means .invoke() method in chain
            response = st.write_stream(get_response(prompt, st.session_state.messages))
        st.session_state.messages.append(AIMessage(content=response))
