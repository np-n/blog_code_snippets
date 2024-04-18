import streamlit as st
from utils.chain import get_conversation_chain

# let's  create the streamlit app
st.set_page_config(page_title=" Conversational Bot!")
st.title("Gemini Chatbot 💬")

# initialize the messages key in streamlit session to store message history
if "messages" not in st.session_state:
    # add greeting message to user
    st.session_state.messages = [{"role": "assistant", "content": "Hi I am a Bot, How can I help you?"}]

# if there are messages already in session, write them on app
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

prompt = st.chat_input("Say Something")

if prompt is not None and prompt != "":
    # add the message to chat message container
    if st.session_state.messages[-1]["role"] != "user":
        st.session_state.messages.append({"role": "user", "content": prompt})
        # display to the streamlit application
        message = st.chat_message("user")
        message.write(f"{prompt}")

    llm, conversational_chain = get_conversation_chain()
    response = conversational_chain.predict(input=prompt)

    if st.session_state.messages[-1]["role"] != "assistant":
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)
