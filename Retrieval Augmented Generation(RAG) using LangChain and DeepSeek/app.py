"""
--------------------------------------------------
Author  : Netra Prasad Neupane
Date    : 7/22/25
Purpose : 
--------------------------------------------------
"""

import streamlit as st
import tempfile
from main import configure_document
from main import get_response


st.set_page_config(page_title="Document Chatbot", page_icon="🤖", layout="wide")
st.write("<h2 style='text-align: center;'>Chat with Documents 📑</h2>", unsafe_allow_html=True)
st.write("\n\n")

# sidebar
st.sidebar.write("Upload your document below:")
st.sidebar.info("If document is already uploaded, then you can directly chat with document without uploading.")
uploaded_file = st.sidebar.file_uploader("Upload document(supports `.pdf` format only)", type="pdf")

index_name = "document-index"
index_path = "./index"

if uploaded_file:
    if st.sidebar.button("Upload"):
        # Get the data related with document
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Create a placeholder in the sidebar
        with st.sidebar:
            spinner_placeholder = st.empty()

        with spinner_placeholder:
            with st.status('Uploading...'):
                response = configure_document(document_path=tmp_path, index_name=index_name, index_path=index_path)

                # Check if the request was successful
                if response["status"]:
                    st.sidebar.success(response["message"])
                else:
                    st.sidebar.error(response["message"])

        # Clear the spinner
        spinner_placeholder.empty()


st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50; /* Green background */
        color: white;
        border: none;
        padding: 10px 30px;
        text-align: center;
        text-decoration: none;
        display: inline-block; /* Get elements to line up properly */
        font-size: 20px;
        font-weight: bold;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 9px;
    }
    .stButton>button:hover {
        background-color: #45a049; /* Darker green on hover */
        color: white;
    }
    .stButton>button:active {
        background-color: #4CAF50; /* Color after click */
        color: white;
    }
    .stButton>button:inactive {
        background-color: #4CAF50; /* Color after click */
        color: white;
    }

    .stButton>download_button {
        background-color: #008CBA; /* Initial blue background */
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
    }

    .stButton>download_button:hover {
        background-color: #005f73; /* Darker blue on hover */
    }

    </style>
    """, unsafe_allow_html=True)




# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []


# Display chat messages from history on app rerun
for message in st.session_state["messages"]:
    if message["role"] == "Error":
        st.error(message["content"])
    else:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Accept user input
prompt = st.chat_input("Ask Anything")
if (prompt is not None) and (prompt != ""):
    # Add user message to chat history
    st.session_state["messages"].append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)


    with st.spinner("generating answer..."):
        # get vectorstore
        response = get_response(index_name=index_name, index_path=index_path,query=prompt)

        if response["status"]:
            with st.chat_message("assistant"):
                st.write(response["answer"])
            # Add assistant message to chat history
            st.session_state["messages"].append({"role": "assistant", "content": response["answer"]})
        else:
            if "message" in response.keys():
                st.error(response["message"])

                # Add error message to chat history
                st.session_state["messages"].append(
                    {"role": "Error", "content": response["message"]})
            else:
                st.error("Exception occurred!")
                st.session_state["messages"].append(
                    {"role": "Error", "content": "Exception occurred!"})



