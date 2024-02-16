# import dependencies
import os
from PIL import Image
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)


# initialize gemini pro model
def initialize_model(model_name="gemini-pro-vision"):
    model = genai.GenerativeModel(model_name)
    return model

def get_response(model, model_behavior, image, prompt):
    response = model.generate_content([model_behavior, image[0], prompt])
    return response.text


def get_image_bytes(uploaded_image):
    if uploaded_image is not None:
        # read the uploaded image in bytes
        image_bytes = uploaded_image.getvalue()

        image_info = [
            {
            "mime_type": uploaded_image.type,
            "data": image_bytes
        }
        ]
        return image_info
    else:
        raise FileNotFoundError("Upload Valid image file!")


def show_response():
    model = initialize_model()

    # create the streamlit ui and get prompt along with image
    st.set_page_config("Invoice Extractor")
    st.header("Invoice Extractor")
    # Read teh prompt in text box
    prompt = st.text_input("Enter your prompt" ,key="prompt")
    # interface to upload image
    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        # Let's display the image
        st.image(image, caption="Your image", use_column_width=True)

    # let's create submit button, to submit image along with image
    submit = st.button("submit")

    # set the model behavior
    model_behavior = """
    Your are an expert who understand invoice overall structures and has deep knowledge on it.
    We will upload the invoice image and you have to answer the question bashed on information 
    present in the invoice image.
    """

    # if user pressed submit button
    if submit or prompt:
        if len(prompt) > 0:
            # get uploaded image file in bytes
            image_info = get_image_bytes(uploaded_image)
            response = get_response(model, model_behavior, image_info, prompt)
            st.write(response)
        else:
            raise ValueError("Please Enter Valid prompt!")


# call the function to show response ui
if __name__ == "__main__":
    show_response()



