from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, HarmBlockThreshold, HarmCategory

load_dotenv()
llm = None


def get_llm_instance():
    """
    Method to return the instance of llm model globally
    :return:
    """
    global llm
    if llm is None:
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            stream=True,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
        )
    return llm