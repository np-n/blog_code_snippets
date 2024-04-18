import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from utils.load_llm import get_llm_instance


def get_response(user_query, conversation_history):
    """
    Method to return the response using the streaming chain
    :param user_query:
    :param conversation_history:
    :return:
    """
    prompt_template = f"""
    You are a AI assistant. Answer the following question considering the history of the conversation:
    
    Chat history: {conversation_history}
    
    User question: {user_query}
    """
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    llm = get_llm_instance()
    expression_language_chain = prompt | llm | StrOutputParser()

    # note: use .invoke() method for non-streaming
    return expression_language_chain.stream(
        {
            "conversation_history": conversation_history,
            "user_query": user_query
        }
    )
