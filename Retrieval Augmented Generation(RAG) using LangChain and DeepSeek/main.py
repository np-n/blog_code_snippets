import faiss
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from dotenv import load_dotenv
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
import os
import traceback

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class IndexProcessingError(Exception):
    """Raised when FAISS index processing fails."""
    pass

class RetrievalError(Exception):
    """Raised when configuring document to the vector database."""
    pass


def get_load_document(file_path):
    loader = PyPDFLoader(file_path)
    pages = []
    for page in loader.lazy_load():
        pages.append(page)

    return pages


def split_document(pages):
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators = ["\n\n", "."],
        is_separator_regex="\n\n",
    )
    chunks = text_splitter.split_documents(pages)

    return chunks


def get_embedding_model():
    model_name = "BAAI/bge-small-en-v1.5"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}

    hf_embedding = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

    return hf_embedding


def store_to_vectorstore(index_name, chunks, hf_embedding, index_path="./index", ):
    # If index doesn't exist then create the index
    try:
        if not (os.path.exists(os.path.join(index_path, f"{index_name}.faiss")) and os.path.exists(os.path.join(index_path, f"{index_name}.pkl"))):
            index = faiss.IndexFlatL2(len(hf_embedding.embed_query("hello world")))
            vector_store = FAISS(
                embedding_function=hf_embedding,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
        else:
            vector_store = FAISS.load_local(folder_path=index_path, embeddings=hf_embedding, index_name=index_name, allow_dangerous_deserialization=True)

        # Now, index is exist, it's time to insert the data to the index
        vector_store.add_documents(documents=chunks)
        vector_store.save_local(folder_path=index_path, index_name=index_name)

    except Exception as e:
        print(traceback.format_exc())
        raise IndexProcessingError(f"Exception occurred while creating index : {e}")


def get_vectorstore_as_retriever(index_name, index_path="./index"):
    try:
        embeddings = get_embedding_model()
        vector_store = FAISS.load_local(folder_path=index_path, index_name=index_name, embeddings=embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever()

        return retriever
    except Exception as e:
        print(traceback.format_exc())
        raise RetrievalError(f"Exception occurred while configuring retriever: {e}")



def get_llm_instance():
    llm = ChatGroq(
        model="deepseek-r1-distill-llama-70b",
        temperature=0,
        max_tokens=None,
        reasoning_format="parsed",
        timeout=None,
        max_retries=2,
        api_key=GROQ_API_KEY,
        # other params...
    )

    return llm



def get_rag_chain(retriever, llm, question):
    template = """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.
    
    Question: {question}
    Context: {context}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever,  "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


    return rag_chain


def configure_document(document_path, index_name, index_path):
    try:
        pages = get_load_document(document_path)
        chunks = split_document(pages)
        hf_embedding_model = get_embedding_model()
        store_to_vectorstore(chunks=chunks, hf_embedding=hf_embedding_model, index_name=index_name, index_path=index_path)
        return {"status": True, "message": "Document ingested successfully, now you can perform the QA over the document."}

    except Exception as e:
        print(f"Failed to configure the document: {e}")
        print(traceback.format_exc())
        return {"status": False, "message": f"Failed to configure the document: {e}"}



def get_response(index_name, index_path, query):
    try:
        retriever = get_vectorstore_as_retriever(index_name=index_name, index_path=index_path)
        llm = get_llm_instance()
        rag_chain = get_rag_chain(retriever, llm, query)
        response = rag_chain.invoke(query)

        return {"status": True, "answer":response, "message":"successfully generated the response for the query."}

    except Exception as e:
        print(f"Exception occurred : {e}")
        print(traceback.format_exc())
        return {"status": False, "message":f"Exception occurred : {e}"}





if __name__ == "__main__":
    document_path = "/home/netra/Downloads/Build-a-Large-Language-Model-From-Scratch-MEAP-V01-Sebastian-Raschka-Z-Library.pdf"
    pages = get_load_document(document_path)
    chunks = split_document(pages)
    hf_embedding_model = get_embedding_model()

    index_name = "document-index"
    index_path = "./index"
    store_to_vectorstore(chunks=chunks, hf_embedding=hf_embedding_model,index_name=index_name, index_path=index_path )
    retriever = get_vectorstore_as_retriever(index_name=index_name, index_path=index_path)
    llm = get_llm_instance()
    query = "What is llm?"
    rag_chain = get_rag_chain(retriever, llm, query)
    response = rag_chain.invoke(query)
    print(response)

