import os
import tempfile
from pathlib import Path
import PyPDF2
from typing import List
import pinecone


from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma, Pinecone
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.vectorstores import Pinecone as LangchainPinecone

import streamlit as st

TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')

# Create necessary directories if they don't exist
TMP_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="RAG")
st.title("Retrieval Augmented Generation Engine")


def load_documents():
    documents = []
    for file_path in TMP_DIR.glob('**/*.pdf'):
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ''
                for page in pdf_reader.pages:
                    text += page.extract_text() + '\n'
                if text.strip():
                    from langchain_core.documents import Document
                    documents.append(Document(
                        page_content=text,
                        metadata={'source': file_path.name}
                    ))
        except Exception as e:
            st.error(f"Error loading file {file_path.name}: {str(e)}")
            continue
    return documents


def split_documents(documents):
    if not documents:
        raise ValueError("No valid documents to process")
    text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts


def embeddings_on_local_vectordb(texts):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(texts, embedding=embeddings,
                                     persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix())
    vectordb.persist()
    retriever = vectordb.as_retriever(search_kwargs={'k': 7})
    return retriever


def embeddings_on_pinecone(texts):
    # Initialize Pinecone
    pc = pinecone.Pinecone(
        api_key=st.session_state.pinecone_api_key,
        environment=st.session_state.pinecone_env
    )
    
    # Create embeddings
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create vectorstore using langchain's Pinecone wrapper
    vectordb = LangchainPinecone.from_existing_index(
        index_name=st.session_state.pinecone_index,
        embedding=embeddings,
        namespace="",  # Optional: specify a namespace if needed
    )
    
    retriever = vectordb.as_retriever()
    return retriever



def query_llm(retriever, query):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=st.session_state.gemini_api_key),
        retriever=retriever,
        return_source_documents=True,
    )
    result = qa_chain({'question': query, 'chat_history': st.session_state.messages})
    result = result['answer']
    st.session_state.messages.append((query, result))
    return result


def input_fields():
    with st.sidebar:
        if "gemini_api_key" in st.secrets:
            st.session_state.gemini_api_key = st.secrets.gemini_api_key
        else:
            st.session_state.gemini_api_key = st.text_input("Google API key", type="password")
        if "pinecone_api_key" in st.secrets:
            st.session_state.pinecone_api_key = st.secrets.pinecone_api_key
        else:
            st.session_state.pinecone_api_key = st.text_input("Pinecone API key", type="password")
        if "pinecone_env" in st.secrets:
            st.session_state.pinecone_env = st.secrets.pinecone_env
        else:
            st.session_state.pinecone_env = st.text_input("Pinecone environment")
        if "pinecone_index" in st.secrets:
            st.session_state.pinecone_index = st.secrets.pinecone_index
        else:
            st.session_state.pinecone_index = st.text_input("Pinecone index name")

    # Toggle for Pinecone DB option
    st.session_state.pinecone_db = st.checkbox('Use Pinecone Vector DB')

    st.session_state.source_docs = st.file_uploader(label="Upload Documents", type="pdf", accept_multiple_files=True)


def process_documents():
    if not st.session_state.gemini_api_key or not st.session_state.pinecone_api_key or not st.session_state.pinecone_env or not st.session_state.pinecone_index or not st.session_state.source_docs:
        st.warning(f"Please upload the documents and provide the missing fields.")
    else:
        try:
            # Initialize Pinecone if using it
            if st.session_state.pinecone_db:
                pc = pinecone.Pinecone(
                    api_key=st.session_state.pinecone_api_key,
                    environment=st.session_state.pinecone_env
                )
                # Verify index exists
                active_indexes = [index.name for index in pc.list_indexes()]
                if st.session_state.pinecone_index not in active_indexes:
                    st.error(f"Index {st.session_state.pinecone_index} does not exist in Pinecone")
                    return

            # Clear existing temporary files
            for file in TMP_DIR.iterdir():
                file.unlink()

            # Save uploaded files
            for source_doc in st.session_state.source_docs:
                temp_path = TMP_DIR / source_doc.name
                with open(temp_path, 'wb') as f:
                    f.write(source_doc.getbuffer())

            documents = load_documents()
            if not documents:
                st.error("No valid documents were processed")
                return

            texts = split_documents(documents)
            if not st.session_state.pinecone_db:
                st.session_state.retriever = embeddings_on_local_vectordb(texts)
            else:
                st.session_state.retriever = embeddings_on_pinecone(texts)

            st.success("Documents processed successfully!")

        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            # Cleanup temporary files
            for file in TMP_DIR.iterdir():
                file.unlink()


def boot():
    input_fields()
    st.button("Submit Documents", on_click=process_documents)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])

    if query := st.chat_input():
        st.chat_message("human").write(query)
        response = query_llm(st.session_state.retriever, query)
        st.chat_message("ai").write(response)


if __name__ == '__main__':
    boot()
