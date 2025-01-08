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
st.title("Retrieval Augmented Generation (RAG) with Langchain")


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


def clear_vector_store():
    try:
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        temp_db = Chroma(persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix(), embedding_function=embeddings)
        temp_db.delete_collection()
        del temp_db  # Release instance

        # Wait for OS to release file locks
        import time
        time.sleep(1)

        # Clean up directory contents silently
        for item in os.listdir(LOCAL_VECTOR_STORE_DIR):
            item_path = os.path.join(LOCAL_VECTOR_STORE_DIR, item)
            try:
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    import shutil
                    shutil.rmtree(item_path)
            except Exception:
                # Silently continue if file is locked
                pass

    except Exception as e:
        st.error(f"Error clearing vector store: {str(e)}")
        raise


def embeddings_on_local_vectordb(texts):
    try:
        st.info("Clearing existing vector database...")
        clear_vector_store()
        
        st.info(f"Creating new embeddings for {len(texts)} chunks...")
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Create new vector store with clean directory
        vectordb = Chroma.from_documents(
            texts, 
            embedding=embeddings,
            persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix()
        )
        
        # Persist and get retriever
        vectordb.persist()
        retriever = vectordb.as_retriever(search_kwargs={'k': 7})
        
        return retriever
    except Exception as e:
        st.error(f"Error in vector database creation: {str(e)}")
        raise


def embeddings_on_pinecone(texts):
    try:
        st.info("Initializing Pinecone...")
        pc = pinecone.Pinecone(
            api_key=st.session_state.pinecone_api_key,
            environment=st.session_state.pinecone_env
        )
        
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Get index
        index = pc.Index(st.session_state.pinecone_index)
        
        # Define a default namespace
        namespace = "default"
        
        # Try to delete existing vectors, handle case where namespace might not exist
        try:
            st.info("Clearing existing vectors...")
            index.delete(deleteAll=True, namespace=namespace)
        except Exception as e:
            # If namespace doesn't exist, that's fine - we'll create it when adding vectors
            st.info("No existing vectors to clear")
        
        st.info(f"Creating new embeddings for {len(texts)} chunks...")
        # Create new vectorstore with documents
        vectordb = LangchainPinecone.from_documents(
            documents=texts,
            embedding=embeddings,
            index_name=st.session_state.pinecone_index,
            namespace=namespace  # Use the same namespace here
        )
        
        retriever = vectordb.as_retriever(search_kwargs={'k': 7})
        return retriever
    except Exception as e:
        st.error(f"Error in Pinecone processing: {str(e)}")
        raise


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
    if not st.session_state.gemini_api_key or (st.session_state.pinecone_db and (not st.session_state.pinecone_api_key or not st.session_state.pinecone_env or not st.session_state.pinecone_index)) or not st.session_state.source_docs:
        st.warning("Please provide all required fields and upload documents.")
        return
        
    try:
        with st.spinner("Processing documents..."):
            # Reset the session state
            for key in ['retriever', 'db_initialized']:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Clear existing temporary files
            for file in TMP_DIR.iterdir():
                file.unlink()

            # Save and verify uploaded files
            total_docs = len(st.session_state.source_docs)
            st.info(f"Processing {total_docs} document(s)...")
            
            for source_doc in st.session_state.source_docs:
                temp_path = TMP_DIR / source_doc.name
                with open(temp_path, 'wb') as f:
                    f.write(source_doc.getbuffer())

            documents = load_documents()
            if not documents:
                st.error("No valid documents were processed")
                return

            texts = split_documents(documents)
            if len(texts) == 0:
                st.error("No valid text content extracted from documents")
                return

            # Reset retriever in session state
            if 'retriever' in st.session_state:
                del st.session_state.retriever

            if not st.session_state.pinecone_db:
                st.session_state.retriever = embeddings_on_local_vectordb(texts)
            else:
                st.session_state.retriever = embeddings_on_pinecone(texts)

            # Verify database was updated
            if hasattr(st.session_state, 'retriever') and st.session_state.retriever is not None:
                st.success(f"Successfully processed {len(texts)} text chunks!")
                st.session_state.db_initialized = True
            else:
                st.error("Failed to initialize retriever")
                st.session_state.db_initialized = False

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.session_state.db_initialized = False
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