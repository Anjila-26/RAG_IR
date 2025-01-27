import os
import tempfile
from pathlib import Path
import PyPDF2
from typing import List
import pinecone

from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma, Pinecone as LangchainPinecone
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document

TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')

# Create directories if they don't exist
TMP_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

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
                    documents.append(Document(
                        page_content=text,
                        metadata={'source': file_path.name}
                    ))
        except Exception as e:
            raise RuntimeError(f"Error loading file {file_path.name}: {str(e)}")
    return documents

def split_documents(documents):
    if not documents:
        raise ValueError("No valid documents to process")
    text_splitter = CharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
    return text_splitter.split_documents(documents)

def clear_vector_store():
    try:
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        temp_db = Chroma(persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix(), embedding_function=embeddings)
        temp_db.delete_collection()
        del temp_db

        import time
        time.sleep(1)

        for item in os.listdir(LOCAL_VECTOR_STORE_DIR):
            item_path = os.path.join(LOCAL_VECTOR_STORE_DIR, item)
            try:
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    import shutil
                    shutil.rmtree(item_path)
            except Exception:
                pass
    except Exception as e:
        raise RuntimeError(f"Error clearing vector store: {str(e)}")

def embeddings_on_local_vectordb(texts):
    try:
        clear_vector_store()
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = Chroma.from_documents(
            texts, 
            embedding=embeddings,
            persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix()
        )
        vectordb.persist()
        return vectordb.as_retriever(search_kwargs={'k': 7})
    except Exception as e:
        raise RuntimeError(f"Error in local vector database creation: {str(e)}")

def embeddings_on_pinecone(texts, pinecone_api_key, pinecone_env, pinecone_index):
    try:
        pc = pinecone.Pinecone(
            api_key=pinecone_api_key,
            environment=pinecone_env
        )
        
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        index = pc.Index(pinecone_index)
        namespace = "default"
        
        try:
            index.delete(deleteAll=True, namespace=namespace)
        except Exception:
            pass
        
        vectordb = LangchainPinecone.from_documents(
            documents=texts,
            embedding=embeddings,
            index_name=pinecone_index,
            namespace=namespace
        )
        
        return vectordb.as_retriever(search_kwargs={'k': 7})
    except Exception as e:
        raise RuntimeError(f"Error in Pinecone processing: {str(e)}")

def get_llm_chain(retriever, google_api_key):
    return ConversationalRetrievalChain.from_llm(
        llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key),
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
        max_tokens_limit=8000,
        get_chat_history=lambda h: h,
    )