import streamlit as st
from pathlib import Path
from rag_backend import (
    TMP_DIR, LOCAL_VECTOR_STORE_DIR,
    load_documents, split_documents,
    embeddings_on_local_vectordb, embeddings_on_pinecone,
    get_llm_chain
)
from scraping_agents import WebScraper

st.set_page_config(page_title="RAG", layout="wide")

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

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

    st.session_state.pinecone_db = st.checkbox('Use Pinecone Vector DB')
    st.session_state.enable_scraping = st.checkbox('Enable Web Scraping')
    st.session_state.source_docs = st.file_uploader(label="Upload Documents", type="pdf", accept_multiple_files=True)

def process_documents():
    try:
        if not st.session_state.gemini_api_key or (st.session_state.pinecone_db and 
            (not st.session_state.pinecone_api_key or not st.session_state.pinecone_env or not st.session_state.pinecone_index)) or not st.session_state.source_docs:
            raise ValueError("Please provide all required fields and upload documents")

        with st.spinner("Processing documents..."):
            for key in ['retriever', 'db_initialized']:
                st.session_state.pop(key, None)

            for file in TMP_DIR.iterdir():
                file.unlink()

            total_docs = len(st.session_state.source_docs)
            st.info(f"Processing {total_docs} document(s)...")
            
            for source_doc in st.session_state.source_docs:
                temp_path = TMP_DIR / source_doc.name
                with open(temp_path, 'wb') as f:
                    f.write(source_doc.getbuffer())

            documents = load_documents()
            if not documents:
                raise ValueError("No valid documents were processed")

            # Web scraping integration
            if st.session_state.enable_scraping:
                scraper = WebScraper(num_results=3)
                web_documents = scraper.process_documents(documents)
                documents.extend(web_documents)

            texts = split_documents(documents)
            if not texts:
                raise ValueError("No valid text content extracted from documents")

            if st.session_state.pinecone_db:
                st.session_state.retriever = embeddings_on_pinecone(
                    texts,
                    st.session_state.pinecone_api_key,
                    st.session_state.pinecone_env,
                    st.session_state.pinecone_index
                )
            else:
                st.session_state.retriever = embeddings_on_local_vectordb(texts)

            if not st.session_state.retriever:
                raise RuntimeError("Failed to initialize retriever")

            st.session_state.qa_chain = get_llm_chain(
                st.session_state.retriever,
                st.session_state.gemini_api_key
            )
            st.session_state.db_initialized = True
            st.success(f"Successfully processed {len(texts)} text chunks!")

    except Exception as e:
        st.error(str(e))
        st.session_state.db_initialized = False
    finally:
        for file in TMP_DIR.iterdir():
            file.unlink()

def main():
    st.title("RAG Chat Interface")
    initialize_session_state()
    
    input_fields()
    st.button("Submit Documents", on_click=process_documents)
    
    # Display chat messages
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])
    
    # Chat input
    if query := st.chat_input():
        st.chat_message("human").write(query)
        try:
            if 'qa_chain' not in st.session_state:
                raise RuntimeError("Please process documents first")
            
            result = st.session_state.qa_chain({
                'question': query, 
                'chat_history': st.session_state.messages
            })
            response = result['answer']
            st.session_state.messages.append((query, response))
            
            if result.get('source_documents'):
                with st.expander("Related Web Links"):
                    for doc in result['source_documents']:
                        if doc.metadata.get('source') == 'web':
                            st.markdown(f"- [{doc.metadata['url']}]({doc.metadata['url']})")
            
            st.chat_message("ai").write(response)
        except Exception as e:
            st.error(str(e))

if __name__ == '__main__':
    main()