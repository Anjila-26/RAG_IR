# Retrieval Augmented Generation (RAG) Chat Interface

An intelligent chat interface that combines document processing with automated web scraping for enhanced context and responses using LangChain and Google's Gemini model.

## Features

- **Document Processing**: 
  - PDF document upload and processing
  - Automated text chunking and embedding
  - Intelligent context retrieval

- **Web Enhancement**:
  - Automatic keyword extraction from documents
  - Relevant web content integration
  - Source tracking and citation

- **Vector Storage Options**:
  - Local: ChromaDB for quick setup
  - Cloud: Pinecone for scalable deployments

- **Chat Interface**:
  - Clean, intuitive Streamlit interface
  - Context-aware responses
  - Source attribution with web links
  - Conversation history tracking

## Setup

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Configure API keys:
   - Set up Google API key for Gemini access
   - (Optional) Configure Pinecone credentials for cloud vector storage

3. Launch the application:
```bash
streamlit run app.py
```

## Usage

1. **Configuration**:
   - Input API keys in the sidebar
   - Choose vector storage type (Local/Pinecone)
   - Enable/disable web scraping feature

2. **Document Upload**:
   - Upload PDF documents through the interface
   - Click "Submit Documents" to process

3. **Chat Interaction**:
   - Ask questions about your documents
   - View AI responses with context
   - Explore related web sources in expandable sections

## Project Structure

```
RAG/
├── app.py              # Main application and UI
├── rag_backend.py      # Core RAG functionality
├── scraping_agents.py  # Web scraping and processing
├── data/
    ├── tmp/           # Temporary storage
    └── vector_store/  # Local vector database
```

## Technical Details

- **LLM**: Google Gemini 1.5 Flash
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Stores**: ChromaDB (local) or Pinecone (cloud)
- **Web Scraping**: Custom implementation with error handling and rate limiting
- **UI Framework**: Streamlit

## Requirements

- Python 3.8+
- langchain
- streamlit
- google-generative-ai
- sentence-transformers
- chromadb/pinecone-client
- beautifulsoup4
- PyPDF2

## Demo Video
https://github.com/Anjila-26/RAG_IR/blob/master/RAG.mp4