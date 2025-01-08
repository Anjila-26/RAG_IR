# RAG (Retrieval Augmented Generation) Engine

A Streamlit-based application that implements Retrieval Augmented Generation using LangChain, allowing users to query their documents using Google's Gemini model with optional Pinecone vector database integration.

## Features

- PDF document processing and text extraction
- Vector embedding generation using SentenceTransformers
- Choice between local Chroma DB or Pinecone for vector storage
- Integration with Google's Gemini 1.5 Flash model
- Interactive chat interface with document-based responses
- Conversation history tracking

## Prerequisites

- Python 3.8+
- Google API key (Gemini)
- Pinecone API key (optional)
- Pinecone index must be created with:
  - Dimension: 384 (for SentenceTransformer all-MiniLM-L6-v2)
  - Metric: cosine
  - Pod type: p1.x1 (or your preferred pod type)

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

### API Keys
You can configure the application using either:
1. Streamlit secrets.toml:
```toml
gemini_api_key = "your-gemini-api-key"
pinecone_api_key = "your-pinecone-api-key"
pinecone_env = "your-pinecone-environment"
pinecone_index = "your-pinecone-index-name"
```

2. Or through the Streamlit UI sidebar

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Configure:
   - Enter API keys in sidebar if not using secrets.toml
   - Toggle 'Use Pinecone Vector DB' if you want to use Pinecone
   - Upload PDF documents

3. Process Documents:
   - Click "Submit Documents"
   - Wait for processing completion
   - Status messages will show progress

4. Query Documents:
   - Use the chat interface at the bottom
   - Type your questions about the documents
   - View conversation history above

## Project Structure

```
RAG/
├── app.py          # Main application file
├── data/           # Data directory
│   ├── tmp/        # Temporary storage for uploaded files
│   └── vector_store/ # Local vector store directory
└── README.md
```

## Technologies Used

- LangChain: Framework for LLM applications
- Google Gemini 1.5 Flash: Large Language Model
- Streamlit: Web interface
- SentenceTransformers: Document embeddings
- Chroma DB: Local vector storage
- Pinecone: Cloud vector database (optional)
- PyPDF2: PDF processing