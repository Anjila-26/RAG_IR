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

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

You can configure the application using either environment variables or the Streamlit UI:

Required for basic functionality:
- Google API key for Gemini

Required for Pinecone integration:
- Pinecone API key
- Pinecone environment
- Pinecone index name

## Usage

1. Run the application:
```bash
streamlit run app.py
```

2. In the sidebar, enter your API keys if not configured through environment variables

3. Toggle 'Use Pinecone Vector DB' if you want to use Pinecone instead of local storage

4. Upload your PDF documents

5. Click "Submit Documents" to process and embed the documents

6. Start chatting with your documents through the interactive interface

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

- LangChain
- Google Gemini
- Streamlit
- Sentence Transformers
- PyPDF2
- Pinecone/Chroma DB

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.