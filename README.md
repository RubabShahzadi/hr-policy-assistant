# HR Policy Assistant

A RAG-based HR Policy Assistant built with Streamlit, FAISS, Sentence Transformers, and Groq.

## How It Works

1. Upload an HR policy PDF.
2. Extract text using PyMuPDF.
3. Split the document into token-aware chunks.
4. Generate embeddings using Sentence Transformers.
5. Store embeddings in a FAISS vector index.
6. Retrieve relevant policy chunks for the user's question.
7. Send the retrieved context to GPT-OSS through Groq.
8. Generate a grounded answer based only on the uploaded policy.

## Technologies

- Python
- Streamlit
- PyMuPDF
- Sentence Transformers
- FAISS
- Hugging Face Transformers
- Groq API
- GPT-OSS

## Deployment

This application is designed for Streamlit Community Cloud.

Add the following secret:

GROQ_API_KEY = "your_groq_api_key"

Never commit an API key to GitHub.
