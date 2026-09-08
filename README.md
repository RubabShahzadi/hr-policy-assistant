# 👩‍💼 HR Policy Assistant

An intelligent **Retrieval-Augmented Generation (RAG)** application that allows users to upload an HR Policy PDF and ask questions about its contents.

The application extracts information from the uploaded document, creates semantic embeddings, retrieves the most relevant policy sections using **FAISS**, and generates grounded answers using **GPT-OSS 20B through the Groq API**.

> **Live Demo:** [HR Policy Assistant](https://hr-policy-assistant-grnj8mbvhcenuchxybcvxx.streamlit.app/)

---

## 📌 Project Overview

The **HR Policy Assistant** is designed to provide accurate, document-grounded answers to HR-related questions.

Instead of relying only on the LLM's general knowledge, the application first searches the uploaded HR policy and provides the most relevant sections to the language model as context.

This helps reduce hallucinations and ensures that answers are based on the organization's uploaded HR policy.

---

## ✨ Features

* 📄 Upload HR Policy PDFs
* 🔍 Extract text from PDF documents
* ✂️ Token-aware document chunking
* 🧠 Generate semantic embeddings using Sentence Transformers
* 📚 Store and search embeddings using FAISS
* 🔎 Retrieve the most relevant policy sections
* 🤖 Generate grounded responses using GPT-OSS 20B
* 📑 Display retrieved policy sources and page numbers
* 📊 Display document statistics
* 🛡️ Prevent unsupported answers and hallucinations
* ☁️ Deployable on Streamlit Community Cloud

---

## 🔄 RAG Architecture

```text
HR Policy PDF
      ↓
PyMuPDF
      ↓
Text Extraction
      ↓
Text Cleaning
      ↓
Token-Aware Chunking
      ↓
Sentence Transformer Embeddings
      ↓
FAISS Vector Index
      ↓
User Question
      ↓
Question Embedding
      ↓
Similarity Search
      ↓
Top-K Relevant Policy Chunks
      ↓
Groq API
      ↓
GPT-OSS 20B
      ↓
Grounded HR Answer + Sources
```

---

## 🛠️ Technologies Used

| Technology                    | Purpose                   |
| ----------------------------- | ------------------------- |
| **Python**                    | Application development   |
| **Streamlit**                 | Web application interface |
| **PyMuPDF**                   | PDF text extraction       |
| **Sentence Transformers**     | Semantic embeddings       |
| **Hugging Face Transformers** | Tokenization              |
| **FAISS**                     | Vector similarity search  |
| **Groq API**                  | LLM inference             |
| **GPT-OSS 20B**               | Answer generation         |

### Embedding Model

```text
sentence-transformers/all-MiniLM-L6-v2
```

### Language Model

```text
openai/gpt-oss-20b
```

---

## ⚙️ How It Works

### 1. Upload PDF

The user uploads an HR Policy PDF through the Streamlit interface.

### 2. Extract Text

PyMuPDF extracts readable text from each PDF page.

### 3. Create Chunks

The extracted text is divided into smaller token-aware chunks so that relevant sections can be efficiently retrieved.

### 4. Generate Embeddings

Sentence Transformers converts each text chunk into a numerical vector representing its semantic meaning.

### 5. Build FAISS Index

The embeddings are stored in a FAISS vector index for efficient similarity search.

### 6. Retrieve Relevant Information

When the user asks a question, the question is converted into an embedding and compared against the document embeddings.

The application retrieves the **Top-K most relevant policy chunks**.

### 7. Generate Grounded Answer

The retrieved policy content is sent to **GPT-OSS 20B through Groq**.

The model is instructed to answer using only the retrieved HR policy context.

### 8. Display Sources

The application displays the retrieved policy sections, including their page numbers and similarity scores.

---

## 🛡️ Grounded Answering

The assistant is designed not to invent information.

If the requested information cannot be found in the uploaded HR Policy, the assistant responds:

> "I could not find this information in the uploaded HR policy."

This makes the application more reliable for document-based question answering.

---

## 📂 Project Structure

```text
hr-policy-assistant/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Deployment

The application can be deployed directly from GitHub using **Streamlit Community Cloud**.

### 1. Clone or Fork the Repository

Use the GitHub repository:

```text
https://github.com/RubabShahzadi/hr-policy-assistant
```

### 2. Configure the Groq API Key

In Streamlit Community Cloud, open:

**App → Settings → Secrets**

Add:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

### 3. Deploy

Configure:

```text
Repository: hr-policy-assistant
Branch: main
Main file: app.py
```

Then deploy the application.

> ⚠️ **Never commit your Groq API key to GitHub.**

---

## 🧪 Example Questions

After uploading an HR Policy PDF, try asking:

* How many annual leave days are employees entitled to?
* Can employees work remotely?
* What are the standard working hours?
* How many sick leave days are available?
* What is the resignation notice period?
* What is the employee salary?

You can also test the system with information that does not exist in the document to verify its grounded-answer behavior.

---

## 📄 Supported Documents

The application currently works best with **text-based PDF documents**.

Scanned or image-only PDFs may not contain extractable text and therefore may not work correctly with the current implementation.

---

## 🔐 Security

The Groq API key should be stored securely using **Streamlit Secrets**.

Do not place API keys directly inside `app.py` or commit them to GitHub.

The `.gitignore` file excludes common environment and secret files.

---

## 🎯 Use Case

This project demonstrates how **Retrieval-Augmented Generation (RAG)** can be used to build an AI assistant for organizational documents.

Potential applications include:

* HR policy assistants
* Employee handbooks
* Company policy search
* Internal documentation assistants
* Knowledge-base question answering
* Document-based AI assistants

---

## 👩‍💻 Author

**Rubab Shahzadi**

Full Stack Developer | AI Developer | Educator

GitHub: [@RubabShahzadi](https://github.com/RubabShahzadi)

---

## ⭐ Acknowledgment

This project was developed as a practical demonstration of **Generative AI, Retrieval-Augmented Generation, vector search, and LLM-based document question answering**.
