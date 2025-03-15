# PDF Question Answering with Retrieval-Augmented Generation (RAG)

## Overview
This project is a **PDF-based Question Answering System** built using **Streamlit**, **FAISS**, **HuggingFace Embeddings**, and **Llama 2**. It enables users to upload a PDF file, process its text, store embeddings in a vector database, and interactively ask questions about the content using **Retrieval-Augmented Generation (RAG)**.

## Features
- Upload a **PDF file**
- Extract text from the PDF
- Split text into **semantic chunks**
- Convert text chunks into **embeddings** using **HuggingFace models**
- Store embeddings in a **FAISS vector store**
- Load **Llama 2 (7B)** model using `LlamaCpp`
- Implement **Conversational Memory** for contextual chat
- Answer user queries related to the uploaded PDF

## Tech Stack
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Backend**:
  - **Llama 2** (via [LlamaCpp](https://github.com/ggerganov/llama.cpp))
  - **LangChain** (for LLM orchestration)
  - **FAISS** (for efficient vector search)
  - **HuggingFace Sentence Transformers** (for text embeddings)
  - **PyPDF2** (for PDF text extraction)

## Installation
### Prerequisites
- **Python 3.9+**
- **GPU (Recommended but Optional)**

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Install FAISS (if not included in `requirements.txt`)
```bash
pip install faiss-cpu  # For CPU users
pip install faiss-gpu  # If you have a CUDA-enabled GPU
```

## Usage
### 1. Run the Streamlit App
```bash
streamlit run app.py
```

### 2. Steps in the UI
1. **Upload a PDF**
2. **Process the PDF** (Extract text, create embeddings, store in FAISS)
3. **Ask questions** about the content
4. **Get responses** powered by Llama 2 with memory retention

## Configuration
### Update Llama 2 Model Path
Ensure the correct model file is set in `load_llm()`:
```python
model_path= <local path lo quantized Llama-2 model downloaded from Huggingface>
```
Modify it based on your directory structure.

## Requirements File
Ensure your `requirements.txt` includes:
```
streamlit==1.35.0
langchain==0.0.272
faiss-cpu==1.7.4  # Or faiss-gpu for CUDA users
sentence-transformers==2.2.2
llama-cpp-python==0.1.78
pypdf2==3.0.1
```


## Acknowledgments
This project was inspired by various RAG implementations and LangChain documentation.

