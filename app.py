import streamlit as st
import io
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Set up Streamlit page
st.set_page_config(page_title="PDF Question Answering with RAG")
st.header("PDF Question Answering with RAG")

@st.cache_resource
def load_llm():
    n_gpu_layers = 40  # Adjust based on your GPU's capabilities
    n_batch = 512  # Adjust based on your model and GPU's VRAM
    
    llm = LlamaCpp(
        model_path="D:/AI_ML_Code/GenAI_POCs/models/llama-2-7b-chat.Q2_K.gguf",  # Update this path
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        temperature=0,
        n_ctx=2048,
        verbose=True,
    )
    return llm

def process_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = load_llm()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = None

# Upload PDF file
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Process PDF and create conversation chain
    with st.spinner("Processing PDF..."):
        pdf_text = process_pdf(uploaded_file)
        text_chunks = get_text_chunks(pdf_text)
        vectorstore = get_vectorstore(text_chunks)
        st.session_state.conversation = get_conversation_chain(vectorstore)
    
    st.success("PDF processed successfully. You can now ask questions about its content.")

    # Chat interface
    if st.session_state.conversation:
        user_question = st.text_input("Ask a question about the PDF:")
        if user_question:
            with st.spinner("Thinking..."):
                response = st.session_state.conversation({'question': user_question})
                st.session_state.chat_history = response['chat_history']

        # Display chat history
        if st.session_state.chat_history:
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.write(f"Human: {message.content}")
                else:
                    st.write(f"AI: {message.content}")
else:
    st.info("Please upload a PDF file to begin.")

