import streamlit as st
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import os
import uuid
import warnings
warnings.filterwarnings('ignore')

# Setting up the LLM model
llm = Ollama(model="llama2")

# Create the temporary directory if it does not exist
if not os.path.exists("temp"):
    os.makedirs("temp")

# Defining the PDF GPT class


class PdfGpt():
    def __init__(self, uploaded_file):
        # Generate a unique file name for the uploaded PDF
        file_id = str(uuid.uuid4())
        file_path = os.path.join("temp", f"{file_id}.pdf")

        # Save the uploaded PDF to a temporary location
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Split the PDF into chunks for processing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=20)
        chunks = text_splitter.split_documents(
            documents=PyMuPDFLoader(file_path=file_path).load())

        # Initialize the embedding model
        embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Create a vector store for the chunks
        vectorstore = FAISS.from_documents(chunks, embedding_model)
        vectorstore.save_local("vectorstore")

        # Define the template for the Prompt
        template = """
        ### System:
        You are an expert financial analyst specializing in market forecasting, risk assessment, and sentiment analysis.
        You must answer user queries using only the context provided from financial reports. If the context is insufficient, 
        state 'I do not have enough data to answer'. Avoid making assumptions.

        ### Context:
        {context}

        ### User Query:
        {question}

        ### Answer:
        """


        # Initialize the Retrieval QA model
        self.hey = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={
                'prompt': PromptTemplate.from_template(template)}
        )


# Add futuristic CSS styling
st.title("PDF GPT 📄✨")


# Customized welcome message with a more visually engaging design
st.markdown(
    """
    <style>
    body {
        color: #f0f0f0; 
        background-color: #333344; 
        font-family: 'Helvetica', sans-serif;
    }
    h1 {
        color: #ccccff;
    }
    .welcome-message {
        background-color: #444466;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        transition-duration: 0.4s;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: white;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Enhanced welcome message
st.markdown(
    """
    <div class="welcome-message">
    <h2>Welcome, Consultants! 🌟</h2>
    <p>Here's how to make the most of our PDF GPT:</p>
    <ol>
        <li><strong>Upload PDFs:</strong> Select your document carefully (Financial Statements, Bulletins, etc.).</li>
        <li><strong>Ask clear questions:</strong> Get insightful responses.</li>
        <li><strong>Explore thoughtfully:</strong> Uncover valuable insights.</li>
        <li><strong>Embrace learning:</strong> Every question is an opportunity.</li>
    </ol>
    <p>Happy consulting!</p>
    </div>
    """,
    unsafe_allow_html=True
)
# File upload section
uploaded_file = st.file_uploader("Upload a PDF", type=[
                                 "pdf"], help="Please upload a PDF file.", key="file_uploader", accept_multiple_files=False)

# If a file is uploaded
if uploaded_file is not None:
    # Processing the PDF
    with st.spinner("Processing PDF..."):
        oracle = PdfGpt(uploaded_file)

    # User input section
    ask = st.text_input("What's up?", key="ask", label_visibility='hidden')

    # If the user input is not empty
    if ask not in [None, "", []]:
        # Displaying context and user input
        st.caption("📓")

        # Displaying prediction with expander
        with st.expander("🦙"):
            st.markdown(llm.predict(ask))

        # Generating response
        with st.spinner("Generating Response..."):
            response = oracle.hey({'query': ask})

        # Displaying response
        st.markdown(response['result'])
