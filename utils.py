import PyPDF2
import os
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

# PDF content into a variable
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Create a doc 
def create_document_from_text(text):
    return Document(page_content=text)


# Initialize Google Generative AI Embeddings
def initialize_gemini_embeddings():
    return GoogleGenerativeAIEmbeddings(
        api_key=os.getenv("GOOGLE_API_KEY")  # Replace with your Google API key
    )

# Create the QnA chain
def create_qa_chain(document):
    # Initialize embeddings model
    embeddings = initialize_gemini_embeddings()

    # Load the document into the vector store
    vector_store = FAISS.from_documents([document], embeddings)

    # Create the QnA chain using the vector store
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=embeddings,
        retriever=vector_store.as_retriever()
    )
    return qa_chain











