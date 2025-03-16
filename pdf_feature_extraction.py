import streamlit as st

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Interview question generation prompt focusing on candidate skills, projects, and domain experience
interview_template = """
You are an interviewer tasked with generating domain-specific interview questions.
Based on the following resume content, focus on the candidate's skills, projects, and overall domain experience.
Generate 3 to 5 concise and insightful questions that assess the candidate's abilities and project experience.
Resume Content: {context}
Interview Questions:
"""

# Directory where PDFs will be saved
pdf_directory = 'C://Users//adhik//OneDrive//Desktop//DeepSeek r-1 RAG//'

# Initialize embeddings and vector store
embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
vector_store = InMemoryVectorStore(embeddings)

# Initialize the language model
model = OllamaLLM(model="deepseek-r1:1.5b")

def upload_pdf(file):
    with open(pdf_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()   
    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_splitter.split_documents(documents)

def index_docs(documents):
    vector_store.add_documents(documents)

def generate_interview_questions(documents):
    # Combine the resume content from all chunks
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(interview_template)
    chain = prompt | model
    return chain.invoke({"context": context})

# File uploader: Accept only PDF resumes
uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf", accept_multiple_files=False)

if uploaded_file:
    upload_pdf(uploaded_file)
    documents = load_pdf(pdf_directory + uploaded_file.name)
    chunked_documents = split_text(documents)
    index_docs(chunked_documents)
    
    st.write("Resume has been successfully processed. Click the button below to generate domain-relevant interview questions focusing on candidate skills and projects.")
    
    if st.button("Generate Interview Questions"):
        # Generate questions using the complete resume context
        questions = generate_interview_questions(chunked_documents)
        st.write("### Interview   :")
        st.write(questions)