import streamlit as st
import os
import pdfplumber
import re
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key="YOUR_GEMINI_API_KEY")

def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def extract_email(text):
    match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    return match.group(0) if match else None

def rank_candidates(resumes):
    prompt = """Rank the following resumes based on relevance to the job description, tech stack, skills, and work experience.
    Return a JSON list of top 20 ranked candidates with their scores.
    
    Resumes:
    """ + "\n\n".join([f"Resume {i+1}: {resume}" for i, resume in enumerate(resumes)])
    
    response = genai.GenerativeModel("gemini-pro").generate_content(prompt)
    return response.text  # Adjust based on API response format

st.title("AI-Powered Job Recruitment Automation")

uploaded_files = st.file_uploader("Upload Resumes (PDF only)", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    st.write("Processing Resumes...")
    resumes = [extract_text_from_pdf(pdf) for pdf in uploaded_files]
    
    ranked_candidates = rank_candidates(resumes)
    
    emails = [extract_email(resume) for resume in resumes if extract_email(resume)]
    
    st.subheader("Top 20 Candidates")
    st.json(ranked_candidates)
    
    st.subheader("Extracted Emails")
    st.write(emails)