import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Function to extract text from PDFs
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip() if text else "No readable text found."

# Function to rank resumes
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    
    return cosine_similarities

# Streamlit UI Setup
st.set_page_config(page_title="AI Resume Screening & Ranking", layout="centered")

# Custom CSS for better styling
st.markdown("""
    <style>
        .stTable {
            border: 2px solid #E1E1E1;
            border-radius: 10px;
            background-color: white;
            padding: 10px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        }
        th {
            background-color: #FF4B4B !important;
            color: white !important;
            text-align: left !important;
            padding: 10px !important;
        }
        td {
            padding: 8px !important;
        }
        .big-font {
            font-size:20px !important;
            font-weight: bold !important;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h2 class='big-font'>📌 AI Resume Screening & Ranking</h2>", unsafe_allow_html=True)

# Input job description
job_description = st.text_area("🔹 Enter the Job Description")

# Upload resumes
uploaded_files = st.file_uploader("📂 Upload PDF Resumes", type=["pdf"], accept_multiple_files=True)

# Processing and ranking resumes
if uploaded_files and job_description:
    resumes = [extract_text_from_pdf(file) for file in uploaded_files]
    scores = rank_resumes(job_description, resumes)

    # Creating a DataFrame for better visualization
    ranked_resumes = sorted(zip(uploaded_files, scores), key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(
        [{"Rank": i+1, "Resume Name": file.name, "Score": round(score, 2)}
         for i, (file, score) in enumerate(ranked_resumes)]
    )

    # Display results
    st.subheader("📊 Ranked Resumes")
    st.dataframe(df.style.format({"Score": "{:.2f}"}))
