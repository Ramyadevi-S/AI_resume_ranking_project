import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip() if text else "No readable text found."

# Function to rank resumes based on similarity to job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes
    vectorizer = TfidfVectorizer().fit_transform(documents)
    vectors = vectorizer.toarray()
    
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    
    return cosine_similarities

# ----- Streamlit UI -----
st.set_page_config(page_title="AI Resume Screening", page_icon="📄", layout="wide")

# Sidebar Navigation
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3281/3281323.png", width=100)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📄 Resume Ranking"])

# Home Page
if page == "🏠 Home":
    st.title("Welcome to AI Resume Screening & Ranking System")
    st.markdown("""
        🔍 **This AI-powered app helps HR teams and recruiters quickly rank resumes based on job descriptions.**  
        - **Upload multiple PDF resumes** 📄  
        - **Enter a job description** ✍️  
        - **Get ranked results instantly** 🚀  
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/3048/3048122.png", width=300)

# Resume Ranking Page
elif page == "📄 Resume Ranking":
    st.title("📊 Resume Ranking System")
    job_description = st.text_area("📝 Enter the Job Description", height=150)

    uploaded_files = st.file_uploader("📂 Upload PDF resumes", type=["pdf"], accept_multiple_files=True)

    if uploaded_files and job_description:
        st.info("Processing resumes... Please wait ⏳")

        resumes = []
        progress_bar = st.progress(0)

        for i, file in enumerate(uploaded_files):
            resumes.append(extract_text_from_pdf(file))
            progress_bar.progress((i + 1) / len(uploaded_files))

        scores = rank_resumes(job_description, resumes)

        ranked_resumes = sorted(zip(uploaded_files, scores), key=lambda x: x[1], reverse=True)

        st.success("✅ Ranking completed!")
        st.subheader("📌 Ranked Resumes")
        


        # Display results in a table
        st.markdown("<h3>📊 Ranked Resumes</h3>", unsafe_allow_html=True)

        # Styled Markdown Table
        table_md = "| Rank | Resume Name | Score |\n|------|--------------|-------|\n"
        for i, (file, score) in enumerate(ranked_resumes, start=1):
            table_md += f"| {i} | {file.name} | {score:.2f} |\n"

        st.markdown(table_md)

       
        # Show top-ranked resume preview
        st.subheader("🏆 Top Ranked Resume")
        top_resume_text = extract_text_from_pdf(ranked_resumes[0][0])
        st.text_area("🔹 Top Resume Content", top_resume_text, height=300)

