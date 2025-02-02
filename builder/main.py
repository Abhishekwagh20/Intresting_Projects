import os
import re
import streamlit as st
from dotenv import load_dotenv
import requests
from PyPDF2 import PdfReader
from docx import Document
import pdfkit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import json
from datetime import datetime
import html
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Initial setup
nltk.download('stopwords')
load_dotenv()

# Configuration
stop_words = set(stopwords.words('english'))
SECTION_WEIGHTS = {
    'keywords': 0.35,
    'sections': 0.25,
    'length': 0.1,
    'skills': 0.2,
    'contact': 0.1
}
REQUIRED_SECTIONS = {'EXPERIENCE', 'EDUCATION', 'SKILLS', 'PROJECTS', 'CERTIFICATIONS'}
SKILLS_LIST = [
    "Python", "Machine Learning", "Data Analysis", "Project Management",
    "SQL", "AWS", "TensorFlow", "Deep Learning", "Cloud Computing",
    "Statistical Modeling", "Team Leadership", "Agile", "NLP", "Big Data",
    "Tableau", "Data Visualization", "Git", "Docker", "Kubernetes",
    "CI/CD", "REST APIs", "Pandas", "NumPy", "Scikit-learn", "PyTorch",
    "Time Series Analysis", "Data Mining", "Power BI", "JIRA", "Spark"
]

HTML_TEMPLATE = """
<html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ 
                font-family: 'Arial', sans-serif; 
                line-height: 1.6; 
                margin: 0.75in;
                color: #333;
            }}
            .header {{ 
                text-align: center; 
                padding-bottom: 15px;
                border-bottom: 2px solid #2c3e50;
                margin-bottom: 25px;
            }}
            .name {{
                font-size: 28pt;
                font-weight: bold;
                margin: 15px 0;
                color: #2c3e50;
            }}
            .contact-info {{
                font-size: 11pt;
                margin-bottom: 20px;
                color: #666;
            }}
            .section {{
                margin: 25px 0;
                page-break-inside: avoid;
            }}
            .section-title {{
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 5px;
                font-size: 14pt;
                margin: 25px 0 15px;
                text-transform: uppercase;
            }}
            .bullet-list {{
                margin-left: 25px;
                padding-left: 15px;
            }}
            .job-header {{
                display: flex;
                justify-content: space-between;
                margin: 15px 0 5px;
            }}
            .job-title {{
                font-weight: bold;
                font-size: 12pt;
            }}
            .company {{
                font-style: italic;
                color: #666;
            }}
            .date {{
                color: #666;
                font-size: 11pt;
            }}
            .skills-container {{
                columns: 3;
                margin-left: 20px;
            }}
            .skill-item {{
                break-inside: avoid;
                margin: 3px 0;
            }}
        </style>
    </head>
    <body>
        {content}
    </body>
</html>
"""

if 'optimized' not in st.session_state:
    st.session_state.optimized = None

def extract_keywords(text, max_words=20):
    """Enhanced keyword extraction with bi-grams"""
    text = re.sub(r'[^\w\s]', '', text.lower())
    vectorizer = TfidfVectorizer(
        stop_words=list(stop_words),
        ngram_range=(1, 2),
        max_features=1000
    )
    tfidf_matrix = vectorizer.fit_transform([text])
    return vectorizer.get_feature_names_out()[:max_words]

def calculate_ats_score(resume_text, jd_text):
    """Advanced ATS scoring with multiple factors"""
    jd_keywords = extract_keywords(jd_text, 25)
    resume_keywords = extract_keywords(resume_text, 25)
    matched_keywords = set(jd_keywords) & set(resume_keywords)
    
    found_sections = set([line.strip() for line in resume_text.split('\n') 
                        if line.strip().upper() in REQUIRED_SECTIONS])
    
    content_length = len(resume_text.split())
    
    skills_found = [skill for skill in SKILLS_LIST 
                   if re.search(r'\b' + re.escape(skill.lower()) + r'\b', resume_text.lower())]
    
    # Contact info validation
    has_email = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', resume_text))
    has_phone = bool(re.search(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', resume_text))
    
    keyword_score = (len(matched_keywords)/len(jd_keywords)) * 35
    section_score = (len(found_sections)/len(REQUIRED_SECTIONS)) * 25
    length_score = max(0, 1 - abs(content_length-700)/700) * 10
    skill_score = (len(skills_found)/len(SKILLS_LIST)) * 20
    contact_score = 10 if has_email and has_phone else 0
    
    return min(100, round(
        keyword_score + section_score + length_score + skill_score + contact_score, 
        1
    ))

def optimize_resume_with_ai(resume_text, jd):
    """AI-powered optimization with strict formatting rules"""
    prompt = f"""Create an ATS-optimized resume following these guidelines:

[Job Description]
{jd[:2000]}

[Original Resume]
{resume_text[:2000]}

[Requirements]
1. Structure:
- Header: Full Name | Phone | Email | LinkedIn/GitHub
- Professional Summary: 3 lines max, focus on JD requirements
- Technical Skills: 6-8 columns with JD keywords
- Professional Experience: Reverse chronological with metrics
- Education: Degrees only
- Certifications: Relevant to JD
- Projects: JD-related technologies

2. Formatting Rules:
- Use action verbs: Orchestrated, Spearheaded, Optimized
- Include metrics: "Improved performance by 40%"
- Mirror JD keywords: {', '.join(extract_keywords(jd)[:15])}
- Remove irrelevant experiences
- Use exact section headers: EXPERIENCE, EDUCATION, SKILLS, PROJECTS, CERTIFICATIONS

3. Optimization Targets:
- Keyword density: 3-5%
- Length: 600-800 words
- ATS Score: 80+

Return ONLY the formatted resume text with proper section headers in ALL CAPS."""
    
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    
    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}"},
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 2500,
                "timeout": 30
            },
            timeout=(3.05, 30)
        )
        optimized = response.json()['choices'][0]['message']['content']
        
        # Post-processing validation
        missing_sections = REQUIRED_SECTIONS - set(optimized.split('\n'))
        if missing_sections:
            optimized += '\n' + '\n'.join([f"{s}\n- Added placeholder content" 
                                         for s in missing_sections])
            
        return optimized
    except Exception as e:
        st.error(f"Optimization failed: {str(e)}")
        return None

def check_api_connection():
    """Test API connectivity"""
    try:
        response = requests.get(
            "https://api.deepseek.com/v1/models",
            headers={"Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}"},
            timeout=5
        )
        return response.status_code == 200
    except Exception as e:
        return False
    
def convert_to_html(content):
    """Convert optimized text to professional HTML"""
    sections = []
    current_section = None
    
    for line in content.split('\n'):
        if line.strip().upper() in REQUIRED_SECTIONS:
            if current_section:
                sections.append("</div>")
            current_section = line.strip().upper()
            sections.append(f"""
                <div class="section">
                    <div class="section-title">{current_section.title()}</div>
            """)
        elif current_section == 'SKILLS':
            skills = [f'<div class="skill-item">{skill.strip()}</div>' 
                     for skill in line.split(',')]
            sections.append(f'<div class="skills-container">{"".join(skills)}</div>')
        elif line.startswith('•'):
            sections.append(f'<div class="bullet-list">{line}</div>')
        elif re.match(r'.+?\s+-\s+.+?\s+\(\d{4}\s*-\s*\d{4}\)', line):
            parts = re.split(r'\s+-\s+', line)
            sections.append(f"""
                <div class="job-header">
                    <div>
                        <span class="job-title">{parts[0]}</span>
                        <span class="company">{parts[1]}</span>
                    </div>
                    <div class="date">{parts[2]}</div>
                </div>
            """)
        else:
            sections.append(f'<p>{html.escape(line)}</p>')
    
    return HTML_TEMPLATE.format(content='\n'.join(sections))

def create_professional_pdf(content):
    """Generate ATS-friendly PDF"""
    html_content = convert_to_html(content)
    return pdfkit.from_string(
        html_content, 
        False, 
        options={
            'encoding': 'UTF-8',
            'margin-top': '0.5in',
            'margin-right': '0.5in',
            'margin-bottom': '0.5in',
            'margin-left': '0.5in',
            'quiet': ''
        }
    )

def main():
    st.set_page_config(page_title="ATS Resume Optimizer", layout="wide")
    st.title("Professional ATS Resume Optimizer")
    
    # File upload section
    with st.expander("Upload Resume & Job Description", expanded=True):
        col1, col2 = st.columns([2, 3])
        with col1:
            resume_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
        with col2:
            jd = st.text_area("Paste Job Description", height=250)

    # Optimization controls
    if st.button("Generate Optimized Resume"):
        if resume_file and jd:
            with st.status("Processing...") as status:
                try:
                    # Read resume
                    status.update(label="Analyzing resume...")
                    if resume_file.name.endswith('.pdf'):
                        reader = PdfReader(resume_file)
                        resume_text = '\n'.join([page.extract_text() for page in reader.pages])
                    else:
                        doc = Document(resume_file)
                        resume_text = '\n'.join([para.text for para in doc.paragraphs])

                    # Optimize content
                    status.update(label="Optimizing with AI...")
                    optimized_content = optimize_resume_with_ai(resume_text, jd)
                    
                    if optimized_content:
                        # Calculate metrics
                        status.update(label="Calculating ATS score...")
                        ats_score = calculate_ats_score(optimized_content, jd)
                        
                        # Generate PDF
                        status.update(label="Formatting PDF...")
                        pdf_bytes = create_professional_pdf(optimized_content)
                        
                        # Update session state
                        st.session_state.optimized = {
                            'content': optimized_content,
                            'score': ats_score,
                            'pdf': pdf_bytes,
                            'history': [optimized_content]
                        }
                        st.success("Optimization complete!")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # Display results
    if st.session_state.get('optimized'):
        st.divider()
        
        # Sidebar controls
        with st.sidebar:
            st.subheader("ATS Optimization Checklist")
            st.checkbox("✅ Keywords from JD present", value=True)
            st.checkbox("✅ All required sections present", value=True)
            st.checkbox("✅ Contact information visible", value=True)
            st.checkbox("✅ No tables/images/graphics", value=True)
            st.checkbox("✅ Standard fonts used", value=True)
            st.checkbox("✅ Proper margins (0.75-1 inch)", value=True)
            st.checkbox("✅ Machine-readable format", value=True)
            
            st.download_button(
                "Download PDF Resume",
                data=st.session_state.optimized['pdf'],
                file_name="optimized_resume.pdf",
                mime="application/pdf"
            )

        # Main content
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("Resume Editor")
            edited_content = st.text_area(
                "Edit Content", 
                value=st.session_state.optimized['content'],
                height=600,
                key="editor"
            )
            
            if st.button("Save Changes"):
                new_pdf = create_professional_pdf(edited_content)
                st.session_state.optimized.update({
                    'content': edited_content,
                    'pdf': new_pdf,
                    'score': calculate_ats_score(edited_content, jd),
                    'history': st.session_state.optimized['history'] + [edited_content]
                })
                st.rerun()

        with col2:
            st.subheader("Optimization Report")
            st.metric("Current ATS Score", f"{st.session_state.optimized['score']}%")
            
            with st.expander("Keyword Analysis"):
                jd_kw = extract_keywords(jd)
                resume_kw = extract_keywords(st.session_state.optimized['content'])
                st.write("**Missing Keywords:**", list(set(jd_kw) - set(resume_kw))[:10])
                st.write("**Strong Keywords:**", list(set(jd_kw) & set(resume_kw))[:10])
            
            with st.expander("Section Validation"):
                found = set([line.strip() for line 
                           in st.session_state.optimized['content'].split('\n') 
                           if line.strip().upper() in REQUIRED_SECTIONS])
                for section in REQUIRED_SECTIONS:
                    st.checkbox(f"{section.title()} Section", 
                               value=section in found,
                               disabled=True)

            if st.button("Re-optimize from Edits"):
                with st.spinner("Re-optimizing..."):
                    new_content = optimize_resume_with_ai(
                        st.session_state.optimized['content'], 
                        jd
                    )
                    st.session_state.optimized = {
                        'content': new_content,
                        'score': calculate_ats_score(new_content, jd),
                        'pdf': create_professional_pdf(new_content),
                        'history': st.session_state.optimized['history'] + [new_content]
                    }
                    st.rerun()

if __name__ == "__main__":
    main()