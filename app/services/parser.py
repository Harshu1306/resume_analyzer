import pdfplumber
import spacy
import re

# Load small English model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

SKILLS_DB = [
    "python", "javascript", "flutter", "react", "node.js", "express", "mysql",
    "sqlite", "fastapi", "machine learning", "data structures", "operating systems",
    "git", "aws", "docker", "c++", "java", "html", "css", "tailwind"
]


def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def extract_skills_and_info(text):
    doc = nlp(text.lower())
    extracted_skills = set()

    for token in doc:
        if token.text in SKILLS_DB:
            extracted_skills.add(token.text)

    text_lowercase = text.lower()
    for skill in SKILLS_DB:
        if " " in skill and skill in text_lowercase:
            extracted_skills.add(skill)

    email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    email = email_match.group(0) if email_match else "Not Found"

    return {
        "skills": list(extracted_skills),
        "email": email,
        "raw_text": text
    }
