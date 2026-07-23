import re
import pdfplumber
import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download as _dl
    _dl("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
SKILLS_DB: frozenset[str] = frozenset({
    "python", "javascript", "typescript", "java", "c++", "c#", "golang", "rust",
    "html", "css", "tailwind", "react", "next.js", "vue", "angular",
    "node.js", "express", "flutter",

    "fastapi", "flask", "django",
    "mysql", "postgresql", "sqlite", "mongodb", "redis",

    "machine learning", "deep learning", "nlp", "computer vision",
    "neural networks", "generative ai", "large language models",
    "retrieval augmented generation", "rag", "prompt engineering",
    "fine tuning", "agentic ai", "ai agents",

    "tensorflow", "pytorch", "keras", "scikit-learn",
    "numpy", "pandas", "matplotlib", "plotly",

    "langchain", "langgraph", "llamaindex",
    "transformers", "hugging face", "sentence transformers",
    "openai", "gemini", "claude", "groq",

    "vector database", "vectordb", "chromadb", "pinecone",
    "faiss", "milvus", "qdrant", "pgvector",

    "docker", "kubernetes", "aws", "azure", "gcp",
    "linux", "git", "github",

    "rest api", "graphql", "microservices",
    "data pipelines", "apache spark", "airflow",

    "data structures", "algorithms", "operating systems",
    "database management system", "dbms",
    "computer networks", "networking",
    "object oriented programming", "oop",

    "deployment", "streamlit", "gradio",

    "sql", "no sql", "agile", "scrum"
})
SKILL_SYNONYMS: dict[str, str] = {
    "js":                  "javascript",
    "ts":                  "typescript",
    "ml":                  "machine learning",
    "dl":                  "deep learning",
    "sk-learn":            "scikit-learn",
    "sklearn":             "scikit-learn",
    "tf":                  "tensorflow",
    "k8s":                 "kubernetes",
    "kube":                "kubernetes",
    "pg":                  "postgresql",
    "postgres":            "postgresql",
    "mongo":               "mongodb",
    "cv":                  "computer vision",
    "nlproc":              "nlp",
    "natural language processing": "nlp",
    "go":                  "golang",
    "nodejs":              "node.js",
    "node":                "node.js",
    "reactjs":             "react",
    "vue.js":              "vue",
    "vuejs":               "vue",
    "angularjs":           "angular",
    "nextjs":              "next.js",
    "rest":                "rest api",
    "restful":             "rest api",
}
_SINGLE_TOKEN_SKILLS: frozenset[str] = frozenset(
    s for s in SKILLS_DB if " " not in s)
_MULTI_TOKEN_SKILLS:  frozenset[str] = frozenset(
    s for s in SKILLS_DB if " " in s)
_LEMMA_TO_SKILL: dict[str, str] = {}
for _skill in _SINGLE_TOKEN_SKILLS:
    _doc = nlp(_skill)
    if _doc:
        _LEMMA_TO_SKILL[_doc[0].lemma_.lower()] = _skill


def normalize_skill(raw: str) -> str:
    cleaned = raw.lower().strip()
    cleaned = re.sub(r"[^\w\s.#+]", "", cleaned)
    return SKILL_SYNONYMS.get(cleaned, cleaned)


def match_skills(text: str) -> set[str]:
    if not text:
        return set()
    found: set[str] = set()
    text_lower = text.lower()
    for alias, canonical in SKILL_SYNONYMS.items():

        text_lower = re.sub(
            r"\b" + re.escape(alias) + r"\b",
            canonical,
            text_lower,
        )
    doc = nlp(text_lower)
    for token in doc:
        lemma = token.lemma_.lower()
        if lemma in _LEMMA_TO_SKILL:
            found.add(_LEMMA_TO_SKILL[lemma])
        # Also check the raw token text directly (handles acronyms / short names)
        if token.text in _SINGLE_TOKEN_SKILLS:
            found.add(token.text)

    # --- Pass 3: multi-word skill boundary matching ---
    for skill in _MULTI_TOKEN_SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.add(skill)

    return found


def extract_text_from_pdf(file_path: str) -> str:

    pages: list[str] = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text)
    return "\n".join(pages)


def extract_skills_and_info(text: str) -> dict:

    skills = match_skills(text)

    email_match = re.search(r"[\w.\-]+@[\w.\-]+\.\w+", text)
    email = email_match.group(0) if email_match else "Not Found"

    return {
        "skills": sorted(skills),   # sorted for stable, readable output
        "email": email,
        "raw_text": text,
    }
