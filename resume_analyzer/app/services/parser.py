"""
parser.py — Resume text extraction and skill-matching utilities.

Public API
----------
extract_text_from_pdf(file_path) -> str
extract_skills_and_info(text)    -> dict
match_skills(text)               -> set[str]   ← shared helper, imported by matcher/llm_helper
normalize_skill(raw)             -> str        ← single normalisation point
"""

import re
import pdfplumber
import spacy

# ---------------------------------------------------------------------------
# spaCy model (lazy fallback download if not installed)
# ---------------------------------------------------------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download as _dl
    _dl("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# ---------------------------------------------------------------------------
# Canonical skill vocabulary — lowercase, deduplicated
# ---------------------------------------------------------------------------
SKILLS_DB: frozenset[str] = frozenset({
    "python", "javascript", "typescript", "flutter", "react", "node.js",
    "express", "mysql", "sqlite", "fastapi", "flask", "django",
    "machine learning", "deep learning", "data structures", "operating systems",
    "git", "aws", "docker", "kubernetes", "c++", "java", "html", "css",
    "tailwind", "postgresql", "mongodb", "redis", "graphql", "vue", "angular",
    "next.js", "pytorch", "tensorflow", "keras", "scikit-learn", "numpy",
    "pandas", "nlp", "computer vision", "rest api", "microservices",
    "data pipelines", "agile", "scrum", "linux", "github", "azure", "gcp",
    "algorithms", "neural networks", "golang", "rust", "c#", "deployment",
})

# ---------------------------------------------------------------------------
# Synonym / alias map — ONE canonical place for alias resolution.
# All keys and values must be lowercase.
# ---------------------------------------------------------------------------
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

# Pre-build a lemma→canonical map from SKILLS_DB for single-token skills
# (multi-word skills are matched via boundary-safe regex separately)
_SINGLE_TOKEN_SKILLS: frozenset[str] = frozenset(s for s in SKILLS_DB if " " not in s)
_MULTI_TOKEN_SKILLS:  frozenset[str] = frozenset(s for s in SKILLS_DB if " " in s)

# Map each skill's spaCy lemma → the canonical skill string
_LEMMA_TO_SKILL: dict[str, str] = {}
for _skill in _SINGLE_TOKEN_SKILLS:
    _doc = nlp(_skill)
    if _doc:
        _LEMMA_TO_SKILL[_doc[0].lemma_.lower()] = _skill


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_skill(raw: str) -> str:
    """
    Lowercase and strip punctuation, then resolve known synonyms.
    Returns the canonical skill name or the cleaned input unchanged.
    """
    cleaned = raw.lower().strip()
    cleaned = re.sub(r"[^\w\s.#+]", "", cleaned)
    return SKILL_SYNONYMS.get(cleaned, cleaned)


def match_skills(text: str) -> set[str]:
    """
    Extract skills present in *text* against SKILLS_DB using:
      1. Token-level lemma matching (single-word skills)
      2. Synonym resolution before lookup
      3. Boundary-safe regex for multi-word skills
    Returns a set of canonical skill names (lowercase).

    This is the single source of truth for skill detection —
    imported and reused by matcher.py and llm_helper.py.
    """
    if not text:
        return set()

    found: set[str] = set()
    text_lower = text.lower()

    # --- Pass 1: synonym pre-substitution on the raw text ---
    # Replace known aliases so downstream matching works on canonical names
    for alias, canonical in SKILL_SYNONYMS.items():
        # Use word-boundary replacement to avoid partial replacements
        text_lower = re.sub(
            r"\b" + re.escape(alias) + r"\b",
            canonical,
            text_lower,
        )

    # --- Pass 2: single-token lemma matching via spaCy ---
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_path: str) -> str:
    """Extract raw text from all pages of a PDF file."""
    pages: list[str] = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text)
    return "\n".join(pages)


def extract_skills_and_info(text: str) -> dict:
    """
    Parse a resume text and return:
      - skills: list of detected canonical skill names
      - email:  first email address found, or "Not Found"
      - raw_text: original text (passed through for downstream use)
    """
    skills = match_skills(text)

    email_match = re.search(r"[\w.\-]+@[\w.\-]+\.\w+", text)
    email = email_match.group(0) if email_match else "Not Found"

    return {
        "skills": sorted(skills),   # sorted for stable, readable output
        "email": email,
        "raw_text": text,
    }
