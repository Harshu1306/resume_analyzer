import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from app.services.parser import match_skills


_semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

KEYWORD_WEIGHT:  float = 0.5   # fraction of final score from keyword overlap
SEMANTIC_WEIGHT: float = 0.5   # fraction of final score from semantic similarity
SEMANTIC_LOW:    float = 0.20  # raw cosine below this → mapped to 0.0
SEMANTIC_HIGH:   float = 0.80  # raw cosine above this → mapped to 1.0
SCORE_MIN:       float = 0.5   # minimum ATS score returned (avoid discouraging 0)
SCORE_MAX:       float = 10.0
CHUNK_SENTENCES: int   = 5     # max sentences per embedding chunk


def _chunk_text(text: str) -> list[str]:
   
    raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    if not sentences:
        return [text.strip()] if text.strip() else []

    return [
        " ".join(sentences[i: i + CHUNK_SENTENCES])
        for i in range(0, len(sentences), CHUNK_SENTENCES)
    ]


def _embed(text: str) -> np.ndarray:
    """Embed *text* by chunking into sentences, batch-encoding all chunks in
    one model call, then mean-pooling the chunk embeddings."""
    chunks = _chunk_text(text)
    if not chunks:
        return np.zeros(_semantic_model.get_sentence_embedding_dimension())

    chunk_embeddings: np.ndarray = _semantic_model.encode(
        chunks,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return chunk_embeddings.mean(axis=0)


def _jd_skills(jd_text: str) -> frozenset:
    """Extract the skill set for a given job description."""
    return frozenset(match_skills(jd_text))


def _keyword_score(
    resume_text: str,
    jd_text: str,
) -> tuple[float, frozenset]:
    """
    Compute the normalised keyword overlap score in [0, 1] and the set of
    skills missing from the resume.

    Returns (score, missing_skills).

    When the JD contains no skills from the known vocabulary, falls back to
    TF-IDF cosine similarity so the metric is never vacuously 0 or 1.
    """
    resume_skills = frozenset(match_skills(resume_text))
    jd_skills = _jd_skills(jd_text)

    if jd_skills:
        matched = resume_skills & jd_skills
        missing = jd_skills - resume_skills
        score = len(matched) / len(jd_skills)
        return score, missing

    # Fallback: TF-IDF similarity when no domain skills detected in JD
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = tfidf.fit_transform([resume_text.lower(), jd_text.lower()])
    score = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
    return score, frozenset()


def _semantic_score(resume_text: str, jd_text: str) -> float:
    """
    Compute the normalised semantic similarity score in [0, 1].

    Raw cosine similarity is normalised from the [SEMANTIC_LOW, SEMANTIC_HIGH]
    empirical range to [0, 1], then clamped. This prevents scores from
    artificially clustering in the middle of the scale.
    """
    resume_emb = _embed(resume_text.lower())
    jd_emb = _embed(jd_text.lower())

    raw: float = float(
        cosine_similarity(resume_emb.reshape(1, -1), jd_emb.reshape(1, -1))[0][0]
    )
    raw = max(0.0, min(1.0, raw))  # clamp before normalising

    span = SEMANTIC_HIGH - SEMANTIC_LOW
    normalised = (raw - SEMANTIC_LOW) / span
    return max(0.0, min(1.0, normalised))


def _build_suggestions(
    missing_skills: frozenset,
    keyword_score: float,
    resume_text: str,
) -> str:
    """
    Build a human-readable suggestions string from the scoring signals.
    Returns at least one suggestion (a positive message when everything is good).
    """
    suggestions: list[str] = []

    if missing_skills:
        formatted = [
            s.upper() if len(s) <= 4 else s.title()
            for s in sorted(missing_skills)
        ]
        suggestions.append(
            f"⚠️ **Missing Core Skills:** The job description requires: "
            f"{', '.join(formatted)}."
        )

    if keyword_score < 0.45:
        suggestions.append(
            "• **Context Density Optimization:** Your resume has too few overlapping "
            "technologies. Expand project bullets to detail *how* you used the missing tools."
        )

    if len(resume_text) < 650:
        suggestions.append(
            "• **Information Depth Warning:** Your resume text is too short. "
            "Add measurable, metrics-driven achievements (e.g., 'Optimised performance "
            "by 15%') to strengthen contextual matching."
        )

    if not suggestions:
        suggestions.append(
            "✨ **Excellent Alignment:** Your profile shows strong keyword overlap "
            "and semantic fit with the job description."
        )

    return "\n\n".join(suggestions)


def calculate_ats_metrics(resume_text: str, job_description: str) -> dict:
    """
    Compute a composite ATS score (out of 10) for a resume against a job description.

    Score breakdown
    ---------------
    - keyword_match  (KEYWORD_WEIGHT = 50 %): normalised overlap of domain skills
    - semantic_match (SEMANTIC_WEIGHT = 50 %): mean-pooled sentence-transformer similarity

    Returns a dict with keys:
      ats_score      float  — final weighted score /10, clamped to [SCORE_MIN, SCORE_MAX]
      keyword_match  float  — keyword sub-score /10
      semantic_match float  — semantic sub-score /10
      suggestions    str    — actionable improvement notes
    """
    if not resume_text.strip() or not job_description.strip():
        return {
            "ats_score": 0.0,
            "keyword_match": 0.0,
            "semantic_match": 0.0,
            "suggestions": "⚠️ Empty input: please provide both a resume and a job description.",
        }

    kw_score, missing = _keyword_score(resume_text, job_description)
    sem_score = _semantic_score(resume_text, job_description)

    hybrid = (kw_score * KEYWORD_WEIGHT) + (sem_score * SEMANTIC_WEIGHT)
    final = round(max(SCORE_MIN, min(SCORE_MAX, hybrid * 10)), 1)

    return {
        "ats_score":      final,
        "keyword_match":  round(kw_score  * 10, 1),
        "semantic_match": round(sem_score * 10, 1),
        "suggestions":    _build_suggestions(missing, kw_score, resume_text),
    }
