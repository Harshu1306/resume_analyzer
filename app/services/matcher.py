import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Lazy load model for contextual semantic processing
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Comprehensive predefined Technical Knowledge Base for Entity Extraction
TECHNICAL_SKILL_GRAPH = {
    # Languages & Frameworks
    'python', 'pytorch', 'tensorflow', 'keras', 'scikit-learn', 'numpy', 'pandas',
    'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'graphql', 'fastapi', 'flask', 'django',
    'javascript', 'typescript', 'react', 'node.js', 'express', 'vue', 'angular', 'next.js',
    'html', 'css', 'java', 'c++', 'c#', 'golang', 'rust',
    # Infrastructure, Devops & Systems
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'ci/cd', 'git', 'github', 'linux',
    # Data Science & Core Engineering Paradigms
    'machine learning', 'deep learning', 'nlp', 'computer vision', 'data engineering',
    'data pipelines', 'agile', 'scrum', 'project management', 'rest api', 'microservices',
    'data structures', 'algorithms', 'neural networks', 'deployment'
}


def extract_entities_via_intersection(text: str) -> set:
    """
    Extract technical keywords from text by mapping tokens and 
    phrases against a broad technology industry entity graph.
    """
    text_clean = text.lower()
    found_entities = set()

    # Check for multi-word or single-word match conditions explicitly boundary-locked
    for skill in TECHNICAL_SKILL_GRAPH:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_clean):
            found_entities.add(skill)

    return found_entities


def calculate_ats_metrics(resume_text: str, job_description: str):
    if not resume_text.strip() or not job_description.strip():
        return {
            "ats_score": 0.0,
            "keyword_match": 0.0,
            "semantic_match": 0.0,
            "suggestions": "Execution halted: Empty input fields detected."
        }

    # --- LAYER 1: HARD TOKEN INTERSECTION & NER MATCHING ---
    resume_entities = extract_entities_via_intersection(resume_text)
    jd_entities = extract_entities_via_intersection(job_description)

    if jd_entities:
        matched_skills = resume_entities.intersection(jd_entities)
        missing_skills = jd_entities.difference(resume_entities)
        # Ratio of matched entities vs total entities requested by the employer
        keyword_score = len(matched_skills) / len(jd_entities)
    else:
        # Resilient fallback: use Tfidf Character N-Grams if the job description contains zero predefined skills
        tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = tfidf.fit_transform(
            [resume_text.lower(), job_description.lower()])
        keyword_score = float(cosine_similarity(
            tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
        missing_skills = set()

    # --- LAYER 2: SOFT SEMANTIC CONTEXT ANALYSIS ---
    resume_embedding = semantic_model.encode([resume_text.lower()])[0]
    jd_embedding = semantic_model.encode([job_description.lower()])[0]
    raw_similarity = float(cosine_similarity(
        [resume_embedding], [jd_embedding])[0][0])

    # Text embeddings rarely hit 0.8+ naturally across distinct structures.
    # Min-Max Normalization rescales a 0.2 -> 0.8 similarity to a clear 0.0 -> 1.0 distribution range.
    semantic_score = max(0.0, min(1.0, (raw_similarity - 0.2) / 0.6)
                         ) if raw_similarity > 0.2 else raw_similarity

    # --- LAYER 3: WEIGHTED HYBRID AGGREGATION (OUT OF 10) ---
    # 50% Explicit Entity Overlap + 50% Conceptual/Role Fit
    final_hybrid_score = (keyword_score * 0.5) + (semantic_score * 0.5)

    # Scale calculation out of 10 and round cleanly
    final_score_out_of_10 = round(final_hybrid_score * 10, 1)
    final_score_out_of_10 = max(
        0.5, min(10.0, final_score_out_of_10))  # Secure bounds

    # --- LAYER 4: DYNAMIC COMPREHENSIVE SUGGESTIONS AUDIT ---
    suggestions = []

    if missing_skills:
        # Clean naming representation formatting (e.g. acronyms capitalized, rest title-cased)
        formatted_missing = [s.upper() if len(s) <= 4 else s.title()
                             for s in missing_skills]
        suggestions.append(
            f"⚠️ **Missing Core Skills:** The job description requires these missing terms: {', '.join(formatted_missing)}.")

    if keyword_score < 0.45:
        suggestions.append(
            "• **Context Density Optimization:** Your resume contains too few overlapping technologies. Try expanding your project bullets to detail *how* you implemented these missing tools.")

    if len(resume_text) < 650:
        suggestions.append("• **Information Depth Warning:** Your general resume profile text is structurally too short. Add measurable, metrics-driven achievements (e.g., 'Optimized performance by 15%') to strengthen contextual matching.")

    if not suggestions:
        suggestions.append(
            "✨ **Flawless Matrix Alignment:** Your engineering profile shows excellent core alignment with all requested keywords and structural parameters.")

    return {
        "ats_score": final_score_out_of_10,
        "keyword_match": round(keyword_score * 10, 1),
        "semantic_match": round(semantic_score * 10, 1),
        "suggestions": "\n\n".join(suggestions)
    }
