# 💼 AI-Powered Resume Analyzer & Career Assistant

An advanced, full-stack hybrid engineering platform designed to audit resumes against technical job descriptions using deterministic entity extraction, soft semantic contextual embeddings, and automated interview simulation.

**🔗 Live Production Space:** [Career Assistant on Hugging Face Spaces](https://huggingface.co/spaces/harshitha1306/resume-analyzer)

---

## 📌 The Problem

Most companies use software called an **Applicant Tracking System (ATS)** to filter resumes. However, traditional systems have two big problems:

1. **They are too literal:** If a job description asks for "REST API" and your resume says "FastAPI" or "Web Services", the system might reject you automatically—even though you have the right skills.
2. **They can be cheated:** Unqualified candidates can simply copy and paste exact keywords into their resumes to trick the system, even if they don't actually know how to use those tools in real projects.

### Our Solution
This project builds a **smarter AI Resume Analyzer**. 

Instead of just checking if words match exactly, it uses advanced AI (Natural Language Processing) to read between the lines. It scores your profile fairly out of **10.0** by looking at both your exact technical keywords and the overall meaning of your experience. Finally, it tells you exactly what is missing and generates custom interview questions to help you prepare!



## 🛠️ Core Technology Stack

The platform is engineered using a robust, modular full-stack Python architecture:

| Component | Framework / Library | Purpose |
| :--- | :--- | :--- |
| **Frontend UI** | Gradio 6.0+ | Modern web interface design featuring dark slate UI layout themes and asynchronous element triggers. |
| **Document Parsing**| `pdfplumber` | High-fidelity unstructured text extraction directly out of raw PDF candidate profiles. |
| **Natural Language**| `spaCy` (en_core_web_sm) | Lexical token analysis and text preprocessing execution blocks. |
| **Semantic Intelligence**| `sentence-transformers` | Contextual semantic analysis powered by the pre-trained `all-MiniLM-L6-v2` transformer model. |
| **Vector Similarity** | `scikit-learn` | Multi-dimensional cosine similarity matching across embedding vectors and fallback character N-Gram TF-IDF matrices. |
| **Local Persistence** | `SQLite3` | Local database engine tracking scanned metrics and candidate metadata securely. |

---

## 🏗️ Architecture & Scoring Pipeline

The calculation matrix drops arbitrary percentages in favor of a clean, transparent **Scale of 10.0**:

1. **Layer 1: Hard Token Intersection** Maps the lowercased text boundary tokens (`\b{keyword}\b`) against a predefined Technical Skill Graph. Resolves the strict ratio of matched terms against total employer requirements.
2. **Layer 2: Soft Semantic Context Analysis** Generates a 384-dimensional dense vector space embedding of the overall text structures. Applies Min-Max normalization to shift organic contextual alignments accurately into a visible distribution curve.
3. **Layer 3: Dynamic Suggestions Audit** Runs an automated rule engine evaluating text depth (character count metrics) and missing skill vectors to emit metrics-driven optimization hints.

---

## 🚀 Getting Started Locally

### Prerequisites
* Python 3.11, 3.12, or 3.13
* Git

### Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
   cd YOUR_REPO_NAME
