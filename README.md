# 💼 AI Resume Analyzer & Career Assistant

An AI-powered resume evaluation platform that analyzes resumes against job descriptions using a hybrid approach combining keyword matching and semantic similarity. The system provides transparent resume scoring, identifies missing technical skills, and generates personalized interview questions to help candidates prepare for technical interviews.

**🔗 Live Demo:** https://huggingface.co/spaces/harshitha1306/resume-analyzer

---

## Overview

Traditional Applicant Tracking Systems (ATS) often rely on exact keyword matching, which can lead to two common issues:

* **Qualified candidates are overlooked** because they use different terminology or synonyms than those listed in the job description.
* **Keyword stuffing** can artificially increase ATS scores without demonstrating genuine experience or project depth.

This project addresses these limitations by combining deterministic keyword matching with transformer-based semantic similarity to provide a more balanced and explainable resume evaluation.

---

## Features

* 📄 PDF resume parsing
* 🎯 Job Description matching
* 🤖 Hybrid ATS scoring (Keyword + Semantic Analysis)
* 📊 Resume score on a 10-point scale
* 🧠 AI-powered skill gap analysis
* 💡 Resume improvement suggestions
* 🎤 Automatically generated technical interview questions
* 💾 SQLite database for storing evaluation history
* 🌐 Interactive Gradio web interface

---

## Technology Stack

| Category             | Technologies                               |
| -------------------- | ------------------------------------------ |
| Frontend             | Gradio                                     |
| Programming Language | Python 3.11+                               |
| PDF Processing       | pdfplumber                                 |
| NLP                  | spaCy                                      |
| Semantic Similarity  | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Machine Learning     | Scikit-learn                               |
| Database             | SQLite                                     |
| Text Processing      | Regular Expressions, TF-IDF                |

---

## System Architecture

The resume evaluation pipeline consists of three stages:

### 1. Keyword Matching (5 Points)

* Extracts required technical skills from the job description.
* Performs exact keyword matching against the resume.
* Calculates a keyword match score based on skill coverage.

### 2. Semantic Similarity (5 Points)

* Generates dense sentence embeddings using **Sentence Transformers**.
* Computes cosine similarity between the resume and job description.
* Rewards contextual relevance even when exact keywords differ.

### 3. Recommendation Engine

* Identifies missing skills.
* Detects weak resume sections.
* Generates actionable suggestions for improvement.
* Creates personalized technical interview questions.

---

## Project Structure

```text
resume_analyzer/
│
├── app/
│   ├── ui.py
│   ├── services/
│   │   ├── parser.py
│   │   ├── matcher.py
│   │   ├── llm_helper.py
│   │   └── database.py
│   └── assets/
│
├── models/
├── data/
├── requirements.txt
├── app.py
└── README.md
```

---

## Scoring Methodology

| Component           |   Weight |
| ------------------- | -------: |
| Keyword Matching    |      5.0 |
| Semantic Similarity |      5.0 |
| **Total Score**     | **10.0** |

The final score combines deterministic keyword matching with semantic similarity, providing a more comprehensive assessment than traditional ATS systems.

---

## Installation

### Prerequisites

* Python 3.11 or later
* Git

### Clone the Repository

```bash
git clone https://github.com/Harshu1306/resume_analyzer.git

cd resume_analyzer
```

### Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

The application will be available locally in your browser.

---

## Future Improvements

* Resume ranking across multiple candidates
* Skill extraction using Named Entity Recognition (NER)
* LLM-powered resume rewriting
* Recruiter dashboard
* Multi-language resume support
* Cloud database integration
* User authentication

---

