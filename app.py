import gradio as gr
from app.services.parser import extract_text_from_pdf, extract_skills_and_info
from app.services.matcher import calculate_ats_metrics
from app.services.llm_helper import generate_interview_questions
from app.database import get_db_connection


def process_ats_diagnostics(file_obj, job_description):
    if not file_obj or not job_description:
        return "### ⚠️ System Warning\nPlease drop both a valid PDF resume and Target Job Details.", "", ""

    # Process pipeline
    resume_text = extract_text_from_pdf(file_obj.name)
    profile = extract_skills_and_info(resume_text)
    metrics = calculate_ats_metrics(resume_text, job_description)

    # Extract score out of 10 straight from the updated backend dictionary metric
    ats_score_out_of_10 = metrics["ats_score"]

    # Save metrics to local persistence engine
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO scans (email, ats_score) VALUES (?, ?)",
                       (profile["email"], ats_score_out_of_10))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database logging skip: {e}")

    skills_output = ", ".join(
        profile["skills"]) if profile["skills"] else "None explicitly found."

    # HTML template completely updated to clean metric values out of 10
    results_markdown = f"""
    <div style="background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.3); padding: 20px; border-radius: 8px;">
        <h2 style='margin: 0 0 15px 0; color: #818cf8;'>🎯 Overall ATS Score: {ats_score_out_of_10}/10</h2>
        <p style='margin: 5px 0;'><strong>🔍 Keyword Match:</strong> {metrics['keyword_match']}/10</p>
        <p style='margin: 5px 0;'><strong>🧬 Semantic Alignment:</strong> {metrics['semantic_match']}/10</p>
        <p style='margin: 5px 0;'><strong>📧 Extracted Email:</strong> {profile['email']}</p>
    </div>
    """

    return results_markdown, skills_output, metrics["suggestions"]


# --- Premium Custom Theme Architecture ---
custom_theme = gr.themes.Base(
    primary_hue="indigo",      # Sleek, premium indigo accent instead of standard blue
    neutral_hue="slate",       # Deep slate grey background palette
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"),
          "ui-sans-serif", "sans-serif"]
)

# Custom CSS for modern typography metrics and container adjustments
custom_css = """
.gradio-container { max-width: 1200px !important; margin: 0 auto; }
h1, h2, h3 { font-weight: 700 !important; letter-spacing: -0.02em; }
footer { display: none !important; } /* Cleans up footer clutter */
"""

# Build Interface
with gr.Blocks(theme=custom_theme, css=custom_css, title="Career Assistant") as demo:

    # Header Area
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                """
                # 💼 Career Assistant
                <p style="color: #94a3b8; font-size: 1.1rem; margin-top: -10px;">
                    Resume Analysis
                </p>
                <hr style="border-color: #334155; margin: 20px 0;">
                """
            )

    with gr.Tabs():
        with gr.TabItem("ATS System"):
            gr.Markdown(
                "<p style='color: #94a3b8; margin-bottom: 20px;'></p>")

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    resume_input = gr.File(
                        label="Upload Resume (PDF)", file_types=[".pdf"])
                    jd_input = gr.Textbox(
                        label="Target Job Description", placeholder="Paste job description details here...", lines=7)
                    run_btn = gr.Button(
                        "Run", variant="primary", size="lg")

                with gr.Column(scale=1):
                    score_display = gr.Markdown(
                        """
                        <div style="border: 1px dashed #334155; padding: 40px; text-align: center; border-radius: 8px; color: #64748b;">
                            Results
                        </div>
                        """
                    )
                    skills_display = gr.Textbox(
                        label="Capabilities", interactive=False)
                    suggestions_display = gr.Textbox(
                        label="Improvements", lines=5, interactive=False)

            run_btn.click(
                fn=process_ats_diagnostics,
                inputs=[resume_input, jd_input],
                outputs=[score_display, skills_display, suggestions_display]
            )

        with gr.TabItem("Interview Simulation"):
            gr.Markdown(
                "<p style='color: #94a3b8; margin-bottom: 20px;'></p>")

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    role_input = gr.Textbox(
                        label="Target Position Title", placeholder="e.g., Full Stack Engineer")
                    skills_input = gr.Textbox(
                        label="Core Capabilities", placeholder="e.g., Python, React, FastAPI")
                    gen_btn = gr.Button(
                        "Generate", variant="primary", size="lg")

                with gr.Column(scale=1):
                    questions_display = gr.Textbox(
                        label="Generated Targets", lines=9, interactive=False)

            gen_btn.click(
                fn=generate_interview_questions,
                inputs=[role_input, skills_input],
                outputs=questions_display
            )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
