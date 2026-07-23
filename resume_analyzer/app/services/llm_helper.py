from transformers import T5ForConditionalGeneration, AutoTokenizer

_MODEL_NAME = "google/flan-t5-base"
_tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
_model = T5ForConditionalGeneration.from_pretrained(_MODEL_NAME)


def generate_interview_questions(role: str, skills: str) -> str:
    if not role.strip() or not skills.strip():
        return "Please provide both a target role and a list of skills."

    prompt = (
        f"Generate a list of 3 challenging technical interview questions "
        f"for a {role.strip()} role who knows {skills.strip()}."
    )

    inputs = _tokenizer(prompt, return_tensors="pt")
    outputs = _model.generate(
        **inputs,
        max_length=150,
        num_beams=4,
        early_stopping=True,
    )
    response = _tokenizer.decode(outputs[0], skip_special_tokens=True)

    questions = [q.strip() for q in response.split("?") if q.strip()][:3]
    if not questions:
        return "Could not generate questions. Try adjusting the role or skills."

    return "\n".join(f"{i}. {q}?" for i, q in enumerate(questions, 1))
