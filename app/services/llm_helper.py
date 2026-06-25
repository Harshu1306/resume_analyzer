from transformers import T5ForConditionalGeneration, AutoTokenizer

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def generate_interview_questions(role: str, skills: str):
    prompt = f"Generate a list of 3 challenging technical interview questions for a {role} role who knows {skills}."
    
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=150, num_beams=4, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up formatting for clean view rendering
    questions = [q.strip() for q in response.split('?') if q.strip()]
    formatted = ""
    for i, q in enumerate(questions[:3], 1):
        formatted += f"{i}. {q}?\n"
    return formatted if formatted else "Could not generate questions. Try adjustment criteria."