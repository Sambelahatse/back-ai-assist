from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import os

app = Flask(__name__)
CORS(app)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

CONTEXT = open("context.txt", "r").read()
conversation_history = []

def ask_ai(question):
    global conversation_history

    model = genai.GenerativeModel("gemini-3-flash-preview")

    prompt = CONTEXT + "\n\n"
    for msg in conversation_history:
        role = "Utilisateur" if msg["role"] == "user" else "Assistant"
        prompt += f"{role} : {msg['content']}\n"

    prompt += f"Utilisateur : {question}"

    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.3,
            "max_output_tokens": 1024
        }
    )

    answer = response.text

    conversation_history.append({"role": "user", "content": question})
    conversation_history.append({"role": "assistant", "content": answer})

    return answer

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    if not data or "prompt" not in data:
        return jsonify({"error": "prompt manquant"}), 400

    return jsonify({
        "response": ask_ai(data["prompt"])
    })

@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "message": "API Flask Hugging Face active"
    })
