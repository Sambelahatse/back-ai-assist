from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
import os
import logging

load_dotenv()

app = Flask(__name__)
CORS(app)

client = OpenAI(
    # FROM .env file
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# 1️⃣ Contexte fixe (ton portfolio)
CONTEXT = open("context.txt", "r").read()

# 2️⃣ Historique mémoire (liste de dictionnaires pour le format ChatCompletion)
conversation_history = []

def ask_ai(question):
    global conversation_history

    # Préparation des messages pour l'API Chat
    messages = [{"role": "system", "content": CONTEXT}]
    
    # Ajout de l'historique existant
    for entry in conversation_history:
        messages.append(entry)
    
    # Ajout de la nouvelle question
    messages.append({"role": "user", "content": question})

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
        )

        answer = response.choices[0].message.content

        # Mise à jour de l'historique
        conversation_history.append({"role": "user", "content": question})
        conversation_history.append({"role": "assistant", "content": answer})

        return answer
    except Exception as e:
        return f"Désolé, une erreur est survenue : {str(e)}"

@app.route('/ask', methods=['POST'])
def ask():
    # Récupération des données JSON envoyées par l'utilisateur
    data = request.json
    
    if not data or 'prompt' not in data:
        return jsonify({
            "status": "error",
            "message": "Le champ 'prompt' est requis dans le corps de la requête JSON."
        }), 400

    prompt = data['prompt']
    
    # Appel de l'IA
    response_text = ask_ai(prompt)
    
    return jsonify({
        "status": "success",
        "prompt": prompt,
        "response": response_text
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "API LLM Mara Sambelahatse est active.",
        "usage": "Envoyez un POST sur /ask avec {\"prompt\": \"votre question\"}"
    })

if __name__ == '__main__':
    if (os.getenv("ENVIRONMENT") == "dev"):
        # comment: port 5000
        logging.info("🚀 Serveur Flask démarré sur http://localhost:5000")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        # comment: port 80
        logging.info("🚀 Serveur Flask démarré sur http://localhost:80")
        app.run(host='0.0.0.0', port=80, debug=False)

