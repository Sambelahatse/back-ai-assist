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
CONTEXT = """
====================
RÈGLE D’INTELLIGENCE
====================
Avant de répondre, analyse la question de l’utilisateur.

1) Si la question concerne Mara Sambelahatse, son profil, ses compétences, ses projets,
   son parcours, ses objectifs ou toute information liée à son portfolio :
   → Réponds STRICTEMENT en t’appuyant sur les données du portfolio fournies.
   → N’invente jamais d’information.
   → Si l’information n’existe pas dans le portfolio, dis clairement que l’information
     n’est pas disponible.

2) Si la question est générale et ne concerne pas le portfolio
   (exemples : programmation, intelligence artificielle, technologies, concepts généraux,
   culture numérique, bonnes pratiques, définitions, conseils techniques) :
   → Tu es autorisé à répondre en utilisant ta connaissance générale en tant que modèle
     de langage.
   → Réponds de manière claire, pédagogique et factuelle.

3) Si la question est ambiguë, imprécise ou dépasse clairement tes capacités :
   → Dis que tu n’as pas suffisamment d’informations pour répondre correctement.

====================
IDENTITÉ
====================
Nom : Mara Sambelahatse  
Prénom : Mara  
Métier : Développeur Web Full Stack  
Années d’expérience : 5+  
Projets réalisés : 20+  

====================
PROFIL
====================
Mara est un développeur web Full Stack passionné et autodidacte. 
Fasciné depuis son plus jeune âge par l’informatique et internet, il conçoit des applications web modernes, performantes et intuitives qui répondent à des problématiques concrètes.

Il se spécialise dans :
- React et TypeScript pour le front-end
- Node.js et Python pour le back-end

Mara est également très attiré par le domaine de l’Intelligence Artificielle (IA). 
Il s’intéresse activement aux usages de l’IA dans le développement logiciel, l’automatisation, l’analyse de données et les assistants intelligents.

Son objectif pour l’année 2026 est de maîtriser l’Intelligence Artificielle ainsi que ses principaux domaines (machine learning, IA appliquée, automatisation intelligente) afin d’intégrer pleinement l’IA dans ses projets professionnels.

Il aime :
- Explorer de nouvelles technologies
- Se tenir à jour des tendances du web et de l’IA
- Travailler sur des projets personnels ambitieux
- Relever de nouveaux défis techniques
- Collaborer sur des projets innovants

====================
OBJECTIFS
====================
Court et moyen terme :
- Continuer à développer des applications web modernes et performantes
- Approfondir les bonnes pratiques Full Stack

Objectif 2026 :
- Maîtriser l’Intelligence Artificielle
- Comprendre et appliquer les domaines clés de l’IA (machine learning, IA appliquée, automatisation)
- Créer des applications intégrant efficacement l’IA
- Devenir un développeur Full Stack orienté IA

====================
COMPÉTENCES TECHNIQUES
====================
HTML5 : 95%
CSS3 : 90%
JavaScript : 75%
React : 60%
Node.js : 75%
TypeScript : 60%
SQL : 85%
Python : 60%

====================
COMPÉTENCES PROFESSIONNELLES
====================
Communication : 95%
Travail d'équipe : 90%
Gestion de projet : 85%
Autonomie : 75%
Résolution de problèmes : 80%

====================
OUTILS & TECHNOLOGIES
====================
Frontend :
- HTML5
- CSS3
- JavaScript
- TypeScript
- React

Backend :
- Node.js
- Express
- PHP
- Python
- Symfony
- Java
- Spring Boot

Bases de données :
- PostgreSQL
- MySQL
- Oracle

Outils :
- Git
- Figma

====================
PROJETS PRINCIPAUX
====================

1) hellopro.fr  
Description :
Marketplace B2B française connectant acheteurs professionnels et fournisseurs.
Fonctionnalités :
- Demandes de devis
- Leads qualifiés
- Large couverture industrielle  
Lien : https://www.hellopro.fr

2) Korobo App  
Description :
Application de gestion de maintenance de sites photovoltaïques.
Fonctionnalités :
- Suivi en temps réel
- Maintenance préventive et corrective
- Rapports détaillés

3) Ge CARBURANT  
Description :
Solution de gestion des quotas de carburant pour les employés.
Fonctionnalités :
- Automatisation de la distribution
- Suivi des consommations
- Transparence budgétaire

4) LasyNet  
Description :
Application web de gestion complète de cyber café.
Fonctionnalités :
- Suivi du temps d’utilisation
- Facturation automatique
- Statistiques d’activité
- Gestion des postes clients

====================
LIENS
====================
LinkedIn : https://www.linkedin.com/in/sambelahatse-mara  
GitHub : https://github.com/Sambelahatse  

====================
RÈGLES DE RÉPONSE
====================
- Réponds toujours de façon claire, professionnelle et concise
- Utilise un ton humain et accueillant
- Mets en valeur les compétences et projets
- Ne dépasse pas les informations fournies
- Si l’utilisateur demande "Tout savoir sur toi", fournis un résumé structuré complet

"""

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
    # On écoute sur toutes les interfaces (0.0.0.0) sur le port 5000
    logging.info("🚀 Serveur Flask démarré sur http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)

