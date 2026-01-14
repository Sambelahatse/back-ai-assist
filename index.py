from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from dotenv import load_dotenv
import os
import logging
import time
from collections import deque
from typing import Dict, List

# Chargement des variables d'environnement
load_dotenv()

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration CORS
CORS(app, resources={
    r"/*": {"origins": os.getenv("ALLOWED_ORIGINS", "*").split(",")}
})

# Initialisation du client OpenAI pour Groq
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# 🔥 MODÈLES ACTUALISÉS (Janvier 2026)
AVAILABLE_MODELS = {
    # "primary": "llama-3.2-90b-text-preview",  # Nouveau modèle flagship
    # "fast": "llama-3.2-11b-text-preview",     # Rapide et efficace
    # "versatile": "llama-3.1-70b-versatile",   # Toujours disponible
    "long_context": "llama-3.1-8b-instant",   # Pour contexte long
    # "multilingual": "llama-3.1-8b-instant",   # Bon en plusieurs langues
    "fallback": "llama-3.1-8b-instant"        # Fallback garanti
}

# 1️⃣ Chargement du contexte avec gestion d'erreur
def load_context(file_path: str = "context.txt") -> str:
    """Charge le contexte depuis un fichier avec gestion des erreurs"""
    try:
        if not os.path.exists(file_path):
            logger.warning(f"Fichier de contexte {file_path} non trouvé")
            return "Tu es un assistant AI professionnel et serviable. Réponds en français de manière claire et concise."
        
        with open(file_path, "r", encoding="utf-8") as f:
            context = f.read().strip()
        
        if not context:
            logger.warning("Le fichier de contexte est vide")
            return "Tu es un assistant AI professionnel. Réponds en français."
        
        logger.info(f"Contexte chargé avec succès ({len(context)} caractères)")
        return context
    except Exception as e:
        logger.error(f"Erreur lors du chargement du contexte: {e}")
        return "Tu es un assistant AI professionnel."

# Chargement initial du contexte
CONTEXT = load_context()

# 2️⃣ Gestion avancée de la mémoire de conversation
class ConversationMemory:
    """Gère l'historique des conversations avec limites et optimisation"""
    
    def __init__(self, max_turns: int = 15, max_total_chars: int = 12000):
        self.max_turns = max_turns
        self.max_total_chars = max_total_chars
        self.history = deque(maxlen=max_turns * 2)
        self.total_chars = 0
    
    def add_interaction(self, user_message: str, assistant_message: str) -> None:
        """Ajoute une interaction à l'historique"""
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": assistant_message})
        
        self.total_chars += len(user_message) + len(assistant_message)
        self._cleanup_if_needed()
    
    def get_history(self) -> List[Dict[str, str]]:
        """Retourne l'historique sous forme de liste"""
        return list(self.history)
    
    def clear(self) -> None:
        """Vide l'historique"""
        self.history.clear()
        self.total_chars = 0
    
    def _cleanup_if_needed(self) -> None:
        """Nettoie l'historique si on dépasse les limites"""
        while self.total_chars > self.max_total_chars and len(self.history) > 2:
            # Supprime la plus ancienne interaction
            if len(self.history) >= 2:
                removed_user = self.history.popleft()
                removed_assistant = self.history.popleft()
                self.total_chars -= len(removed_user["content"]) + len(removed_assistant["content"])
    
    def get_summary_statistics(self) -> Dict[str, any]:
        """Retourne des statistiques sur la mémoire"""
        return {
            "turns": len(self.history) // 2,
            "total_characters": self.total_chars,
            "estimated_tokens": self.total_chars // 4,
            "max_turns": self.max_turns,
            "max_characters": self.max_total_chars
        }

# Initialisation de la mémoire
conversation_memory = ConversationMemory()

# 3️⃣ Fonction pour tester la disponibilité des modèles
def test_model_availability(model_name: str) -> bool:
    """Teste si un modèle est disponible"""
    try:
        test_response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=1
        )
        logger.info(f"Modèle {model_name} est disponible")
        return True
    except Exception as e:
        logger.warning(f"Modèle {model_name} non disponible: {e}")
        return False

# 4️⃣ Fonction améliorée pour interagir avec l'IA
def ask_ai(question: str, model_preference: str = None) -> Dict[str, any]:
    """Pose une question à l'IA avec gestion avancée des erreurs"""
    
    # Déterminer le modèle à utiliser
    if model_preference and model_preference in AVAILABLE_MODELS.values():
        selected_model = model_preference
    else:
        selected_model = AVAILABLE_MODELS["primary"]
    
    # Préparation des messages
    messages = [{"role": "system", "content": CONTEXT}]
    
    # Ajout de l'historique (limité à 5 derniers tours pour économiser des tokens)
    recent_history = list(conversation_memory.history)[-10:]  # 5 derniers tours max
    messages.extend(recent_history)
    
    # Ajout de la nouvelle question
    messages.append({"role": "user", "content": question})
    
    # Log des métriques
    total_chars = sum(len(m["content"]) for m in messages)
    estimated_tokens = total_chars // 4
    logger.info(f"Envoi de {estimated_tokens} tokens estimés avec le modèle {selected_model}")
    
    # Tentative avec le modèle principal
    models_to_try = [
        selected_model,
        AVAILABLE_MODELS["versatile"],
        AVAILABLE_MODELS["fast"],
        AVAILABLE_MODELS["fallback"]
    ]
    
    for model_attempt in models_to_try:
        try:
            start_time = time.time()
            
            response = client.chat.completions.create(
                model=model_attempt,
                messages=messages,
                temperature=0.7,
                max_tokens=1000,  # Limite raisonnable pour Groq
                top_p=0.9
            )
            
            response_time = time.time() - start_time
            answer = response.choices[0].message.content
            
            # Mise à jour de la mémoire
            conversation_memory.add_interaction(question, answer)
            
            # Log de succès
            logger.info(f"Réponse reçue en {response_time:.2f}s via {model_attempt} - {len(answer)} caractères")
            
            return {
                "answer": answer,
                "response_time": response_time,
                "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else estimated_tokens,
                "model_used": model_attempt,
                "success": True
            }
            
        except Exception as e:
            logger.warning(f"Échec avec {model_attempt}: {str(e)[:100]}...")
            continue
    
    # Si tous les modèles échouent
    logger.error("Tous les modèles ont échoué")
    return {
        "answer": "Désolé, le service est temporairement indisponible. Veuillez réessayer dans quelques instants.",
        "response_time": 0,
        "tokens_used": 0,
        "model_used": "none",
        "success": False
    }

# 5️⃣ Routes Flask améliorées
@app.route('/ask', methods=['POST'])
def ask():
    """Endpoint principal pour les questions"""
    client_ip = request.remote_addr
    logger.info(f"Requête reçue de {client_ip}")
    
    # Validation des données
    if not request.is_json:
        return jsonify({
            "status": "error",
            "message": "Le Content-Type doit être application/json"
        }), 400
    
    data = request.get_json()
    
    if not data or 'prompt' not in data:
        return jsonify({
            "status": "error",
            "message": "Le champ 'prompt' est requis"
        }), 400
    
    prompt = data['prompt'].strip()
    
    if not prompt:
        return jsonify({
            "status": "error",
            "message": "Le prompt ne peut pas être vide"
        }), 400
    
    if len(prompt) > 2000:
        return jsonify({
            "status": "error", 
            "message": "Le prompt est trop long (max 2000 caractères)"
        }), 400
    
    # Paramètre optionnel pour choisir le modèle
    model_preference = data.get('model')
    
    # Appel à l'IA
    result = ask_ai(prompt, model_preference)
    
    # Réponse structurée
    response_data = {
        "status": "success" if result["success"] else "partial",
        "prompt": prompt,
        "response": result["answer"],
        "metadata": {
            "response_time": f"{result['response_time']:.2f}s",
            "tokens_used": result["tokens_used"],
            "model_used": result["model_used"],
            "memory_stats": conversation_memory.get_summary_statistics()
        }
    }
    
    return jsonify(response_data)

@app.route('/models', methods=['GET'])
def get_available_models():
    """Retourne la liste des modèles disponibles"""
    # Tester la disponibilité des modèles
    available = {}
    for key, model in AVAILABLE_MODELS.items():
        if test_model_availability(model):
            available[key] = model
    
    return jsonify({
        "status": "success",
        "available_models": available,
        "recommended": AVAILABLE_MODELS["primary"],
        "note": "Les modèles Llama 3 70b-8192 et 8b-8192 ont été retirés par Groq"
    })

@app.route('/memory', methods=['GET', 'DELETE'])
def handle_memory():
    """Gestion de la mémoire de conversation"""
    if request.method == 'GET':
        return jsonify({
            "status": "success",
            "memory": conversation_memory.get_history(),
            "statistics": conversation_memory.get_summary_statistics()
        })
    
    elif request.method == 'DELETE':
        conversation_memory.clear()
        logger.info("Mémoire de conversation vidée")
        return jsonify({
            "status": "success",
            "message": "Mémoire de conversation vidée"
        })

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de vérification de santé"""
    # Tester la connexion à Groq
    groq_healthy = False
    try:
        test_response = client.chat.completions.create(
            model=AVAILABLE_MODELS["fallback"],
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1
        )
        groq_healthy = True
    except:
        groq_healthy = False
    
    return jsonify({
        "status": "healthy" if groq_healthy else "degraded",
        "service": "Mara Sambelahatse AI API",
        "timestamp": time.time(),
        "groq_connection": groq_healthy,
        "memory_stats": conversation_memory.get_summary_statistics(),
        "context_loaded": len(CONTEXT) > 0
    })

@app.route('/', methods=['GET'])
def home():
    """Page d'accueil avec documentation"""
    return jsonify({
        "service": "API LLM Mara Sambelahatse",
        "version": "3.0.0",
        "updated": "2026-01-14",
        "note": "Migration vers les nouveaux modèles Llama 3.2",
        "endpoints": {
            "POST /ask": "Poser une question à l'IA",
            "GET /models": "Voir les modèles disponibles",
            "GET /memory": "Voir l'historique",
            "DELETE /memory": "Effacer l'historique",
            "GET /health": "Vérifier l'état du service"
        },
        "default_model": AVAILABLE_MODELS["primary"]
    })

# 6️⃣ Gestion des erreurs globale
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "status": "error",
        "message": "Endpoint non trouvé"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Erreur serveur: {error}")
    return jsonify({
        "status": "error",
        "message": "Erreur interne du serveur"
    }), 500

# 7️⃣ Configuration du serveur
if __name__ == '__main__':
    port = int(os.getenv("PORT", 3000))
    debug_mode = os.getenv("ENVIRONMENT", "dev") == "dev"
    
    # Tester la connexion Groq au démarrage
    logger.info("Test de connexion à l'API Groq...")
    if test_model_availability(AVAILABLE_MODELS["fallback"]):
        logger.info("✅ Connexion Groq OK")
    else:
        logger.warning("⚠️  Connexion Groq problématique")
    
    if debug_mode:
        logger.info(f"🚀 Serveur démarré en mode développement sur http://localhost:{port}")
        app.run(host='0.0.0.0', port=port, debug=True)
    else:
        logger.info(f"🚀 Serveur démarré en mode production sur le port {port}")
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)