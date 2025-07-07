# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from dataclasses import asdict
from enum import Enum
from typing import Any
from dotenv import load_dotenv # <-- NOVA IMPORTAÇÃO
import os # <-- NOVA IMPORTAÇÃO

# --- CARREGAR VARIÁVEIS DE AMBIENTE ---
load_dotenv()

# Importe a sua classe principal do seu arquivo
from arquiteto_final import DreamSystemV12Final, LLMClient

app = Flask(__name__)
CORS(app)

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

print("🧠 Initializing DREAM V12.3 System...")
try:
    # A inicialização agora não precisa de parâmetros
    agent = DreamSystemV12Final()
    if agent.clients:
        print("✅ DREAM System Initialized Successfully.")
        print(f"   Available models: {list(agent.clients.keys())}")
    else:
        print("🚨 FATAL ERROR: No LLM clients could be initialized.")
        agent = None

except Exception as e:
    print(f"🚨 FATAL ERROR during initialization: {e}")
    agent = None

def convert_enums_to_strings(data: Any) -> Any:
    if isinstance(data, dict):
        return {key: convert_enums_to_strings(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_enums_to_strings(item) for item in data]
    elif isinstance(data, Enum):
        return data.value
    else:
        return data

# --- NOVO ENDPOINT ---
@app.route('/available_models', methods=['GET'])
def get_available_models():
    if not agent:
        return jsonify({"error": "DREAM system is not available."}), 500
    
    available_models = list(agent.clients.keys())
    return jsonify(available_models)

@app.route('/solve', methods=['POST'])
def solve():
    if not agent:
        return jsonify({"error": "DREAM system is not available due to an initialization error."}), 500

    data = request.json
    problem = data.get('problem')
    model_name = data.get('model') # <-- NOVO: Obter nome do modelo
    
    if not problem:
        return jsonify({"error": "No 'problem' field provided"}), 400
    
    try:
        print(f"\n🔄 Received problem: '{problem}' for model: '{model_name or 'default'}'")
        # --- NOVO: Passar o nome do modelo para o agente ---
        cognitive_state = agent.solve_problem(problem, model_name=model_name)
        
        response_dict = asdict(cognitive_state)
        serializable_response = convert_enums_to_strings(response_dict)
        
        print(f"✅ Responded successfully. Strategy: {serializable_response.get('strategy', 'N/A')}")
        
        return jsonify(serializable_response)
        
    except Exception as e:
        logging.error(f"Error processing request: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred", "details": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Flask server for DREAM V12.2...")
    print("   API Endpoints available at:")
    print("   - http://127.0.0.1:5000/solve (POST)")
    print("   - http://127.0.0.1:5000/available_models (GET)")
    app.run(host='0.0.0.0', port=5000)