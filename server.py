# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from dataclasses import asdict
from enum import Enum
from typing import Any

# Importe a sua classe principal do seu arquivo
from arquiteto_final import DreamSystemV12Final

app = Flask(__name__)
CORS(app)

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

print("🧠 Initializing DREAM V12.2 System...")
try:
    agent = DreamSystemV12Final(ollama_model='gemma3')
    print("✅ DREAM System Initialized Successfully.")
except Exception as e:
    print(f"🚨 FATAL ERROR during initialization: {e}")
    agent = None

# --- NOVA FUNÇÃO AUXILIAR ---
def convert_enums_to_strings(data: Any) -> Any:
    """
    Percorre recursivamente um dicionário ou lista e converte todos os
    objetos Enum em seus valores de string.
    """
    if isinstance(data, dict):
        return {key: convert_enums_to_strings(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_enums_to_strings(item) for item in data]
    elif isinstance(data, Enum):
        # Converte o Enum para seu valor (a string)
        return data.value
    else:
        # Retorna o valor como está se não for um dict, list, ou Enum
        return data

@app.route('/solve', methods=['POST'])
def solve():
    if not agent:
        return jsonify({"error": "DREAM system is not available due to an initialization error."}), 500

    data = request.json
    problem = data.get('problem')
    
    if not problem:
        return jsonify({"error": "No 'problem' field provided"}), 400
    
    try:
        print(f"\n🔄 Received problem: '{problem}'")
        cognitive_state = agent.solve_problem(problem)
        
        # Converte o dataclass para um dicionário
        response_dict = asdict(cognitive_state)
        
        # --- PASSO CRÍTICO ADICIONADO ---
        # Converte todos os Enums no dicionário para strings
        serializable_response = convert_enums_to_strings(response_dict)
        
        print(f"✅ Responded successfully. Strategy: {serializable_response.get('strategy', 'N/A')}")
        
        # Agora o jsonify funcionará perfeitamente
        return jsonify(serializable_response)
        
    except Exception as e:
        logging.error(f"Error processing request: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred", "details": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Flask server for DREAM V12.2...")
    print("   API Endpoint available at http://127.0.0.1:5000/solve")
    app.run(host='0.0.0.0', port=5000)