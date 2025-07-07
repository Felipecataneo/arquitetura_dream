import json
import logging
import re
import time
import ast
import subprocess
import tempfile
import os
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set, Union
from collections import defaultdict
import traceback
import hashlib
from abc import ABC, abstractmethod

# --- CONFIGURAÇÃO ROBUSTA COM SUPORTE A MÚLTIPLOS MODELOS ---

# Checagem do Ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("Biblioteca 'ollama' não encontrada. O cliente Ollama estará indisponível.")

# Checagem do OpenAI
try:
    import openai
    from openai import OpenAI
    # A chave será carregada pelo server.py usando python-dotenv
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if OPENAI_API_KEY:
        OPENAI_AVAILABLE = True
        logging.info("Chave da API OpenAI encontrada. O cliente OpenAI estará disponível.")
    else:
        OPENAI_AVAILABLE = False
        logging.warning("Váriavel de ambiente OPENAI_API_KEY não encontrada. Cliente OpenAI indisponível.")
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("Biblioteca 'openai' não encontrada. O cliente OpenAI estará indisponível.")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- SISTEMA DE VALIDAÇÃO APRIMORADO ---
class ValidationError(Exception):
    """Erro de validação customizado"""
    pass

class ResponseValidator:
    @staticmethod
    def validate_json_response(response_text: str, required_fields: List[str] = None) -> Dict:
        if not response_text:
            raise ValidationError("Resposta vazia recebida")
        
        json_patterns = [
            r'```json\s*(\{.*?\})\s*```',
            r'(\{.*\})',
            r'```\s*(\{.*?\})\s*```'
        ]
        
        extracted_json = None
        for pattern in json_patterns:
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                extracted_json = match.group(1)
                break
        
        if not extracted_json:
            extracted_json = response_text.strip()
        
        try:
            data = json.loads(extracted_json)
            
            if required_fields:
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    for field in missing_fields:
                        data[field] = ResponseValidator._get_default_value(field)
                    logging.warning(f"Campos ausentes preenchidos com padrão: {missing_fields}")
            
            if 'code' in data and len(data['code'].strip()) < 50:
                logging.warning("Código muito curto detectado - expandindo")
                data['code'] = ResponseValidator._expand_minimal_code(data['code'], data.get('language', 'python'))
            
            return data
            
        except json.JSONDecodeError as e:
            logging.error(f"Erro ao decodificar JSON: {e}")
            return ResponseValidator._create_fallback_response(response_text, required_fields)
    
    @staticmethod
    def _get_default_value(field_name: str) -> Any:
        defaults = {
            'code': '# Código não gerado devido a erro\nprint("Sistema em modo de recuperação")',
            'explanation': 'Explicação não disponível devido a erro na geração',
            'executable': True,
            'dependencies': [],
            'complexity': 'O(1)',
            'features': ['Funcionalidade básica'],
            'dream_insights': ['Gerado com fallback'],
            'improvements': ['Otimizar geração de código'],
            'confidence': 0.5,
            'classification': 'ESTABLISHED',
            'fact_or_question': 'Informação básica',
            'synthesis': 'Síntese padrão',
            'limitations': 'Gerado com fallback',
            'logical_answer': 'Não foi possível determinar.',
            'trap_analysis': 'Não foi possível analisar.',
            'reasoning': 'Raciocínio padrão'
        }
        return defaults.get(field_name, 'Valor padrão')

    @staticmethod
    def _expand_minimal_code(code: str, language: str = 'python') -> str:
        if language == 'html':
            return f'''<!DOCTYPE html><html lang="pt-BR"><head><meta charset="UTF-8"><title>Web</title></head><body>{code}</body></html>'''
        elif language == 'python':
            return f'def main():\n    {code}\n\nif __name__ == "__main__":\n    main()'
        return code
    
    @staticmethod
    def _create_fallback_response(text: str, required_fields: List[str] = None) -> Dict:
        fallback = {
            'raw_response': text, 'fallback_mode': True,
            'error': 'Falha na decodificação JSON - usando recuperação'
        }
        if required_fields:
            for field in required_fields:
                fallback[field] = ResponseValidator._get_default_value(field)
        return fallback

# --- DEFINIÇÕES FUNDAMENTAIS ---
class IntentType(Enum):
    TRIVIAL_QUERY = "Consulta Trivial"
    RIDDLE_LOGIC = "Charada / Lógica Lateral"
    FACTUAL_QUERY = "Pergunta Factual Simples"
    PLANNING_TASK = "Tarefa de Planejamento"
    ACADEMIC_SPECIALIZED = "Consulta Acadêmica / Especializada"
    CREATIVE_SYNTHESIS = "Síntese Criativa"
    PHILOSOPHICAL_INQUIRY = "Investigação Filosófica"
    CODE_GENERATION = "Geração de Código"
    UNKNOWN = "Intenção Desconhecida"

class ReasoningStrategy(Enum):
    CODE_BASED_SOLVER = "Solucionador Baseado em Código"
    RIDDLE_ANALYSIS = "Análise de Lógica de Charadas"
    NEURAL_INTUITIVE = "Intuição Neural"
    HIERARCHICAL_PLANNING = "Planejamento Hierárquico"
    DREAM_CODE_GENERATION = "Geração de Código via DREAM"
    FALLBACK_RECOVERY = "Recuperação por Fallback"

@dataclass
class CodeExecution:
    language: str = ""
    code: str = ""
    output: str = ""
    error: str = ""
    execution_time: float = 0.0
    success: bool = False

@dataclass
class CognitiveState:
    problem: str = ""
    intent: IntentType = IntentType.UNKNOWN
    strategy: Optional[ReasoningStrategy] = None
    confidence: float = 0.0
    solution: Any = None
    reasoning_trace: List[str] = field(default_factory=list)
    success: bool = False
    error: Optional[str] = None
    decision_time: float = 0.0
    code_execution: Optional[CodeExecution] = None
    generated_code: List[Dict] = field(default_factory=list)
    hierarchical_plan: Dict = field(default_factory=dict)
    pragmatic_context: Dict = field(default_factory=dict)
    meta_insights: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)
    sentiment_analysis: Dict = field(default_factory=dict)
    cache_hit: bool = False
    fallback_mode: bool = False

# --- ABSTRAÇÃO E CLIENTES DE LLM ---
class LLMClient(ABC):
    """Classe base abstrata para todos os clientes LLM."""
    def __init__(self, model: str, max_retries: int = 3, timeout: int = 60):
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.request_count = 0
        self.error_count = 0
        self.available = self._check_availability()
    
    @abstractmethod
    def _check_availability(self) -> bool:
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict], format: Optional[str] = None, temperature: float = 0.3) -> Dict:
        pass
        
    def get_stats(self) -> Dict:
        return {
            'available': self.available, 'request_count': self.request_count,
            'error_count': self.error_count,
            'success_rate': ((self.request_count - self.error_count) / max(self.request_count, 1)) * 100
        }

class RobustOllamaClient(LLMClient):
    """Cliente robusto para a API do Ollama."""
    def _check_availability(self) -> bool:
        if not OLLAMA_AVAILABLE: return False
        try:
            ollama.show(self.model)
            return True
        except Exception as e:
            logging.warning(f"Ollama ou modelo '{self.model}' não disponível: {e}")
            return False
    
    def chat(self, messages: List[Dict], format: Optional[str] = None, temperature: float = 0.3) -> Dict:
        if not self.available: raise ConnectionError(f"Ollama '{self.model}' não disponível")
        self.request_count += 1
        for attempt in range(self.max_retries):
            try:
                options = {'temperature': temperature}
                response = ollama.chat(model=self.model, messages=messages, format=format, options=options)
                return response
            except Exception as e:
                self.error_count += 1
                logging.warning(f"Ollama - Tentativa {attempt + 1} falhou: {e}")
                if attempt == self.max_retries - 1: raise
                time.sleep(2 ** attempt)
        raise RuntimeError("Ollama - Todas as tentativas falharam")

class OpenAIClient(LLMClient):
    """Cliente robusto para a API da OpenAI."""
    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        if self.available:
            self.client = OpenAI(api_key=OPENAI_API_KEY, timeout=self.timeout)

    def _check_availability(self) -> bool:
        if not OPENAI_AVAILABLE: return False
        try:
            client = OpenAI(api_key=OPENAI_API_KEY, timeout=10)
            client.models.retrieve(self.model)
            return True
        except openai.AuthenticationError:
            logging.error("OpenAI: Chave da API inválida.")
            return False
        except Exception as e:
            logging.warning(f"OpenAI não disponível: {e}")
            return False
    
    def chat(self, messages: List[Dict], format: Optional[str] = None, temperature: float = 0.3) -> Dict:
        if not self.available: raise ConnectionError("OpenAI não está disponível")
        self.request_count += 1
        params = {"model": self.model, "messages": messages, "temperature": temperature}
        if format == 'json':
            params["response_format"] = {"type": "json_object"}

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(**params)
                return {"message": {"content": response.choices[0].message.content}}
            except Exception as e:
                self.error_count += 1
                logging.warning(f"OpenAI - Tentativa {attempt + 1} falhou: {e}")
                if attempt == self.max_retries - 1: raise
                time.sleep(2 ** attempt)
        raise RuntimeError("OpenAI - Todas as tentativas falharam")


# --- SISTEMA DE CACHE INTELIGENTE --- (Sem alterações)
class IntelligentCache:
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.access_count = defaultdict(int)
        self.max_size = max_size
    
    def _hash_key(self, problem: str, intent: IntentType) -> str:
        content = f"{problem.lower().strip()}_{intent.value}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, problem: str, intent: IntentType) -> Optional[Dict]:
        key = self._hash_key(problem, intent)
        if key in self.cache:
            self.access_count[key] += 1
            return self.cache[key]
        return None
    
    def set(self, problem: str, intent: IntentType, response: Dict):
        if len(self.cache) >= self.max_size:
            self._cleanup_cache()
        key = self._hash_key(problem, intent)
        self.cache[key] = {'response': response, 'timestamp': time.time(), 'access_count': 1}
    
    def _cleanup_cache(self):
        sorted_items = sorted(self.cache.items(), key=lambda x: x[1]['access_count'])
        items_to_remove = len(sorted_items) // 4
        for key, _ in sorted_items[:items_to_remove]:
            del self.cache[key]
            if key in self.access_count: del self.access_count[key]

# --- COMPONENTES AUXILIARES ROBUSTOS ---
# NOTA: Componentes agora recebem o `client` como parâmetro do método, não no construtor.
class RobustIntentClassifier:
    def __init__(self):
        self.patterns = {
            IntentType.CODE_GENERATION: [r"(crie|create|write|desenvolva|develop|implemente|implement|faça|make|gere|generate|escreva|programa|código|code|app|script|software)"],
            IntentType.RIDDLE_LOGIC: [r"charada|riddle|enigma|pegadinha"],
            IntentType.PLANNING_TASK: [r"plano|planeje|passo a passo|step by step|como fazer"],
        }
    
    def classify(self, problem: str, client: LLMClient) -> IntentType:
        problem_lower = problem.lower()
        for intent_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, problem_lower):
                    return intent_type
        
        if client and client.available:
            try:
                prompt = f"""Classifique a intenção desta pergunta em UMA categoria: "{problem}"
Categorias: {', '.join([i.name for i in IntentType])}. Responda apenas com o nome da categoria."""
                response = client.chat(messages=[{'role': 'user', 'content': prompt}], temperature=0.0)
                intent_name = response['message']['content'].strip().upper()
                for intent in IntentType:
                    if intent.name in intent_name: return intent
            except Exception as e:
                logging.warning(f"Classificação LLM falhou: {e}")
        
        return IntentType.FACTUAL_QUERY

class SentimentAnalyzer:
    def analyze(self, problem: str) -> Dict:
        return {'urgency': 0.1, 'politeness': 0.5, 'complexity': 0.3, 'length': len(problem.split())}

# --- HANDLER DE CÓDIGO APRIMORADO ---
class RobustCodeGenerationHandler:
    def __init__(self):
        # O cliente será passado no método handle
        self.code_templates = { 'python_calculator': 'def calculator():\n    print("Calculadora Simples")\n\ncalculator()' }
    
    def handle(self, state: CognitiveState, client: LLMClient):
        state.strategy = ReasoningStrategy.DREAM_CODE_GENERATION
        state.reasoning_trace.append("🧠 DREAM: Iniciando Geração de Código")
        try:
            self._robust_problem_analysis(state)
            self._robust_code_generation(state, client)
            self._robust_validation(state)
            self._build_robust_response(state)
        except Exception as e:
            logging.error(f"Erro no handler de código: {e}", exc_info=True)
            self._create_fallback_code_response(state, str(e))

    def _robust_problem_analysis(self, state: CognitiveState):
        state.reasoning_trace.append("🔍 Análise de requisitos")
        lang = 'python'
        if any(w in state.problem.lower() for w in ['html', 'css', 'web', 'javascript']):
            lang = 'html'
        state.pragmatic_context = {'language': lang, 'project_type': 'utility'}
        state.reasoning_trace.append(f"🎯 Contexto: {state.pragmatic_context}")

    def _robust_code_generation(self, state: CognitiveState, client: LLMClient):
        state.reasoning_trace.append(f"💻 Geração de código com {client.model}")
        if not client or not client.available:
            return self._use_template_fallback(state)

        prompt = self._create_robust_prompt(state)
        try:
            response = client.chat(messages=[{'role': 'user', 'content': prompt}], format='json', temperature=0.2)
            data = ResponseValidator.validate_json_response(
                response['message']['content'], 
                required_fields=['code', 'explanation', 'executable', 'dependencies']
            )
            if not data.get('code') or len(data['code'].strip()) < 20:
                raise ValidationError("Código gerado muito curto")
            
            data['language'] = state.pragmatic_context.get('language', 'python')
            state.generated_code.append(data)
            state.reasoning_trace.append("✅ Código gerado com sucesso via LLM")
            return
        except Exception as e:
            logging.warning(f"Geração de código via LLM falhou: {e}")
            self._use_template_fallback(state)

    def _use_template_fallback(self, state: CognitiveState):
        state.reasoning_trace.append("🔄 Usando template de fallback")
        code = self.code_templates['python_calculator']
        data = {
            'code': code, 'language': 'python', 'explanation': 'Template de calculadora de fallback.',
            'executable': True, 'dependencies': [], 'complexity': 'O(1)',
            'features': ['Funcionalidade básica'], 'template_fallback': True
        }
        state.generated_code.append(data)

    def _create_robust_prompt(self, state: CognitiveState) -> str:
        lang = state.pragmatic_context.get('language', 'python')
        return f"""Você é um programador expert em {lang}. Crie um código completo e funcional para o seguinte problema:
"{state.problem}"
Responda APENAS com um JSON válido contendo: "code", "explanation", "executable" (boolean), "dependencies" (list).
O código deve ser auto-contido e pronto para executar."""

    def _robust_validation(self, state: CognitiveState):
        state.reasoning_trace.append("🔧 Validação do código")
        if not state.generated_code: raise ValidationError("Nenhum código gerado")
        
        code_result = state.generated_code[-1]
        if code_result.get('language') == 'python':
            try: ast.parse(code_result['code']); state.reasoning_trace.append("✅ Sintaxe Python válida")
            except SyntaxError as e: state.validation_errors.append(f"Erro de sintaxe: {e}")
        
        # Tenta executar código seguro
        if code_result.get('language') == 'python' and 'input(' not in code_result['code']:
            try:
                execution = self._safe_execute_python(code_result['code'])
                state.code_execution = execution
                state.reasoning_trace.append(f"🚀 Execução: {'Sucesso' if execution.success else 'Falha'}")
            except Exception as e:
                state.reasoning_trace.append(f"⚠️ Execução não realizada: {e}")

    def _safe_execute_python(self, code: str) -> CodeExecution:
        execution = CodeExecution(language='python', code=code)
        start_time = time.time()
        
        # Implementação simplificada para segurança
        dangerous_keywords = ['os.', 'sys.', 'subprocess', 'open(', 'eval(']
        if any(danger in code for danger in dangerous_keywords):
            execution.error = "Execução bloqueada por segurança."
            execution.success = False
            return execution
            
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                temp_file = f.name
                f.write(code)
            
            result = subprocess.run(['python', temp_file], capture_output=True, text=True, timeout=5)
            execution.output = result.stdout
            execution.error = result.stderr
            execution.success = result.returncode == 0
        except subprocess.TimeoutExpired:
            execution.error = "Timeout: Código demorou muito para executar."
        except Exception as e:
            execution.error = f"Erro na execução: {e}"
        finally:
            if 'temp_file' in locals() and os.path.exists(temp_file): os.unlink(temp_file)
        
        execution.execution_time = time.time() - start_time
        return execution

    def _build_robust_response(self, state: CognitiveState):
        if not state.generated_code: raise ValueError("Código indisponível")
        code_result = state.generated_code[-1]
        state.solution = f"""**🧠 DREAM CODE GENERATION**
**💻 CÓDIGO ({code_result['language'].upper()})**
```{code_result['language']}
{code_result['code']}
```
**📖 EXPLICAÇÃO:**
{code_result['explanation']}
"""
        state.success = True
        state.confidence = 0.9 if not code_result.get('template_fallback') else 0.7

    def _create_fallback_code_response(self, state: CognitiveState, error_msg: str):
        state.reasoning_trace.append("🔄 Criando resposta de fallback de código")
        state.solution = f"""**🔄 MODO DE RECUPERAÇÃO DE CÓDIGO**
**⚠️ AVISO:** Erro na geração de código: {error_msg}
**🎯 PROBLEMA:** {state.problem}
**💻 CÓDIGO FALLBACK:**
```python
# Código de fallback devido a erro.
print("Sistema em modo de recuperação para o problema: {state.problem}")
```"""
        state.success = True; state.confidence = 0.4; state.fallback_mode = True

# --- SISTEMA PRINCIPAL UNIFICADO ---
class DreamSystemV12Final:
    def __init__(self):
        # Gerenciamento de múltiplos clientes
        self.clients: Dict[str, LLMClient] = {}
        self.default_client_name: Optional[str] = None

        if OLLAMA_AVAILABLE:
            gemma_client = RobustOllamaClient('gemma3')
            if gemma_client.available:
                self.clients['gemma3'] = gemma_client
                if not self.default_client_name: self.default_client_name = 'gemma3'
        
        if OPENAI_AVAILABLE:
            # Modelo pode ser configurado aqui
            openai_client = OpenAIClient('gpt-4o-mini')
            if openai_client.available:
                self.clients['gpt-4o-mini'] = openai_client
                if not self.default_client_name: self.default_client_name = 'gpt-4o-mini'
        
        if not self.clients:
            logging.error("Nenhum cliente LLM está disponível. O sistema não pode funcionar.")
        else:
            logging.info(f"Clientes LLM disponíveis: {list(self.clients.keys())}")
            logging.info(f"Cliente padrão: {self.default_client_name}")

        # Componentes inicializados sem cliente
        self.intent_classifier = RobustIntentClassifier()
        self.code_handler = RobustCodeGenerationHandler()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.cache = IntelligentCache()
        
        self.performance_metrics = defaultdict(float)
        self.system_health = 'EXCELLENT'

    def solve_problem(self, problem: str, model_name: Optional[str] = None) -> CognitiveState:
        state = CognitiveState(problem=problem)
        start_time = time.time()
        
        # Seleciona o cliente para esta requisição
        selected_client_name = model_name if model_name in self.clients else self.default_client_name
        active_client = self.clients.get(selected_client_name)
        
        if not active_client or not active_client.available:
            state.error = f"O modelo selecionado '{selected_client_name}' não está disponível."
            self._handle_fallback(state)
            state.decision_time = time.time() - start_time
            return state

        state.reasoning_trace.append(f"🧠 Usando modelo: {selected_client_name}")

        try:
            self.performance_metrics['total_problems'] += 1
            if not problem or not problem.strip(): raise ValueError("Problema vazio")
            
            state.sentiment_analysis = self.sentiment_analyzer.analyze(problem)
            state.intent = self.intent_classifier.classify(problem, active_client)
            state.reasoning_trace.append(f"🎯 Intenção classificada: {state.intent.value}")
            
            cached = self.cache.get(problem, state.intent)
            if cached:
                state.cache_hit = True; state.solution = cached['response']['solution']
                state.confidence = cached['response']['confidence']; state.success = True
                state.reasoning_trace.append("🎯 Resposta recuperada do cache")
                state.decision_time = time.time() - start_time
                return state

            # Roteador de tarefas
            if state.intent == IntentType.CODE_GENERATION:
                self.code_handler.handle(state, active_client)
            elif state.intent == IntentType.RIDDLE_LOGIC:
                self._handle_riddle_logic(state, active_client)
            else: # Factual, Trivial, etc.
                self._handle_general_query(state, active_client)

            if state.success:
                self.performance_metrics['successful_solutions'] += 1
                self.cache.set(problem, state.intent, {'solution': state.solution, 'confidence': state.confidence})

        except Exception as e:
            logging.error(f"Erro no processamento: {e}", exc_info=True)
            state.error = str(e); state.success = False
            self._handle_fallback(state)

        state.decision_time = time.time() - start_time
        return state

    def _handle_riddle_logic(self, state: CognitiveState, client: LLMClient):
        state.strategy = ReasoningStrategy.RIDDLE_ANALYSIS
        state.reasoning_trace.append("🧩 Analisando charada")
        prompt = f"""Analise esta charada: "{state.problem}". Responda em JSON com "logical_answer" e "explanation"."""
        try:
            response = client.chat(messages=[{'role': 'user', 'content': prompt}], format='json', temperature=0.1)
            data = ResponseValidator.validate_json_response(response['message']['content'], ['logical_answer', 'explanation'])
            state.solution = f"**💡 RESPOSTA:** {data['logical_answer']}\n**📖 EXPLICAÇÃO:** {data['explanation']}"
            state.success = True; state.confidence = 0.9
        except Exception as e:
            logging.error(f"Erro ao analisar charada: {e}")
            self._handle_fallback(state)

    def _handle_general_query(self, state: CognitiveState, client: LLMClient):
        state.strategy = ReasoningStrategy.NEURAL_INTUITIVE
        state.reasoning_trace.append("🧠 Processando consulta geral")
        prompt = f"Responda à seguinte pergunta de forma clara e precisa: {state.problem}"
        try:
            response = client.chat(messages=[{'role': 'user', 'content': prompt}], temperature=0.5)
            state.solution = f"**💡 RESPOSTA:**\n{response['message']['content']}"
            state.success = True; state.confidence = 0.85
        except Exception as e:
            logging.error(f"Erro na consulta geral: {e}")
            self._handle_fallback(state)
            
    def _handle_fallback(self, state: CognitiveState):
        state.strategy = ReasoningStrategy.FALLBACK_RECOVERY
        state.reasoning_trace.append("🔄 Executando fallback")
        state.solution = f"""**🔄 MODO FALLBACK**
Não foi possível processar a sua solicitação: "{state.problem}"
Motivo: {state.error or 'Intenção não suportada ou erro interno.'}
Tente reformular sua pergunta."""
        state.success = True
        state.confidence = 0.3
        state.fallback_mode = True

# --- MAIN PRINCIPAL (PARA TESTE NO CONSOLE) ---
if __name__ == "__main__":
    print("="*50)
    print("🧠 DREAM V12.3 - MODO DE TESTE NO CONSOLE")
    print("="*50)
    
    try:
        agent = DreamSystemV12Final()
        if not agent.clients:
            print("🚨 Nenhum modelo de LLM disponível. Encerrando.")
            exit()
            
        print(f"✅ Sistema inicializado. Modelos disponíveis: {list(agent.clients.keys())}")
        
        while True:
            user_input = input("\n> ").strip()
            if user_input.lower() in ['sair', 'exit']: break
            
            # Para teste, pode-se prefixar com o nome do modelo
            # Ex: gpt-4o-mini:crie um hello world
            model_to_use = None
            if ":" in user_input:
                model_name, problem_text = user_input.split(":", 1)
                if model_name in agent.clients:
                    model_to_use = model_name
                    user_input = problem_text.strip()
            
            result = agent.solve_problem(user_input, model_name=model_to_use)
            
            print("\n--- RELATÓRIO ---")
            print(f"PROBLEMA: {result.problem}")
            print(f"SUCESSO: {'Sim' if result.success else 'Não'}")
            print(f"CONFIANÇA: {result.confidence:.2f}")
            print(f"ESTRATÉGIA: {result.strategy.value if result.strategy else 'N/A'}")
            print("--- SOLUÇÃO ---")
            print(result.solution)
            print("-----------------")

    except Exception as e:
        print(f"\n❌ Erro fatal no sistema: {e}")
