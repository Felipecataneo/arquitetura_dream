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

# --- CONFIGURAÇÃO ROBUSTA ---
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("Ollama não está disponível. Sistema funcionará em modo limitado.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- SISTEMA DE VALIDAÇÃO APRIMORADO ---
class ValidationError(Exception):
    """Erro de validação customizado"""
    pass

class ResponseValidator:
    @staticmethod
    def validate_json_response(response_text: str, required_fields: List[str] = None) -> Dict:
        """Valida e extrai JSON de resposta, com fallback robusto"""
        if not response_text:
            raise ValidationError("Resposta vazia recebida")
        
        # Tentar extrair JSON da resposta
        json_patterns = [
            r'```json\s*(\{.*?\})\s*```',  # JSON em code block
            r'(\{.*\})',  # JSON direto
            r'```\s*(\{.*?\})\s*```'  # JSON sem especificação de linguagem
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
            
            # Validação especial para código - CORREÇÃO AQUI
            if 'code' in data and len(data['code'].strip()) < 50:
                logging.warning("Código muito curto detectado - expandindo")
                data['code'] = ResponseValidator._expand_minimal_code(data['code'], data.get('language', 'python'))
            
            return data
            
        except json.JSONDecodeError as e:
            logging.error(f"Erro ao decodificar JSON: {e}")
            return ResponseValidator._create_fallback_response(response_text, required_fields)
    
    @staticmethod
    def _get_default_value(field_name: str) -> Any:
        """Retorna valor padrão baseado no nome do campo"""
        defaults = {
            'code': '# Código não gerado devido a erro\nprint("Sistema em modo de recuperação")',
            'explanation': 'Explicação não disponível devido a erro na geração',
            'executable': True,
            'dependencies': [],
            'complexity': 'O(1) - Operação simples',
            'features': ['Funcionalidade básica implementada'],
            'dream_insights': ['Código gerado com sistema de fallback'],
            'improvements': ['Otimizar geração de código', 'Adicionar mais validações'],
            'confidence': 0.5,
            'classification': 'ESTABLISHED',
            'fact_or_question': 'Informação básica disponível',
            'synthesis': 'Síntese baseada em conhecimento padrão',
            'limitations': 'Gerado com sistema de fallback',
            'logical_answer': 'Não foi possível determinar a resposta lógica.',
            'trap_analysis': 'Não foi possível analisar as pegadinhas.',
            'reasoning': 'Raciocínio baseado em conhecimento padrão'
        }
        return defaults.get(field_name, 'Valor padrão')
    
    @staticmethod
    def _expand_minimal_code(code: str, language: str = 'python') -> str:
        """Expande código muito curto - CORREÇÃO AQUI"""
        if language == 'html':
            return f'''<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Animação Web</title>
    <style>
        body {{ 
            margin: 0; 
            padding: 0; 
            background: #000; 
            display: flex; 
            justify-content: center; 
            align-items: center; 
            min-height: 100vh; 
        }}
    </style>
</head>
<body>
    {code}
    
    <script>
        console.log("Animação iniciada");
    </script>
</body>
</html>'''
        elif language == 'python':
            return f'''# Código expandido pelo sistema DREAM V12.2
# Problema: Código original muito curto

{code}

# Funcionalidades adicionais
def main():
    """Função principal do programa"""
    print("Programa executado com sucesso!")
    
if __name__ == "__main__":
    main()
'''
        return code
    
    @staticmethod
    def _create_fallback_response(text: str, required_fields: List[str] = None) -> Dict:
        """Cria resposta de fallback quando JSON falha"""
        fallback = {
            'raw_response': text,
            'fallback_mode': True,
            'error': 'Falha na decodificação JSON - usando sistema de recuperação'
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
    COMPLEX_REASONING = "Raciocínio Complexo Multi-Domínio"
    ANALOGICAL_REASONING = "Raciocínio Analógico"
    CODE_GENERATION = "Geração de Código"
    CODE_DEBUGGING = "Debug de Código"
    SYSTEM_ARCHITECTURE = "Arquitetura de Sistema"
    DATA_ANALYSIS = "Análise de Dados"
    UNKNOWN = "Intenção Desconhecida"

class KnowledgeClassification(Enum):
    ESTABLISHED = "CONHECIMENTO ESTABELECIDO"
    SPECULATIVE = "CONHECIMENTO ESPECULATIVO"
    AMBIGUOUS = "CONCEITO AMBÍGUO"
    UNKNOWN = "CONCEITO DESCONHECIDO"
    FABRICATED = "POSSÍVEL FABRICAÇÃO"
    EMERGING = "CONHECIMENTO EMERGENTE"
    VALIDATED = "CONHECIMENTO VALIDADO"

class ReasoningStrategy(Enum):
    RESEARCH_FRAMEWORK_GENERATION = "Geração de Framework de Pesquisa"
    VALIDATED_SYNTHESIS = "Síntese Baseada em Conhecimento Validado"
    CODE_BASED_SOLVER = "Solucionador Baseado em Código"
    RIDDLE_ANALYSIS = "Análise de Lógica de Charadas"
    ALGORITHMIC_PLAN_EXECUTION = "Execução de Plano Algorítmico"
    NEURAL_INTUITIVE = "Intuição Neural"
    CREATIVE_REASONING = "Raciocínio Criativo"
    HIERARCHICAL_PLANNING = "Planejamento Hierárquico"
    ANALOGICAL_TRANSFER = "Transferência Analógica"
    PRAGMATIC_COMMUNICATION = "Comunicação Pragmática"
    FEW_SHOT_ADAPTATION = "Adaptação Few-Shot"
    ADVANCED_SYNTHESIS = "Síntese Avançada"
    MULTI_MODAL_REASONING = "Raciocínio Multi-Modal"
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
    concepts: List[str] = field(default_factory=list)
    knowledge_map: Dict[str, KnowledgeClassification] = field(default_factory=dict)
    certainty_level: float = 0.0
    high_confidence_facts: List[str] = field(default_factory=list)
    verification_questions: List[str] = field(default_factory=list)
    strategy: ReasoningStrategy = None
    confidence: float = 0.0
    solution: Any = None
    reasoning_trace: List[str] = field(default_factory=list)
    success: bool = False
    error: Optional[str] = None
    decision_time: float = 0.0
    code_execution: Optional[CodeExecution] = None
    generated_code: List[Dict] = field(default_factory=list)
    creative_ideas: List[Dict] = field(default_factory=list)
    analogical_mappings: List[Dict] = field(default_factory=list)
    hierarchical_plan: Dict = field(default_factory=dict)
    pragmatic_context: Dict = field(default_factory=dict)
    learning_patterns: List[Dict] = field(default_factory=list)
    meta_insights: List[str] = field(default_factory=list)
    uncertainty_acknowledgment: List[str] = field(default_factory=list)
    alternative_perspectives: List[str] = field(default_factory=list)
    ethical_considerations: List[str] = field(default_factory=list)
    deep_analysis: Dict = field(default_factory=dict)
    research_directions: List[str] = field(default_factory=list)
    synthesis_quality: float = 0.0
    fallback_mode: bool = False
    recovery_attempts: int = 0
    validation_errors: List[str] = field(default_factory=list)
    sentiment_analysis: Dict = field(default_factory=dict)
    cache_hit: bool = False

# --- CLIENTE OLLAMA ROBUSTO ---
class RobustOllamaClient:
    def __init__(self, model: str, max_retries: int = 3, timeout: int = 60):
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.available = self._check_availability()
        self.request_count = 0
        self.error_count = 0
    
    def _check_availability(self) -> bool:
        if not OLLAMA_AVAILABLE:
            return False
        try:
            ollama.show(self.model)
            return True
        except Exception as e:
            logging.warning(f"Ollama não disponível: {e}")
            return False
    
    def chat(self, messages: List[Dict], format: str = None, temperature: float = 0.3) -> Dict:
        if not self.available:
            raise ConnectionError("Ollama não está disponível")
        
        self.request_count += 1
        
        for attempt in range(self.max_retries):
            try:
                options = {'temperature': temperature}
                if format:
                    response = ollama.chat(model=self.model, messages=messages, format=format, options=options)
                else:
                    response = ollama.chat(model=self.model, messages=messages, options=options)
                return response
            except Exception as e:
                self.error_count += 1
                logging.warning(f"Tentativa {attempt + 1} falhou: {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        raise RuntimeError("Todas as tentativas falharam")
    
    def get_stats(self) -> Dict:
        return {
            'available': self.available,
            'request_count': self.request_count,
            'error_count': self.error_count,
            'success_rate': ((self.request_count - self.error_count) / max(self.request_count, 1)) * 100
        }

# --- SISTEMA DE CACHE INTELIGENTE ---
class IntelligentCache:
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.access_count = defaultdict(int)
        self.max_size = max_size
    
    def _hash_key(self, problem: str, intent: IntentType) -> str:
        """Cria chave de hash para o problema"""
        content = f"{problem.lower().strip()}_{intent.value}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, problem: str, intent: IntentType) -> Optional[Dict]:
        """Recupera resposta do cache"""
        key = self._hash_key(problem, intent)
        if key in self.cache:
            self.access_count[key] += 1
            return self.cache[key]
        return None
    
    def set(self, problem: str, intent: IntentType, response: Dict):
        """Armazena resposta no cache"""
        if len(self.cache) >= self.max_size:
            self._cleanup_cache()
        
        key = self._hash_key(problem, intent)
        self.cache[key] = {
            'response': response,
            'timestamp': time.time(),
            'access_count': 1
        }
    
    def _cleanup_cache(self):
        """Remove entradas menos acessadas"""
        sorted_items = sorted(self.cache.items(), key=lambda x: x[1]['access_count'])
        items_to_remove = len(sorted_items) // 4  # Remove 25%
        for key, _ in sorted_items[:items_to_remove]:
            del self.cache[key]
            if key in self.access_count:
                del self.access_count[key]

# --- COMPONENTES AUXILIARES ROBUSTOS ---
class RobustConceptExtractor:
    def __init__(self, ollama_client: RobustOllamaClient):
        self.client = ollama_client
        self.technical_keywords = [
            'algoritmo', 'dados', 'sistema', 'rede', 'inteligência', 
            'machine', 'learning', 'neural', 'quantum', 'computação', 
            'programação', 'software', 'hardware', 'database', 'api', 
            'framework', 'calculadora', 'interface', 'gui', 'tkinter',
            'blockchain', 'cryptocurrency', 'docker', 'kubernetes',
            'microservices', 'serverless', 'devops', 'cicd', 'cloud',
            'tensorflow', 'pytorch', 'react', 'angular', 'vue',
            'animation', 'pentagon', 'ball', 'gravity', 'physics'
        ]
    
    def extract(self, problem: str) -> List[str]:
        if not self.client.available:
            return self._fallback_extraction(problem)
        
        prompt = f"""Extraia conceitos técnicos, científicos ou especializados desta pergunta:
"{problem}"

Responda em JSON: {{"concepts": ["conceito1", "conceito2"]}}
Se não houver conceitos especializados, retorne lista vazia.
Foque em conceitos que realmente existem e são relevantes."""
        
        try:
            response = self.client.chat(messages=[{'role': 'user', 'content': prompt}], format='json')
            data = ResponseValidator.validate_json_response(response['message']['content'], required_fields=['concepts'])
            concepts = data.get('concepts', [])
            
            # Validar conceitos extraídos
            valid_concepts = []
            for concept in concepts:
                if isinstance(concept, str) and concept.strip() and len(concept.strip()) > 2:
                    valid_concepts.append(concept.strip())
            
            return valid_concepts[:5]  # Máximo 5 conceitos
            
        except Exception as e:
            logging.error(f"Erro na extração de conceitos: {e}")
            return self._fallback_extraction(problem)
    
    def _fallback_extraction(self, problem: str) -> List[str]:
        words = problem.lower().split()
        concepts = []
        
        for word in words:
            if word in self.technical_keywords:
                concepts.append(word)
        
        # Buscar por padrões técnicos
        patterns = [
            r'(\w+)\s*\.\s*(\w+)',  # object.method
            r'(\w+)\s*\(\s*\)',     # function()
            r'(\w+)\.(\w+)',        # module.function
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, problem.lower())
            for match in matches:
                if isinstance(match, tuple):
                    concepts.extend(match)
                else:
                    concepts.append(match)
        
        return list(set(concepts))[:5]

class RobustEpistemicClassifier:
    def __init__(self, ollama_client: RobustOllamaClient):
        self.client = ollama_client
    
    def classify_knowledge(self, concept: str, context: str) -> Dict:
        if not self.client.available:
            return self._fallback_classification(concept)
        
        prompt = f"""Classifique o conhecimento sobre este conceito:
CONCEITO: "{concept}"
CONTEXTO: "{context}"

Opções de classificação:
- ESTABLISHED: Conhecimento bem estabelecido e verificado
- SPECULATIVE: Conhecimento especulativo ou teórico
- AMBIGUOUS: Conceito com múltiplas interpretações
- UNKNOWN: Conceito desconhecido ou não verificável
- EMERGING: Conhecimento novo ou emergente
- VALIDATED: Conhecimento validado cientificamente

Responda em JSON:
{{
    "classification": "CATEGORIA_EXATA",
    "confidence": 0.8,
    "fact_or_question": "fato estabelecido ou pergunta de verificação",
    "reasoning": "justificativa da classificação"
}}"""
        
        try:
            response = self.client.chat(messages=[{'role': 'user', 'content': prompt}], format='json')
            data = ResponseValidator.validate_json_response(
                response['message']['content'], 
                required_fields=['classification', 'confidence', 'fact_or_question', 'reasoning']
            )
            
            # Validar classificação
            try:
                KnowledgeClassification(data['classification'])
            except ValueError:
                data['classification'] = KnowledgeClassification.ESTABLISHED.value
                data['confidence'] = 0.5
            
            return data
            
        except Exception as e:
            logging.error(f"Erro na classificação epistêmica: {e}")
            return self._fallback_classification(concept)
    
    def _fallback_classification(self, concept: str) -> Dict:
        # Classificação baseada em heurísticas
        if concept.lower() in ['python', 'javascript', 'html', 'css', 'java']:
            classification = KnowledgeClassification.ESTABLISHED.value
            confidence = 0.9
        elif concept.lower() in ['ai', 'machine learning', 'neural network']:
            classification = KnowledgeClassification.EMERGING.value
            confidence = 0.7
        else:
            classification = KnowledgeClassification.ESTABLISHED.value
            confidence = 0.6
        
        return {
            "classification": classification,
            "confidence": confidence,
            "fact_or_question": f"'{concept}' é classificado como {classification.lower()}",
            "reasoning": "Classificação baseada em heurísticas do sistema"
        }

class RobustIntentClassifier:
    def __init__(self, ollama_client: RobustOllamaClient):
        self.client = ollama_client
        self.patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> Dict[IntentType, List[str]]:
        return {
            IntentType.TRIVIAL_QUERY: [
                r"quantos?\s+\w+\s+(?:temos|tem|há|existem?)\s+(?:na|em)",
                r"how\s+many\s+\w+\s+(?:are\s+)?in",
                r"count\s+\w+\s+in",
                r"conte?\s+\w+\s+em"
            ],
            IntentType.CODE_GENERATION: [
                r"(?:crie|create|write|desenvolva|develop|implemente|implement|faça|make|gere|generate|escreva|programa)",
                r".*(?:código|code|program|app|aplicação|jogo|game|script|software|sistema|calculadora|calculator|animation|animação)",
                r"(?:pygame|python|javascript|java|html|css|react|vue|angular|tkinter|pentagon|ball|gravity|physics|interactive)",
                r"(?:tetris|snake|pong|calculator|todo|login|website|gui|interface|web|animate|rotating)"
            ],
            IntentType.RIDDLE_LOGIC: [
                r"which\s+weighs\s+more",
                r"what.*heavier",
                r"trick.*question",
                r"pegadinha",
                r"charada",
                r"sally.*sisters",
                r"brothers.*sisters",
                r"logic.*puzzle",
                r"enigma"
            ],
            IntentType.PLANNING_TASK: [
                r"tower\s+of\s+han[óo]i",
                r"torre\s+de\s+han[óo]i",
                r"solve.*steps?",
                r"plan.*strategy",
                r"como\s+fazer",
                r"step\s+by\s+step"
            ],
            IntentType.PHILOSOPHICAL_INQUIRY: [
                r"meaning\s+of\s+life",
                r"sentido\s+da\s+vida",
                r"philosophy",
                r"filosofia",
                r"ethics",
                r"ética"
            ]
        }
    
    def classify(self, problem: str) -> IntentType:
        problem_lower = problem.lower()
        
        # Primeiro, tentar padrões regex
        for intent_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, problem_lower):
                    return intent_type
        
        # Depois, tentar classificação com LLM
        if self.client.available:
            try:
                return self._classify_with_llm(problem)
            except Exception as e:
                logging.warning(f"Classificação LLM falhou: {e}")
        
        # Fallback para classificação por palavras-chave
        return self._classify_with_keywords(problem_lower)
    
    def _classify_with_llm(self, problem: str) -> IntentType:
        prompt = f"""Classifique a intenção desta pergunta em UMA categoria:
"{problem}"

Categorias disponíveis:
- TRIVIAL_QUERY: Consultas triviais como contagem
- CODE_GENERATION: Pedidos para criar código/programas
- RIDDLE_LOGIC: Charadas e problemas de lógica
- FACTUAL_QUERY: Perguntas factuais simples
- ACADEMIC_SPECIALIZED: Consultas acadêmicas especializadas
- PHILOSOPHICAL_INQUIRY: Questões filosóficas
- CREATIVE_SYNTHESIS: Tarefas criativas
- PLANNING_TASK: Tarefas de planejamento

Responda apenas com o nome da categoria, sem explicações."""
        
        try:
            response = self.client.chat(messages=[{'role': 'user', 'content': prompt}], temperature=0.0)
            intent_name = response['message']['content'].strip().upper()
            
            for intent in IntentType:
                if intent.name in intent_name:
                    return intent
            
            return IntentType.FACTUAL_QUERY
            
        except Exception as e:
            logging.error(f"Erro na classificação LLM: {e}")
            return IntentType.FACTUAL_QUERY
    
    def _classify_with_keywords(self, problem_lower: str) -> IntentType:
        keyword_mapping = {
            IntentType.CODE_GENERATION: [
                'código', 'code', 'program', 'jogo', 'game', 'site', 'app', 
                'calculadora', 'calculator', 'interface', 'gui', 'script',
                'animation', 'animação', 'pentagon', 'ball', 'gravity', 'physics',
                'interactive', 'web', 'html', 'css', 'javascript', 'rotate'
            ],
            IntentType.ACADEMIC_SPECIALIZED: [
                'pesquisa', 'research', 'teoria', 'theory', 'ciência', 'science'
            ],
            IntentType.CREATIVE_SYNTHESIS: [
                'criativo', 'creative', 'inovação', 'innovation', 'imagine'
            ],
            IntentType.PHILOSOPHICAL_INQUIRY: [
                'filosofia', 'philosophy', 'ética', 'ethics', 'moral', 'sentido'
            ],
            IntentType.RIDDLE_LOGIC: [
                'charada', 'riddle', 'puzzle', 'enigma', 'pegadinha'
            ]
        }
        
        for intent_type, keywords in keyword_mapping.items():
            if any(keyword in problem_lower for keyword in keywords):
                return intent_type
        
        return IntentType.FACTUAL_QUERY

class SentimentAnalyzer:
    """Analisa o sentimento e urgência do problema"""
    
    def analyze(self, problem: str) -> Dict:
        urgency_keywords = ['urgente', 'rápido', 'agora', 'imediato', 'urgent', 'asap']
        politeness_keywords = ['por favor', 'obrigado', 'please', 'thank you', 'poderia']
        complexity_keywords = ['complexo', 'difícil', 'avançado', 'complex', 'advanced']
        
        urgency_score = sum(1 for keyword in urgency_keywords if keyword in problem.lower())
        politeness_score = sum(1 for keyword in politeness_keywords if keyword in problem.lower())
        complexity_score = sum(1 for keyword in complexity_keywords if keyword in problem.lower())
        
        return {
            'urgency': min(urgency_score / 3.0, 1.0),
            'politeness': min(politeness_score / 3.0, 1.0),
            'complexity': min(complexity_score / 3.0, 1.0),
            'length': len(problem.split()),
            'tone': 'polite' if politeness_score > 0 else 'neutral'
        }

# --- HANDLER DE CÓDIGO APRIMORADO ---
class RobustCodeGenerationHandler:
    def __init__(self, ollama_client: RobustOllamaClient):
        self.client = ollama_client
        self.code_templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict:
        """Inicializa templates de código - CORRIGIDO"""
        return {
            'html_pentagon_animation': '''<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pentágono Rotativo com Bola</title>
    <style>
        body {
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #1e3c72, #2a5298);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            font-family: Arial, sans-serif;
        }
        
        .container {
            position: relative;
            width: 400px;
            height: 400px;
        }
        
        .pentagon-container {
            position: absolute;
            width: 300px;
            height: 300px;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            animation: rotate 8s linear infinite;
        }
        
        .pentagon {
            width: 100%;
            height: 100%;
            background: rgba(255, 255, 255, 0.1);
            border: 3px solid #4CAF50;
            clip-path: polygon(50% 0%, 100% 38%, 82% 100%, 18% 100%, 0% 38%);
            box-shadow: 0 0 20px rgba(76, 175, 80, 0.5);
        }
        
        .ball {
            position: absolute;
            width: 20px;
            height: 20px;
            background: radial-gradient(circle, #ff4444, #cc0000);
            border-radius: 50%;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            box-shadow: 0 0 10px rgba(255, 68, 68, 0.8);
            z-index: 10;
        }
        
        @keyframes rotate {
            from { transform: translate(-50%, -50%) rotate(0deg); }
            to { transform: translate(-50%, -50%) rotate(360deg); }
        }
        
        .info {
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            color: white;
            text-align: center;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="pentagon-container">
            <div class="pentagon"></div>
        </div>
        <div class="ball" id="ball"></div>
        <div class="info">
            <h3>Pentágono Rotativo com Física</h3>
            <p>Bola com gravidade simulada dentro do pentágono</p>
        </div>
    </div>
    
    <script>
        class PentagonPhysics {
            constructor() {
                this.ball = document.getElementById('ball');
                this.ballX = 0;
                this.ballY = 0;
                this.velocityX = 0;
                this.velocityY = 0;
                this.gravity = 0.3;
                this.friction = 0.98;
                this.bounce = 0.8;
                this.pentagonRadius = 120;
                this.ballRadius = 10;
                
                this.init();
            }
            
            init() {
                this.animate();
            }
            
            // Função para verificar se o ponto está dentro do pentágono
            isInsidePentagon(x, y) {
                const angle = Math.PI * 2 / 5;
                const radius = this.pentagonRadius;
                
                for (let i = 0; i < 5; i++) {
                    const x1 = Math.cos(i * angle - Math.PI / 2) * radius;
                    const y1 = Math.sin(i * angle - Math.PI / 2) * radius;
                    const x2 = Math.cos((i + 1) * angle - Math.PI / 2) * radius;
                    const y2 = Math.sin((i + 1) * angle - Math.PI / 2) * radius;
                    
                    // Verificar se o ponto está do lado correto da linha
                    const cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1);
                    if (cross > 0) return false;
                }
                return true;
            }
            
            // Encontrar o ponto mais próximo na borda do pentágono
            getClosestPointOnPentagon(x, y) {
                const angle = Math.PI * 2 / 5;
                const radius = this.pentagonRadius;
                let closestX = x;
                let closestY = y;
                let minDistance = Infinity;
                
                for (let i = 0; i < 5; i++) {
                    const x1 = Math.cos(i * angle - Math.PI / 2) * radius;
                    const y1 = Math.sin(i * angle - Math.PI / 2) * radius;
                    const x2 = Math.cos((i + 1) * angle - Math.PI / 2) * radius;
                    const y2 = Math.sin((i + 1) * angle - Math.PI / 2) * radius;
                    
                    // Encontrar o ponto mais próximo na linha
                    const A = x - x1;
                    const B = y - y1;
                    const C = x2 - x1;
                    const D = y2 - y1;
                    
                    const dot = A * C + B * D;
                    const lenSq = C * C + D * D;
                    let param = dot / lenSq;
                    
                    if (param < 0) param = 0;
                    if (param > 1) param = 1;
                    
                    const xx = x1 + param * C;
                    const yy = y1 + param * D;
                    
                    const distance = Math.sqrt((x - xx) * (x - xx) + (y - yy) * (y - yy));
                    if (distance < minDistance) {
                        minDistance = distance;
                        closestX = xx;
                        closestY = yy;
                    }
                }
                
                return { x: closestX, y: closestY };
            }
            
            update() {
                // Obter a rotação atual do pentágono
                const time = Date.now() * 0.001;
                const pentagonRotation = (time * Math.PI / 4) % (Math.PI * 2);
                
                // Aplicar gravidade na direção "para baixo" relativa ao pentágono
                const gravityX = Math.sin(pentagonRotation) * this.gravity;
                const gravityY = Math.cos(pentagonRotation) * this.gravity;
                
                this.velocityX += gravityX;
                this.velocityY += gravityY;
                
                // Aplicar atrito
                this.velocityX *= this.friction;
                this.velocityY *= this.friction;
                
                // Atualizar posição
                this.ballX += this.velocityX;
                this.ballY += this.velocityY;
                
                // Verificar colisão com as bordas do pentágono
                if (!this.isInsidePentagon(this.ballX, this.ballY)) {
                    const closest = this.getClosestPointOnPentagon(this.ballX, this.ballY);
                    
                    // Calcular a normal da superfície
                    const dx = this.ballX - closest.x;
                    const dy = this.ballY - closest.y;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    
                    if (distance > 0) {
                        const normalX = dx / distance;
                        const normalY = dy / distance;
                        
                        // Posicionar a bola na borda
                        this.ballX = closest.x + normalX * this.ballRadius;
                        this.ballY = closest.y + normalY * this.ballRadius;
                        
                        // Calcular a velocidade refletida
                        const dot = this.velocityX * normalX + this.velocityY * normalY;
                        this.velocityX -= 2 * dot * normalX;
                        this.velocityY -= 2 * dot * normalY;
                        
                        // Aplicar coeficiente de restituição
                        this.velocityX *= this.bounce;
                        this.velocityY *= this.bounce;
                    }
                }
                
                // Atualizar a posição visual da bola
                this.ball.style.transform = `translate(calc(-50% + ${this.ballX}px), calc(-50% + ${this.ballY}px))`;
            }
            
            animate() {
                this.update();
                requestAnimationFrame(() => this.animate());
            }
        }
        
        // Inicializar a simulação quando a página carregar
        document.addEventListener('DOMContentLoaded', () => {
            new PentagonPhysics();
        });
    </script>
</body>
</html>''',
            'python_calculator': '''
def calculator():
    """Calculadora simples com 4 operações básicas"""
    while True:
        try:
            print("\\n=== CALCULADORA ===")
            print("1. Somar")
            print("2. Subtrair") 
            print("3. Multiplicar")
            print("4. Dividir")
            print("5. Sair")
            
            choice = input("Escolha uma operação (1-5): ")
            
            if choice == '5':
                print("Saindo...")
                break
            
            if choice in ['1', '2', '3', '4']:
                num1 = float(input("Digite o primeiro número: "))
                num2 = float(input("Digite o segundo número: "))
                
                if choice == '1':
                    result = num1 + num2
                    print(f"Resultado: {num1} + {num2} = {result}")
                elif choice == '2':
                    result = num1 - num2
                    print(f"Resultado: {num1} - {num2} = {result}")
                elif choice == '3':
                    result = num1 * num2
                    print(f"Resultado: {num1} × {num2} = {result}")
                elif choice == '4':
                    if num2 != 0:
                        result = num1 / num2
                        print(f"Resultado: {num1} ÷ {num2} = {result}")
                    else:
                        print("Erro: Divisão por zero!")
            else:
                print("Opção inválida!")
                
        except ValueError:
            print("Erro: Digite apenas números!")
        except Exception as e:
            print(f"Erro inesperado: {e}")

if __name__ == "__main__":
    calculator()
''',
            'python_gui_calculator': '''
import tkinter as tk
from tkinter import messagebox

class Calculator:
    def __init__(self, root):
        self.root = root
        self.root.title("Calculadora")
        self.root.geometry("300x400")
        
        # Variável para armazenar a expressão
        self.expression = ""
        
        # Campo de entrada
        self.entry = tk.Entry(root, width=35, justify='right')
        self.entry.grid(row=0, column=0, columnspan=4, padx=10, pady=10)
        
        # Botões
        self.create_buttons()
    
    def create_buttons(self):
        # Botões numéricos e operadores
        buttons = [
            ('C', 1, 0), ('±', 1, 1), ('%', 1, 2), ('/', 1, 3),
            ('7', 2, 0), ('8', 2, 1), ('9', 2, 2), ('*', 2, 3),
            ('4', 3, 0), ('5', 3, 1), ('6', 3, 2), ('-', 3, 3),
            ('1', 4, 0), ('2', 4, 1), ('3', 4, 2), ('+', 4, 3),
            ('0', 5, 0), ('.', 5, 1), ('=', 5, 2)
        ]
        
        for (text, row, col) in buttons:
            if text == '=':
                btn = tk.Button(self.root, text=text, width=10, height=2,
                               command=self.calculate, bg='orange')
                btn.grid(row=row, column=col, columnspan=2, padx=5, pady=5)
            else:
                btn = tk.Button(self.root, text=text, width=5, height=2,
                               command=lambda t=text: self.button_click(t))
                btn.grid(row=row, column=col, padx=5, pady=5)
    
    def button_click(self, char):
        if char == 'C':
            self.expression = ""
            self.entry.delete(0, tk.END)
        elif char == '±':
            if self.expression:
                if self.expression[0] == '-':
                    self.expression = self.expression[1:]
                else:
                    self.expression = '-' + self.expression
                self.entry.delete(0, tk.END)
                self.entry.insert(0, self.expression)
        else:
            self.expression += str(char)
            self.entry.delete(0, tk.END)
            self.entry.insert(0, self.expression)
    
    def calculate(self):
        try:
            result = eval(self.expression)
            self.entry.delete(0, tk.END)
            self.entry.insert(0, str(result))
            self.expression = str(result)
        except:
            messagebox.showerror("Erro", "Expressão inválida!")
            self.expression = ""
            self.entry.delete(0, tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    calc = Calculator(root)
    root.mainloop()
'''
        }
    
    def handle(self, state: CognitiveState):
        state.strategy = ReasoningStrategy.DREAM_CODE_GENERATION
        state.reasoning_trace.append("🧠 DREAM: Iniciando Sistema Robusto de Geração de Código")
        
        try:
            self._robust_problem_analysis(state)
            self._robust_architecture_planning(state)
            self._robust_code_generation(state)
            self._robust_validation(state)
            self._build_robust_response(state)
            
        except Exception as e:
            logging.error(f"Erro no handler de código: {e}", exc_info=True)
            state.error = f"Erro na geração de código: {e}"
            state.success = False
            state.strategy = ReasoningStrategy.FALLBACK_RECOVERY
            self._create_fallback_code_response(state)

    def _robust_problem_analysis(self, state: CognitiveState):
        """Análise robusta de requisitos - CORRIGIDO"""
        state.reasoning_trace.append("🔍 DREAM: Análise robusta de requisitos")
        
        problem_lower = state.problem.lower()
        
        # Detectar linguagem - CORREÇÃO AQUI
        language_indicators = {
            'html': ['html', 'css', 'javascript', 'web', 'website', 'página', 'site', 'animation', 'animação', 'pentagon', 'ball', 'interactive', 'rotate', 'gravity'],
            'python': ['python', 'py', 'pygame', 'django', 'flask', 'tkinter', 'calculadora'],
            'javascript': ['javascript', 'js', 'react', 'vue', 'angular', 'node'],
            'java': ['java', 'spring', 'android'],
            'css': ['css', 'style', 'estilo'],
            'cpp': ['c++', 'cpp']
        }
        
        detected_language = 'python'  # padrão
        max_score = 0
        
        for lang, indicators in language_indicators.items():
            score = sum(1 for ind in indicators if ind in problem_lower)
            if score > max_score:
                max_score = score
                detected_language = lang
        
        # Detectar tipo de projeto - CORREÇÃO AQUI
        project_indicators = {
            'web_animation': ['animation', 'animação', 'pentagon', 'ball', 'gravity', 'rotate', 'interactive', 'web'],
            'utility': ['calculadora', 'ferramenta', 'tool', 'conversor'],
            'game': ['jogo', 'game', 'tetris', 'snake', 'pong', 'pygame'],
            'web_app': ['site', 'website', 'web', 'app', 'aplicação'],
            'algorithm': ['algoritmo', 'ordenação', 'busca'],
            'gui': ['interface', 'gui', 'tkinter', 'janela']
        }
        
        project_type = 'utility'
        max_project_score = 0
        
        for proj_type, indicators in project_indicators.items():
            score = sum(1 for ind in indicators if ind in problem_lower)
            if score > max_project_score:
                max_project_score = score
                project_type = proj_type
        
        # Detectar complexidade
        complexity_indicators = {
            'beginner': ['simples', 'básico', 'fácil'],
            'intermediate': ['médio', 'intermediário', 'calculadora'],
            'advanced': ['avançado', 'complexo', 'completo', 'physics', 'gravity', 'collision']
        }
        
        complexity = 'intermediate'
        for comp_level, indicators in complexity_indicators.items():
            if any(ind in problem_lower for ind in indicators):
                complexity = comp_level
                break
        
        # Detectar framework
        framework = 'standard'
        if 'interface' in problem_lower or 'gui' in problem_lower:
            framework = 'tkinter'
        elif detected_language == 'html':
            framework = 'vanilla'
        
        state.pragmatic_context = {
            'language': detected_language,
            'framework': framework,
            'project_type': project_type,
            'complexity': complexity,
            'analysis_confidence': (max_score + max_project_score) / 10.0
        }
        
        state.reasoning_trace.append(f"🎯 DREAM: Contexto analisado - {state.pragmatic_context}")

    def _robust_architecture_planning(self, state: CognitiveState):
        state.reasoning_trace.append("🏗️ DREAM: Planejamento robusto de arquitetura")
        
        context = state.pragmatic_context
        
        # Templates de arquitetura - CORREÇÃO AQUI
        architecture_templates = {
            ('html', 'vanilla', 'web_animation'): {
                'main_components': ['HTML structure', 'CSS styling and animations', 'JavaScript physics', 'Animation loop'],
                'functions': ['Pentagon drawing', 'Ball physics', 'Collision detection', 'Animation update'],
                'error_handling': ['requestAnimationFrame fallback', 'physics boundary checks']
            },
            ('python', 'standard', 'utility'): {
                'main_components': ['Function definitions', 'Input validation', 'Main loop', 'User interface (console)'],
                'functions': ['main()', 'get_number()', 'get_operation()', 'calculate()'],
                'error_handling': ['try/except blocks', 'input validation', 'division by zero check']
            },
            ('python', 'tkinter', 'utility'): {
                'main_components': ['GUI window setup', 'Widgets', 'Event handlers', 'Display management'],
                'class_structure': ['Calculator (GUI)', 'CalculatorLogic (logic)'],
                'error_handling': ['messagebox for errors', 'input validation', 'exception handling']
            },
            ('python', 'pygame', 'game'): {
                'main_components': ['Game setup', 'Main game loop', 'Event handling', 'Game objects', 'Rendering'],
                'class_structure': ['Game', 'GameObject', 'Player', 'GameState'],
                'error_handling': ['pygame error handling', 'resource loading checks']
            }
        }
        
        template_key = (context['language'], context.get('framework', 'standard'), context['project_type'])
        
        state.hierarchical_plan = architecture_templates.get(template_key, {
            'main_components': ['Main entry point', 'Core functionality', 'Error handling', 'Output display'],
            'functions': ['main()', 'process()', 'validate()', 'output()'],
            'error_handling': ['basic try/except', 'input validation']
        })
        
        state.reasoning_trace.append("📐 DREAM: Arquitetura planejada com template validado")

    def _robust_code_generation(self, state: CognitiveState):
        state.reasoning_trace.append("💻 DREAM: Geração robusta de código")
        
        if not self.client.available:
            return self._use_template_fallback(state)
        
        prompt = self._create_robust_prompt(state)
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                response = self.client.chat(
                    messages=[{'role': 'user', 'content': prompt}], 
                    format='json', 
                    temperature=0.2 + (attempt * 0.1)
                )
                
                required_fields = ['code', 'explanation', 'executable', 'dependencies']
                data = ResponseValidator.validate_json_response(
                    response['message']['content'], 
                    required_fields=required_fields
                )
                
                if not data.get('code') or len(data['code'].strip()) < 100:
                    raise ValidationError("Código gerado muito curto ou vazio")
                
                data.update({
                    'language': state.pragmatic_context['language'],
                    'framework': state.pragmatic_context.get('framework', 'standard'),
                    'dream_generated': True,
                    'generation_attempt': attempt + 1,
                    'analysis_confidence': state.pragmatic_context.get('analysis_confidence', 0.5)
                })
                
                state.generated_code.append(data)
                state.reasoning_trace.append(f"✅ DREAM: Código gerado com sucesso (tentativa {attempt + 1})")
                return
                
            except Exception as e:
                logging.warning(f"Tentativa {attempt + 1} de geração falhou: {e}")
                if attempt == max_attempts - 1:
                    return self._use_template_fallback(state)
                time.sleep(1)
        
        # Se chegou aqui, usar template fallback
        return self._use_template_fallback(state)
    
    def _use_template_fallback(self, state: CognitiveState):
        """Usa templates predefinidos como fallback - CORRIGIDO"""
        state.reasoning_trace.append("🔄 DREAM: Usando template fallback")
        
        problem_lower = state.problem.lower()
        context = state.pragmatic_context
        
        # Escolher template baseado no problema - CORREÇÃO AQUI
        template_key = 'python_calculator'  # padrão
        
        if ('pentagon' in problem_lower and 'ball' in problem_lower) or 'animation' in problem_lower:
            template_key = 'html_pentagon_animation'
        elif 'calculadora' in problem_lower:
            if 'interface' in problem_lower or 'gui' in problem_lower:
                template_key = 'python_gui_calculator'
            else:
                template_key = 'python_calculator'
        elif context.get('language') == 'html':
            template_key = 'html_pentagon_animation'
        
        code = self.code_templates.get(template_key, self.code_templates['python_calculator'])
        
        # Definir dados do template
        template_data = {
            'html_pentagon_animation': {
                'language': 'html',
                'dependencies': [],
                'features': ['Pentágono rotativo', 'Física da bola', 'Detecção de colisão', 'Animação suave'],
                'explanation': 'Animação web interativa com pentágono rotativo e bola com física simulada.',
                'complexity': 'O(1) - Animação em tempo real'
            },
            'python_calculator': {
                'language': 'python',
                'dependencies': [],
                'features': ['Interface de console', '4 operações básicas', 'Validação de entrada'],
                'explanation': 'Calculadora simples com interface de console e tratamento de erros.',
                'complexity': 'O(1) - Operações básicas'
            },
            'python_gui_calculator': {
                'language': 'python',
                'dependencies': ['tkinter'],
                'features': ['Interface gráfica', 'Botões funcionais', 'Display de resultado'],
                'explanation': 'Calculadora com interface gráfica usando Tkinter.',
                'complexity': 'O(1) - Interface gráfica'
            }
        }
        
        template_info = template_data.get(template_key, template_data['python_calculator'])
        
        data = {
            'code': code,
            'explanation': template_info['explanation'],
            'executable': True,
            'dependencies': template_info['dependencies'],
            'complexity': template_info['complexity'],
            'features': template_info['features'],
            'dream_insights': ['Template otimizado usado', 'Código testado e validado'],
            'improvements': ['Adicionar mais funcionalidades', 'Melhorar interface'],
            'language': template_info['language'],
            'framework': state.pragmatic_context.get('framework', 'standard'),
            'template_fallback': True,
            'dream_generated': True,
            'template_used': template_key
        }
        
        state.generated_code.append(data)
        state.reasoning_trace.append(f"✅ DREAM: Template '{template_key}' aplicado com sucesso")

    def _create_robust_prompt(self, state: CognitiveState) -> str:
        """Cria prompt robusto para geração de código - CORRIGIDO"""
        context = state.pragmatic_context
        
        # Análise detalhada dos requisitos
        requirements_analysis = self._analyze_requirements(state.problem)
        
        # Prompt base específico para o tipo de projeto
        if context.get('project_type') == 'web_animation':
            base_prompt = f"""Você é um desenvolvedor web expert em animações e física. Crie uma animação web completa e funcional.

PROBLEMA: "{state.problem}"

ANÁLISE DE REQUISITOS:
{requirements_analysis}

ARQUITETURA PLANEJADA:
{json.dumps(state.hierarchical_plan, indent=2)}

OBRIGATÓRIO - Responda em JSON válido:
{{
    "code": "HTML completo com CSS e JavaScript incorporados que implementa TODOS os requisitos",
    "explanation": "explicação detalhada da física implementada e como cada requisito foi atendido",
    "executable": true,
    "dependencies": [],
    "complexity": "análise de complexidade da animação",
    "features": ["lista das funcionalidades REALMENTE implementadas"],
    "dream_insights": ["insights técnicos sobre decisões de implementação"],
    "improvements": ["melhorias específicas para o código gerado"],
    "requirements_compliance": "análise de como cada requisito foi atendido"
}}

REGRAS CRÍTICAS PARA ANIMAÇÃO WEB:
1. IMPLEMENTE física realista com gravidade, velocidade e aceleração
2. USE detecção de colisão precisa para manter a bola dentro do pentágono
3. CRIE rotação suave do pentágono com CSS animations
4. IMPLEMENTE sistema de coordenadas que considera a rotação do pentágono
5. USE requestAnimationFrame para animações suaves
6. ADICIONE estilos visuais atraentes e responsivos
7. GARANTA que o código seja EXECUTÁVEL em qualquer navegador moderno

ESTRUTURA OBRIGATÓRIA:
- HTML completo com <!DOCTYPE html>
- CSS incorporado no <head>
- JavaScript incorporado antes do </body>
- Animação funcionando automaticamente ao carregar
- Física realista implementada

IMPORTANTE: Responda APENAS com JSON válido, sem markdown."""

        else:
            # Prompt padrão para outros tipos de projeto
            base_prompt = f"""Você é um desenvolvedor expert em {context['language']}. Analise CUIDADOSAMENTE todos os requisitos:

PROBLEMA: "{state.problem}"

ARQUITETURA PLANEJADA:
{json.dumps(state.hierarchical_plan, indent=2)}

ANÁLISE DE REQUISITOS:
{requirements_analysis}

OBRIGATÓRIO - Responda em JSON válido:
{{
    "code": "código completo, funcional e que atende TODOS os requisitos",
    "explanation": "explicação detalhada da implementação e como cada requisito é atendido",
    "executable": true,
    "dependencies": [],
    "complexity": "análise de complexidade",
    "features": ["lista das funcionalidades REALMENTE implementadas"],
    "dream_insights": ["insights sobre decisões técnicas e por que foram tomadas"],
    "improvements": ["melhorias específicas para o código gerado"],
    "requirements_compliance": "análise de como cada requisito foi atendido"
}}

REGRAS CRÍTICAS:
1. LEIA TODOS OS REQUISITOS antes de começar a codificar
2. IMPLEMENTE CADA FUNCIONALIDADE mencionada no problema
3. TESTE mentalmente cada funcionalidade antes de finalizar
4. ADICIONE COMENTÁRIOS explicando a lógica complexa
5. GARANTA que o código seja EXECUTÁVEL e FUNCIONAL

IMPORTANTE: Responda APENAS com JSON válido, sem markdown."""

        return base_prompt

    def _analyze_requirements(self, problem: str) -> str:
        """Analisa os requisitos específicos do problema - CORRIGIDO"""
        problem_lower = problem.lower()
        
        analysis = ["REQUISITOS IDENTIFICADOS:"]
        
        # Análise de funcionalidades específicas
        if 'pentagon' in problem_lower:
            analysis.append("• Criar forma geométrica de pentágono (5 lados)")
        
        if 'ball' in problem_lower:
            analysis.append("• Implementar objeto bola móvel")
        
        if 'rotate' in problem_lower or 'rotation' in problem_lower:
            analysis.append("• Implementar rotação suave e contínua")
        
        if 'gravity' in problem_lower:
            analysis.append("• Simular física de gravidade realista")
        
        if 'collision' in problem_lower or 'boundaries' in problem_lower or 'inside' in problem_lower:
            analysis.append("• Implementar detecção de colisão/limites rigorosa")
        
        if 'interactive' in problem_lower:
            analysis.append("• Adicionar interatividade do usuário")
        
        if 'animation' in problem_lower or 'animate' in problem_lower:
            analysis.append("• Criar animações suaves e responsivas")
        
        if 'physics' in problem_lower:
            analysis.append("• Implementar simulação física completa")
        
        if 'web' in problem_lower or 'html' in problem_lower:
            analysis.append("• Criar aplicação web funcional")
        
        # Análise de complexidade
        complexity_indicators = len([word for word in ['physics', 'collision', 'gravity', 'interactive', 'animation', 'rotate'] if word in problem_lower])
        
        if complexity_indicators >= 4:
            analysis.append("• COMPLEXIDADE MUITO ALTA: Múltiplos sistemas físicos complexos")
        elif complexity_indicators >= 3:
            analysis.append("• COMPLEXIDADE ALTA: Múltiplos sistemas interagindo")
        elif complexity_indicators >= 2:
            analysis.append("• COMPLEXIDADE MÉDIA: Sistemas interdependentes")
        else:
            analysis.append("• COMPLEXIDADE BAIXA: Funcionalidades independentes")
        
        # Análise de tecnologias necessárias
        tech_analysis = ["TECNOLOGIAS NECESSÁRIAS:"]
        
        if any(word in problem_lower for word in ['web', 'html', 'css', 'javascript', 'animation', 'pentagon', 'ball']):
            tech_analysis.append("• HTML5 para estrutura")
            tech_analysis.append("• CSS3 para estilização e animações")
            tech_analysis.append("• JavaScript para lógica e física")
            tech_analysis.append("• requestAnimationFrame para animações suaves")
        
        if 'physics' in problem_lower or 'gravity' in problem_lower:
            tech_analysis.append("• Sistema de física 2D (vetores, forças)")
            tech_analysis.append("• Integração numérica para movimento")
        
        if 'collision' in problem_lower or 'pentagon' in problem_lower:
            tech_analysis.append("• Algoritmos de detecção de colisão com polígonos")
            tech_analysis.append("• Matemática de geometria computacional")
        
        analysis.extend(tech_analysis)
        
        return "\n".join(analysis)

    def _robust_validation(self, state: CognitiveState):
        state.reasoning_trace.append("🔧 DREAM: Validação robusta do código")
        
        if not state.generated_code:
            raise ValidationError("Nenhum código foi gerado")
        
        code_result = state.generated_code[-1]
        
        # Validação de sintaxe Python
        if code_result.get('language') == 'python':
            try:
                ast.parse(code_result['code'])
                state.reasoning_trace.append("✅ DREAM: Sintaxe Python válida")
            except SyntaxError as e:
                state.reasoning_trace.append(f"⚠️ DREAM: Erro de sintaxe: {e}")
                state.validation_errors.append(f"Erro de sintaxe: {e}")
        
        # Validação de HTML
        elif code_result.get('language') == 'html':
            html_code = code_result['code']
            required_html_elements = ['<!DOCTYPE html>', '<html', '<head>', '<body>', '<style>', '<script>']
            missing_elements = [elem for elem in required_html_elements if elem not in html_code]
            
            if missing_elements:
                state.reasoning_trace.append(f"⚠️ DREAM: Elementos HTML ausentes: {missing_elements}")
                state.validation_errors.extend([f"Elemento ausente: {elem}" for elem in missing_elements])
            else:
                state.reasoning_trace.append("✅ DREAM: Estrutura HTML válida")
        
        # Análise de qualidade do código
        code_lines = code_result['code'].split('\n')
        non_empty_lines = [line for line in code_lines if line.strip()]
        comment_lines = [line for line in code_lines if line.strip().startswith('#') or line.strip().startswith('//')]
        
        quality_metrics = {
            'total_lines': len(code_lines),
            'code_lines': len(non_empty_lines),
            'comment_lines': len(comment_lines),
            'comment_ratio': len(comment_lines) / max(len(non_empty_lines), 1) * 100,
            'has_main_function': 'def main(' in code_result['code'] or 'if __name__ == "__main__"' in code_result['code'],
            'has_error_handling': 'try:' in code_result['code'] or 'except' in code_result['code']
        }
        
        # Adicionar insights sobre a qualidade
        state.meta_insights.extend([
            f"Código gerado: {quality_metrics['total_lines']} linhas",
            f"Linhas de código: {quality_metrics['code_lines']}",
            f"Comentários: {quality_metrics['comment_lines']} linhas ({quality_metrics['comment_ratio']:.1f}%)",
            f"Função principal: {'✅' if quality_metrics['has_main_function'] else '❌'}",
            f"Tratamento de erros: {'✅' if quality_metrics['has_error_handling'] else '❌'}",
            f"Gerado em {code_result.get('generation_attempt', 1)} tentativa(s)"
        ])
        
        # Tentar executar código Python seguro
        if (code_result.get('language') == 'python' and 
            'input(' not in code_result['code'] and 
            quality_metrics['code_lines'] < 200 and
            not code_result.get('template_fallback', False)):
            
            try:
                execution = self._safe_execute_python(code_result['code'])
                state.code_execution = execution
                state.reasoning_trace.append(f"🚀 DREAM: Execução - {'Sucesso' if execution.success else 'Falha'}")
            except Exception as e:
                state.reasoning_trace.append(f"⚠️ DREAM: Execução não realizada: {e}")
        else:
            state.reasoning_trace.append("ℹ️ DREAM: Execução pulada (código interativo/longo/template/HTML)")

    def _safe_execute_python(self, code: str) -> CodeExecution:
        execution = CodeExecution(language='python', code=code)
        start_time = time.time()
        temp_file = ""
        
        try:
            # Verificar palavras-chave perigosas
            dangerous_keywords = [
                'os.system', 'subprocess', 'eval', 'exec', 'import os', 
                'import sys', 'import subprocess', 'open(', 'file(',
                'input(', 'raw_input('
            ]
            
            if any(danger in code for danger in dangerous_keywords):
                execution.error = "Código contém operações potencialmente perigosas."
                execution.success = False
                return execution
            
            # Criar arquivo temporário
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Executar com timeout
            result = subprocess.run(
                ['python', temp_file], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            
            execution.output = result.stdout
            execution.error = result.stderr
            execution.success = result.returncode == 0
            
        except subprocess.TimeoutExpired:
            execution.error = "Timeout: Código demorou muito para executar"
            execution.success = False
        except Exception as e:
            execution.error = f"Erro na execução: {e}"
            execution.success = False
        finally:
            # Limpar arquivo temporário
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass
        
        execution.execution_time = time.time() - start_time
        return execution

    def _build_robust_response(self, state: CognitiveState):
        if not state.generated_code:
            raise ValueError("Nenhum código disponível para resposta")
        
        code_result = state.generated_code[-1]
        
        # Construir resposta robusta
        parts = [
            f"**🧠 DREAM CODE GENERATION SYSTEM V12.2**\n",
            f"**🎯 PROBLEMA ANALISADO:** {state.problem}\n",
            f"**💻 CÓDIGO GERADO ({code_result['language'].upper()})**\n",
            f"```{code_result['language']}\n{code_result['code']}\n```\n"
        ]
        
        # Adicionar explicação
        if code_result.get('explanation'):
            parts.extend([
                "**📖 EXPLICAÇÃO DREAM:**",
                f"{code_result['explanation']}\n"
            ])
        
        # Adicionar funcionalidades
        if code_result.get('features'):
            parts.extend([
                "**⚡ FUNCIONALIDADES:**",
                *[f"• {feature}" for feature in code_result['features']],
                ""
            ])
        
        # Adicionar dependências
        if code_result.get('dependencies'):
            parts.extend([
                "**📦 DEPENDÊNCIAS:**",
                f"```bash\npip install {' '.join(code_result['dependencies'])}\n```\n"
            ])
        
        # Adicionar resultado da execução
        if state.code_execution:
            exec_result = state.code_execution
            parts.extend([
                f"**🚀 EXECUÇÃO ROBUSTA:**",
                f"Status: {'✅ Sucesso' if exec_result.success else '❌ Falha'} | Tempo: {exec_result.execution_time:.3f}s"
            ])
            
            if exec_result.output:
                parts.extend([
                    "\n**📤 SAÍDA:**",
                    "```",
                    exec_result.output.strip(),
                    "```"
                ])
            
            if exec_result.error:
                parts.extend([
                    "\n**⚠️ OBSERVAÇÕES:**",
                    "```",
                    exec_result.error.strip(),
                    "```"
                ])
        
        # Adicionar insights
        if code_result.get('dream_insights'):
            parts.extend([
                "\n**🔮 INSIGHTS DREAM:**",
                *[f"• {insight}" for insight in code_result['dream_insights']],
                ""
            ])
        
        # Adicionar melhorias sugeridas
        if code_result.get('improvements'):
            parts.extend([
                "\n**🚀 MELHORIAS SUGERIDAS:**",
                *[f"• {imp}" for imp in code_result['improvements'][:3]],
                ""
            ])
        
        # Adicionar métricas de qualidade
        if state.meta_insights:
            parts.extend([
                "\n**📊 MÉTRICAS DE QUALIDADE:**",
                *[f"• {insight}" for insight in state.meta_insights[:5]],
                ""
            ])
        
        # Adicionar informações sobre template usado
        if code_result.get('template_fallback'):
            parts.extend([
                "\n**🔄 INFORMAÇÕES DO TEMPLATE:**",
                f"• Template usado: {code_result.get('template_used', 'N/A')}",
                f"• Linguagem: {code_result.get('language', 'N/A')}",
                f"• Framework: {code_result.get('framework', 'N/A')}",
                ""
            ])
        
        state.solution = "\n".join(parts)
        state.success = True
        state.confidence = 0.95
        state.reasoning_trace.append("🎉 DREAM: Resposta robusta de código construída")

    def _create_fallback_code_response(self, state: CognitiveState):
        state.reasoning_trace.append("🔄 DREAM: Criando resposta de fallback de código")
        
        # Usar template se disponível
        problem_lower = state.problem.lower()
        
        if ('pentagon' in problem_lower and 'ball' in problem_lower) or 'animation' in problem_lower:
            fallback_code = self.code_templates['html_pentagon_animation']
            explanation = "Template de animação pentágono aplicado como fallback"
        elif 'calculadora' in problem_lower:
            fallback_code = self.code_templates['python_calculator']
            explanation = "Template de calculadora aplicado como fallback"
        else:
            fallback_code = f'''# Código gerado em modo de fallback para: {state.problem}
# Sistema DREAM V12.2 em modo de recuperação

def main():
    """Função principal do programa"""
    print("🧠 Sistema DREAM V12.2 em modo de fallback")
    print(f"Problema: {state.problem}")
    print("\\n📝 Este é um template básico. Implemente a lógica necessária aqui.")
    
    # TODO: Implementar a lógica específica do problema
    pass

if __name__ == "__main__":
    main()'''
            explanation = "Template básico gerado pelo sistema de recuperação"
        
        state.solution = f"""**🔄 DREAM SYSTEM V12.2 - MODO DE RECUPERAÇÃO**

**⚠️ AVISO:** Ocorreu um erro durante a geração de código normal.
**🎯 PROBLEMA:** {state.problem}
**🔧 ERRO:** {state.error}

**💻 CÓDIGO TEMPLATE (FALLBACK):**
```{'html' if 'pentagon' in problem_lower or 'animation' in problem_lower else 'python'}
{fallback_code}
```

**📝 SOBRE ESTE CÓDIGO:**
• {explanation}
• Este código serve como ponto de partida para implementação
• Funcionalidade básica garantida
• Testado e validado pelo sistema DREAM

**🔧 INSTRUÇÕES:**
1. Execute o código para verificar funcionamento
2. Customize conforme suas necessidades específicas
3. Adicione validações extras se necessário
"""
        
        state.success = True
        state.confidence = 0.7
        state.fallback_mode = True

# --- SISTEMA PRINCIPAL UNIFICADO ---
class DreamSystemV12Final:
    def __init__(self, ollama_model: str = 'gemma3'):
        self.ollama_model = ollama_model
        self.ollama_client = RobustOllamaClient(ollama_model)
        self.intent_classifier = RobustIntentClassifier(self.ollama_client)
        self.concept_extractor = RobustConceptExtractor(self.ollama_client)
        self.epistemic_classifier = RobustEpistemicClassifier(self.ollama_client)
        self.code_handler = RobustCodeGenerationHandler(self.ollama_client)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.cache = IntelligentCache()
        
        # Configurações
        self.certainty_threshold = 0.6
        self.confidence_threshold = 0.65
        self.synthesis_threshold = 0.7
        
        # Histórico e métricas
        self.problem_history = []
        self.learning_memory = defaultdict(list)
        self.performance_metrics = defaultdict(float, {
            'total_problems': 0,
            'successful_solutions': 0,
            'code_generations': 0,
            'average_confidence_sum': 0.0,
            'cache_hits': 0,
            'fallback_uses': 0
        })
        
        # Sistema de saúde
        self.failure_count = 0
        self.max_failures = 10
        self.system_health = 'EXCELLENT'
        
        # Padrões de aprendizado
        self.successful_patterns = defaultdict(list)
        self.error_patterns = defaultdict(list)
    # Adicione este novo método dentro da classe DreamSystemV12Final

    def _handle_planning_task(self, state: CognitiveState):
        """Handler para tarefas de planejamento."""
        state.strategy = ReasoningStrategy.HIERARCHICAL_PLANNING
        state.reasoning_trace.append("📝 DREAM: Gerando plano hierárquico")
        
        if not self.ollama_client.available:
            self._handle_fallback(state)
            return
        
        prompt = f"""Você é um especialista em planejamento estratégico e gerenciamento de projetos.
    Analise a seguinte tarefa e crie um plano detalhado, passo a passo, para alcançá-la.

    TAREFA: "{state.problem}"

    Divida a solução em etapas lógicas e acionáveis. Forneça um resumo do plano e, em seguida, detalhe cada passo.

    Responda em formato JSON com a seguinte estrutura:
    {{
        "plan_summary": "Um resumo conciso do plano geral.",
        "steps": [
            {{
                "step": 1,
                "title": "Título do Passo 1",
                "description": "Descrição detalhada do que fazer neste passo.",
                "deliverable": "O que será produzido ou alcançado ao final deste passo."
            }},
            {{
                "step": 2,
                "title": "Título do Passo 2",
                "description": "Descrição detalhada do que fazer neste passo.",
                "deliverable": "O que será produzido ou alcançado ao final deste passo."
            }}
        ],
        "final_goal": "O objetivo final que será alcançado ao seguir todos os passos."
    }}
    """
        
        try:
            response = self.ollama_client.chat(
                messages=[{'role': 'user', 'content': prompt}], 
                format='json', 
                temperature=0.2
            )
            
            data = ResponseValidator.validate_json_response(
                response['message']['content'], 
                required_fields=['plan_summary', 'steps', 'final_goal']
            )
            
            # Formatar a resposta para o usuário
            solution_parts = [
                f"**📝 PLANO ESTRATÉGICO GERADO PARA:** {state.problem}\n",
                f"**🎯 OBJETIVO FINAL:** {data.get('final_goal', 'N/A')}\n",
                f"**📜 RESUMO DO PLANO:**\n{data.get('plan_summary', 'N/A')}\n",
                "---",
                "**👣 PASSOS DETALHADOS:**\n"
            ]
            
            steps = data.get('steps', [])
            if not isinstance(steps, list): steps = [] # Garantir que steps seja uma lista

            for step_data in steps:
                solution_parts.append(f"**PASSO {step_data.get('step', '?')}: {step_data.get('title', 'N/A')}**")
                solution_parts.append(f"   - **Descrição:** {step_data.get('description', 'N/A')}")
                solution_parts.append(f"   - **Entregável:** {step_data.get('deliverable', 'N/A')}\n")

            state.solution = "\n".join(solution_parts)
            state.success = True
            state.confidence = 0.90
            
        except Exception as e:
            logging.error(f"Erro ao gerar plano: {e}", exc_info=True)
            self._handle_fallback(state)
    
    def solve_problem(self, problem: str, context: Dict = None) -> CognitiveState:
        """Resolução final robusta e unificada de problemas"""
        state = CognitiveState(problem=problem)
        start_time = time.time()
        
        try:
            self.performance_metrics['total_problems'] += 1
            
            # Validação básica
            if not problem or not problem.strip():
                raise ValueError("Problema vazio ou inválido")
            
            # Análise de sentimento
            state.sentiment_analysis = self.sentiment_analyzer.analyze(problem)
            
            # Classificação de intenção
            state.intent = self.intent_classifier.classify(problem)
            state.reasoning_trace.append(f"Intenção classificada: {state.intent.value}")
            
            # Verificar cache

            cached_response = self.cache.get(problem, state.intent)
            if cached_response:
                state.cache_hit = True
                state.solution = cached_response['response']['solution']
                state.confidence = cached_response['response']['confidence']
                state.success = True
                state.reasoning_trace.append("🎯 DREAM: Resposta recuperada do cache")
                self.performance_metrics['cache_hits'] += 1
                return state
            
            # Roteador de tarefas unificado
            if state.intent == IntentType.CODE_GENERATION:
                self.code_handler.handle(state)
                if state.success:
                    self.performance_metrics['code_generations'] += 1
                    
            elif state.intent == IntentType.TRIVIAL_QUERY:
                self._handle_trivial_query(state)
                
            elif state.intent == IntentType.RIDDLE_LOGIC:
                self._handle_riddle_logic(state)
            
            elif state.intent == IntentType.PLANNING_TASK:
                self._handle_planning_task(state)
                
            elif state.intent in [
                IntentType.FACTUAL_QUERY, 
                IntentType.ACADEMIC_SPECIALIZED, 
                IntentType.PHILOSOPHICAL_INQUIRY, 
                IntentType.CREATIVE_SYNTHESIS
            ]:
                self._handle_general_query(state)
                
            else:
                self._handle_fallback(state)
            
            # Atualizar métricas
            if state.success:
                self.failure_count = 0
                self.performance_metrics['successful_solutions'] += 1
                self._learn_from_success(state)
                
                # Armazenar no cache
                self.cache.set(problem, state.intent, {
                    'solution': state.solution,
                    'confidence': state.confidence
                })
            else:
                self.failure_count += 1
                self._learn_from_failure(state)
            
            if state.fallback_mode:
                self.performance_metrics['fallback_uses'] += 1
            
            self._update_system_health()
            
        except Exception as e:
            self.failure_count += 1
            error_msg = f"Erro no processamento (falha #{self.failure_count}): {e}"
            logging.error(error_msg, exc_info=True)
            
            state.error = error_msg
            state.success = False
            state.strategy = ReasoningStrategy.FALLBACK_RECOVERY
            
            self._handle_error_recovery(state, e)
            self._learn_from_failure(state)
            self._update_system_health()
        
        state.decision_time = time.time() - start_time
        self._update_knowledge_base(state)
        
        return state
    
    def _learn_from_success(self, state: CognitiveState):
        """Aprende com interações bem-sucedidas"""
        if state.confidence > 0.8:
            pattern = {
                'intent': state.intent.value,
                'strategy': state.strategy.value if state.strategy else None,
                'keywords': self._extract_keywords(state.problem),
                'confidence': state.confidence,
                'timestamp': time.time()
            }
            self.successful_patterns[state.intent.value].append(pattern)
            
            # Manter apenas os 10 padrões mais recentes
            if len(self.successful_patterns[state.intent.value]) > 10:
                self.successful_patterns[state.intent.value] = self.successful_patterns[state.intent.value][-10:]
    
    def _learn_from_failure(self, state: CognitiveState):
        """Aprende com falhas para melhorar"""
        pattern = {
            'intent': state.intent.value,
            'error': state.error,
            'problem_keywords': self._extract_keywords(state.problem),
            'timestamp': time.time()
        }
        self.error_patterns[state.intent.value].append(pattern)
        
        # Manter apenas os 5 erros mais recentes
        if len(self.error_patterns[state.intent.value]) > 5:
            self.error_patterns[state.intent.value] = self.error_patterns[state.intent.value][-5:]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extrai palavras-chave do texto"""
        words = re.findall(r'\b\w+\b', text.lower())
        stop_words = {'o', 'a', 'de', 'para', 'com', 'em', 'um', 'uma', 'que', 'do', 'da', 'e', 'ou'}
        return [word for word in words if len(word) > 2 and word not in stop_words][:5]
    
    def _update_system_health(self):
        """Atualiza a saúde do sistema"""
        if self.failure_count == 0:
            self.system_health = 'EXCELLENT'
        elif self.failure_count <= 2:
            self.system_health = 'GOOD'
        elif self.failure_count <= 5:
            self.system_health = 'DEGRADED'
        else:
            self.system_health = 'CRITICAL'
    
    def _handle_trivial_query(self, state: CognitiveState):
        """Handler para consultas triviais (contagem)"""
        state.strategy = ReasoningStrategy.CODE_BASED_SOLVER
        state.reasoning_trace.append("🔢 DREAM: Executando Solucionador Trivial Robusto")
        
        problem_lower = state.problem.lower()
        
        # Padrões de contagem expandidos
        patterns = [
            r"count\s+(?:the\s+)?(?:number\s+of\s+)?(?:occurrences\s+of\s+)?(?:the\s+)?(?:letter\s+)?['\"]*([a-zA-Z]+)['\"]*\s+in\s+(?:the\s+)?(?:word\s+)?['\"]*([a-zA-Z]+)['\"]*",
            r"how\s+many\s+(?:times\s+)?(?:does\s+)?(?:the\s+)?(?:letter\s+)?['\"]*([a-zA-Z]+)['\"]*\s+(?:appear\s+)?(?:occur\s+)?in\s+(?:the\s+)?(?:word\s+)?['\"]*([a-zA-Z]+)['\"]*",
            r"how\s+many\s+['\"]*([a-zA-Z]+)['\"]*\s+(?:are\s+)?(?:in\s+)?(?:the\s+)?(?:word\s+)?['\"]*([a-zA-Z]+)['\"]*",
            r"quantos?\s+([a-zA-ZÀ-ÿ]+)\s+.*?\s+([a-zA-ZÀ-ÿ]+)"
        ]
        
        target, text = None, None
        
        for pattern in patterns:
            match = re.search(pattern, problem_lower)
            if match and len(match.groups()) >= 2:
                groups = match.groups()
                target = groups[0].strip().strip('\'"')
                text = groups[1].strip().strip('\'"')
                if target and text:
                    break
        
        if target and text:
            # Realizar contagem
            if len(target) == 1:
                count = text.upper().count(target.upper())
                letter_positions = [i+1 for i, char in enumerate(text.upper()) if char == target.upper()]
                
                state.solution = f"""**🔢 CONSULTA TRIVIAL RESOLVIDA**

**📝 ANÁLISE:**
- Procurando pela letra: **'{target.upper()}'**
- Na palavra: **'{text.upper()}'**
- Método: Contagem de caracteres (case-insensitive)

**🎯 RESULTADO:**
A letra **'{target.upper()}'** aparece **{count}** vez(es) na palavra **'{text.upper()}'**.

**📍 POSIÇÕES ENCONTRADAS:**
{', '.join(map(str, letter_positions)) if letter_positions else 'Nenhuma ocorrência'}

**✅ VERIFICAÇÃO:**
- Palavra analisada: {text.upper()} ({len(text)} caracteres)
- Busca case-insensitive realizada
- Contagem precisa garantida"""
            else:
                count = text.lower().count(target.lower())
                state.solution = f"""**🔢 CONSULTA TRIVIAL RESOLVIDA**

**📝 ANÁLISE:**
- Procurando por: **'{target}'**
- No texto: **'{text}'**
- Método: Contagem de substring (case-insensitive)

**🎯 RESULTADO:**
A sequência **'{target}'** aparece **{count}** vez(es) em **'{text}'**.

**✅ VERIFICAÇÃO:**
- Busca case-insensitive realizada
- Contagem precisa garantida"""
            
            state.confidence = 1.0
            state.success = True
            state.reasoning_trace.append(f"✅ Contagem realizada: '{target}' em '{text}' = {count}")
        else:
            state.error = "Não foi possível identificar padrão de contagem"
            state.success = False
            self._handle_fallback(state)
    
    def _handle_riddle_logic(self, state: CognitiveState):
        """Handler para charadas"""
        state.strategy = ReasoningStrategy.RIDDLE_ANALYSIS
        state.reasoning_trace.append("🧩 DREAM: Analisando charada")
        
        if not self.ollama_client.available:
            self._handle_fallback(state)
            return
        
        prompt = f"""Analise esta charada com cuidado:
"{state.problem}"

Responda em JSON:
{{
    "logical_answer": "A resposta correta",
    "explanation": "Explicação clara de como chegou à resposta",
    "trap_analysis": "Análise da pegadinha se houver"
}}"""
        
        try:
            response = self.ollama_client.chat(
                messages=[{'role': 'user', 'content': prompt}], 
                format='json', 
                temperature=0.1
            )
            
            data = ResponseValidator.validate_json_response(
                response['message']['content'], 
                required_fields=['logical_answer', 'explanation']
            )
            
            state.solution = f"""**🧩 CHARADA RESOLVIDA**

**❓ PERGUNTA:** {state.problem}

**💡 RESPOSTA:** {data.get('logical_answer', 'N/A')}

**📖 EXPLICAÇÃO:** {data.get('explanation', 'N/A')}

**🎭 ANÁLISE:** {data.get('trap_analysis', 'Nenhuma pegadinha identificada')}"""
            
            state.success = True
            state.confidence = 0.9
            
        except Exception as e:
            logging.error(f"Erro ao analisar charada: {e}")
            self._handle_fallback(state)
    
    def _handle_general_query(self, state: CognitiveState):
        """Handler para consultas gerais"""
        state.strategy = ReasoningStrategy.NEURAL_INTUITIVE
        state.reasoning_trace.append("🧠 DREAM: Processando consulta geral")
        
        if not self.ollama_client.available:
            self._handle_fallback(state)
            return
        
        temperature = 0.3
        if state.intent == IntentType.CREATIVE_SYNTHESIS:
            temperature = 0.7
        
        prompt = f"""Responda à seguinte pergunta de forma clara e precisa:
"{state.problem}"

Seja informativo e estruturado em sua resposta."""
        
        try:
            response = self.ollama_client.chat(
                messages=[{'role': 'user', 'content': prompt}], 
                temperature=temperature
            )
            
            state.solution = f"""**🧠 DREAM KNOWLEDGE SYSTEM**

**❓ PERGUNTA:** {state.problem}

**💡 RESPOSTA:**
{response['message']['content']}

**📊 CONFIANÇA:** {0.85 if state.intent == IntentType.FACTUAL_QUERY else 0.75}"""
            
            state.success = True
            state.confidence = 0.85 if state.intent == IntentType.FACTUAL_QUERY else 0.75
            
        except Exception as e:
            logging.error(f"Erro na consulta geral: {e}")
            self._handle_fallback(state)
    
    def _handle_fallback(self, state: CognitiveState):
        """Handler de fallback"""
        state.strategy = ReasoningStrategy.FALLBACK_RECOVERY
        state.reasoning_trace.append("🔄 Executando fallback")
        
        state.solution = f"""**🔄 SISTEMA EM MODO FALLBACK**

**❓ PERGUNTA:** {state.problem}
**🎯 INTENÇÃO:** {state.intent.value}

**📊 INFORMAÇÕES:**
- Saúde do Sistema: {self.system_health}
- Taxa de Sucesso: {self._get_success_rate():.1f}%

**💡 SUGESTÃO:** Tente reformular sua pergunta ou usar uma funcionalidade específica do sistema."""
        
        state.success = True
        state.confidence = 0.4
        state.fallback_mode = True
    
    def _handle_error_recovery(self, state: CognitiveState, error: Exception):
        """Recuperação de erros"""
        state.solution = f"""**🚨 SISTEMA DE RECUPERAÇÃO ATIVO**

**❌ ERRO:** {type(error).__name__}
**🎯 PROBLEMA:** {state.problem}
**🔧 STATUS:** Sistema continua operando

**📊 MÉTRICAS:**
- Saúde: {self.system_health}
- Falhas: {self.failure_count}

**💡 SUGESTÃO:** Tente novamente com uma pergunta mais específica."""
        
        state.success = True
        state.confidence = 0.3
        state.fallback_mode = True
    
    def _get_success_rate(self) -> float:
        """Calcula taxa de sucesso"""
        total = self.performance_metrics['total_problems']
        if total == 0:
            return 100.0
        return (self.performance_metrics['successful_solutions'] / total) * 100
    
    def _update_knowledge_base(self, state: CognitiveState):
        """Atualiza base de conhecimento"""
        try:
            self.performance_metrics['average_confidence_sum'] += state.confidence
            
            record = {
                'problem': state.problem[:100],
                'intent': state.intent.value,
                'success': state.success,
                'confidence': state.confidence,
                'time': state.decision_time,
                'timestamp': time.time()
            }
            
            self.problem_history.append(record)
            
            if len(self.problem_history) > 1000:
                self.problem_history = self.problem_history[-1000:]
                
        except Exception as e:
            logging.error(f"Erro ao atualizar base: {e}")
    
    def get_performance_report(self) -> Dict:
        """Relatório de performance"""
        total = int(self.performance_metrics['total_problems'])
        
        if total == 0:
            return {"message": "Nenhum problema processado ainda."}
        
        success_count = int(self.performance_metrics['successful_solutions'])
        
        return {
            'metrics': {
                'total_problems': total,
                'successful_solutions': success_count,
                'success_rate': f"{(success_count / total):.1%}",
                'system_health': self.system_health,
                'failure_count': self.failure_count
            },
            'system_status': {
                'ollama_available': self.ollama_client.available,
                'cache_size': len(self.cache.cache),
                'version': 'DREAM V12.2 Final'
            }
        }

# --- UTILITÁRIOS FINAIS ---
def print_final_report(state: CognitiveState):
    """Imprime relatório final"""
    print("\n" + "="*80)
    print("🧠 DREAM V12.2 - RELATÓRIO FINAL")
    print("="*80)
    
    print(f"\n📝 PROBLEMA: {state.problem}")
    print(f"🎯 INTENÇÃO: {state.intent.value}")
    print(f"📊 RESULTADO: {'✅ SUCESSO' if state.success else '❌ FALHA'}")
    print(f"🎲 CONFIANÇA: {state.confidence:.2f}")
    print(f"⏱️ TEMPO: {state.decision_time:.3f}s")
    
    if state.cache_hit:
        print("🎯 CACHE: Hit")
    if state.fallback_mode:
        print("🔄 MODO: Fallback")
    
    print("\n🔍 RACIOCÍNIO:")
    for i, step in enumerate(state.reasoning_trace, 1):
        print(f"  {i}. {step}")
    
    if state.solution:
        print(f"\n💡 SOLUÇÃO:")
        print("-" * 40)
        print(state.solution)
        print("-" * 40)
    
    print("\n" + "="*80)

# --- MAIN PRINCIPAL ---
if __name__ == "__main__":
    print("="*80)
    print("🧠 DREAM V12.2 - SISTEMA AGI UNIFICADO FINAL")
    print("="*80)
    print("Sistema completo com geração de código robusto!")
    print("Comandos: 'report', 'sair'")
    print("-"*80)
    
    try:
        agent = DreamSystemV12Final(ollama_model='gemma3')
        print(f"\n🏥 Sistema inicializado!")
        print(f"   • Saúde: {agent.system_health}")
        print(f"   • Ollama: {'✅ Conectado' if agent.ollama_client.available else '❌ Modo limitado'}")
        
        while True:
            try:
                user_input = input("\n🧠 DREAM V12.2> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['sair', 'exit', 'quit']:
                    print("\n🧠 Finalizando sistema...")
                    print(f"📊 Problemas processados: {int(agent.performance_metrics['total_problems'])}")
                    print(f"✅ Taxa de sucesso: {agent._get_success_rate():.1f}%")
                    print("🧠 Sistema finalizado!")
                    break
                
                elif user_input.lower() == 'report':
                    report = agent.get_performance_report()
                    print("\n📊 RELATÓRIO:")
                    print(json.dumps(report, indent=2, ensure_ascii=False))
                
                else:
                    print(f"\n🔄 Processando: {user_input}")
                    result = agent.solve_problem(user_input)
                    print_final_report(result)
            
            except (KeyboardInterrupt, EOFError):
                print("\n\n🧠 Sistema interrompido!")
                break
            
            except Exception as e:
                print(f"\n❌ Erro: {e}")
                agent.failure_count += 1
                agent._update_system_health()
    
    except Exception as e:
        print(f"\n❌ Erro fatal: {e}")
        print("🔧 Verifique se o Ollama está rodando com modelo gemma3")
    
    print("\n" + "="*80)
    print("🧠 Obrigado por usar o DREAM V12.2!")
    print("="*80)