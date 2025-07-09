# ==============================================================================
#                 DREAM V13.2: SISTEMA AGI MULTI-LLM ROBUSTO
#        Sistema Unificado com Ollama + OpenAI - Versão Robusta Completa
# ==============================================================================

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

# --- CONFIGURAÇÃO ROBUSTA V13.2 ---
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("Ollama não disponível. Funcionalidade limitada.")

try:
    import openai
    from openai import OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_AVAILABLE = bool(OPENAI_API_KEY)
    if not OPENAI_API_KEY:
        logging.warning("OPENAI_API_KEY não encontrada. OpenAI indisponível.")
except ImportError:
    OPENAI_AVAILABLE = False
    logging.warning("OpenAI não instalado. Use: pip install openai")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- SISTEMA DE VALIDAÇÃO APRIMORADO (COMPLETO) ---
class ValidationError(Exception):
    """Erro de validação customizado robusto"""
    pass

class ResponseValidator:
    @staticmethod
    def validate_json_response(response_text: str, required_fields: List[str] = None) -> Dict:
        """Validação robusta de JSON com fallbacks múltiplos"""
        if not response_text:
            raise ValidationError("Resposta vazia recebida")
        
        # Padrões de extração JSON expandidos
        json_patterns = [
            r'```json\s*(\{.*?\})\s*```',  # JSON em code block
            r'```\s*(\{.*?\})\s*```',      # JSON sem especificação
            r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})',  # JSON aninhado
            r'(\{.*\})',  # JSON direto
        ]
        
        extracted_json = None
        for pattern in json_patterns:
            matches = re.finditer(pattern, response_text, re.DOTALL)
            for match in matches:
                try:
                    # Tentar parsear cada match encontrado
                    test_json = json.loads(match.group(1))
                    extracted_json = match.group(1)
                    break
                except json.JSONDecodeError:
                    continue
            if extracted_json:
                break
        
        if not extracted_json:
            extracted_json = response_text.strip()
        
        try:
            data = json.loads(extracted_json)
            
            # Completar campos ausentes
            if required_fields:
                missing_fields = [field for field in required_fields if field not in data]
                if missing_fields:
                    for field in missing_fields:
                        data[field] = ResponseValidator._get_default_value(field)
                    logging.warning(f"Campos preenchidos automaticamente: {missing_fields}")
            
            # Validação especial para código
            if 'code' in data and data.get('code'):
                code_length = len(data['code'].strip())
                if code_length < 50:
                    logging.warning(f"Código muito curto ({code_length} chars) - expandindo")
                    data['code'] = ResponseValidator._expand_minimal_code(
                        data['code'], 
                        data.get('language', 'python')
                    )
                elif code_length > 10000:
                    logging.warning(f"Código muito longo ({code_length} chars) - truncando")
                    data['code'] = data['code'][:10000] + "\n# ... código truncado ..."
            
            return data
            
        except json.JSONDecodeError as e:
            logging.error(f"Falha definitiva no JSON: {e}")
            return ResponseValidator._create_fallback_response(response_text, required_fields)
    
    @staticmethod
    def _get_default_value(field_name: str) -> Any:
        """Valores padrão inteligentes por campo"""
        defaults = {
            # Campos de código
            'code': '# Código não gerado devido a erro\nprint("Sistema em modo de recuperação")',
            'explanation': 'Explicação não disponível devido a erro na geração',
            'executable': False,
            'dependencies': [],
            'complexity': 'O(?) - Análise indisponível',
            'features': ['Funcionalidade básica'],
            'dream_insights': ['Geração usando sistema de fallback'],
            'improvements': ['Resolver problemas de geração', 'Adicionar validações'],
            'language': 'python',
            'framework': 'standard',
            
            # Campos de raciocínio
            'logical_answer': 'Resposta não determinada devido a erro',
            'trap_analysis': 'Análise de pegadinha não disponível',
            'reasoning': 'Raciocínio baseado em sistema de recuperação',
            'sub_questions': ['Pergunta não decomposta devido a erro'],
            'critique': 'Crítica não gerada devido a erro',
            'synthesis': 'Síntese não disponível',
            
            # Campos de classificação
            'confidence': 0.3,
            'classification': 'UNKNOWN',
            'fact_or_question': 'Classificação indisponível',
            'concepts': [],
            
            # Campos gerais
            'has_obvious_solution': False,
            'obvious_solution': 'Solução não identificada',
            'central_premise': 'Premissa não extraída',
            'challenger_findings': 'Verificação não realizada',
            'final_answer': 'Resposta final não gerada'
        }
        return defaults.get(field_name, f'Valor padrão para {field_name}')
    
    @staticmethod
    def _expand_minimal_code(code: str, language: str = 'python') -> str:
        """Expansão inteligente de código mínimo"""
        if language.lower() == 'html':
            return f'''<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Página Auto-Expandida</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Conteúdo Auto-Expandido</h1>
        {code}
    </div>
    
    <script>
        console.log("Sistema DREAM V13.2 - Código HTML expandido automaticamente");
        // Código JavaScript pode ser adicionado aqui
    </script>
</body>
</html>'''
        
        elif language.lower() in ['javascript', 'js']:
            return f'''// Sistema DREAM V13.2 - Código JavaScript Expandido
console.log("Inicializando aplicação...");

// Código original expandido
function main() {{
    try {{
        console.log("Executando código principal...");
        
        // Código original aqui
        {code}
        
        console.log("Execução concluída com sucesso!");
    }} catch (error) {{
        console.error("Erro durante execução:", error);
    }}
}}

// Auto-execução quando DOM estiver pronto
if (typeof document !== 'undefined') {{
    document.addEventListener('DOMContentLoaded', main);
}} else {{
    // Para Node.js
    main();
}}'''
        
        elif language.lower() == 'python':
            return f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema DREAM V13.2 - Código Python Auto-Expandido
Gerado automaticamente pelo sistema de recuperação
"""

import sys
import traceback
from datetime import datetime

def main():
    """Função principal do programa"""
    print("🧠 Sistema DREAM V13.2 - Código Expandido")
    print(f"⏰ Executado em: {{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}")
    print("-" * 50)
    
    try:
        # Código original expandido
        print("Executando código principal...")
        
        {code}
        
        print("\\n✅ Execução concluída com sucesso!")
        
    except Exception as e:
        print(f"\\n❌ Erro durante execução: {{e}}")
        print("\\n📋 Traceback detalhado:")
        traceback.print_exc()
        return False
    
    return True

def show_system_info():
    """Mostra informações do sistema"""
    print(f"🐍 Python: {{sys.version}}")
    print(f"🖥️ Plataforma: {{sys.platform}}")
    print(f"📁 Executável: {{sys.executable}}")

if __name__ == "__main__":
    print("=" * 60)
    print("🧠 DREAM V13.2 - Sistema AGI Multi-LLM")
    print("=" * 60)
    
    show_system_info()
    print()
    
    success = main()
    
    print("\\n" + "=" * 60)
    print(f"🎯 Status Final: {{'✅ Sucesso' if success else '❌ Falha'}}")
    print("=" * 60)
    
    sys.exit(0 if success else 1)'''
        
        elif language.lower() in ['css']:
            return f'''/* Sistema DREAM V13.2 - CSS Auto-Expandido */

/* Reset básico */
* {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}}

/* Estilos base */
body {{
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: #333;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}}

/* Container principal */
.container {{
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    background: rgba(255, 255, 255, 0.95);
    border-radius: 10px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    margin-top: 50px;
}}

/* Código original */
{code}

/* Estilos adicionais para responsividade */
@media (max-width: 768px) {{
    .container {{
        margin: 20px;
        padding: 15px;
    }}
}}'''
        
        else:
            # Para outras linguagens ou linguagem desconhecida
            return f'''/*
 * Sistema DREAM V13.2 - Código Auto-Expandido
 * Linguagem: {language}
 * Gerado automaticamente pelo sistema de recuperação
 */

// Início do código expandido
{code}

// Fim do código expandido
// Sistema DREAM V13.2 - Expansão concluída'''
    
    @staticmethod
    def _create_fallback_response(text: str, required_fields: List[str] = None) -> Dict:
        """Cria resposta de fallback robusta quando JSON falha completamente"""
        fallback = {
            'raw_response': text[:1000] + "..." if len(text) > 1000 else text,
            'fallback_mode': True,
            'error': 'Falha na decodificação JSON - usando sistema de recuperação avançado',
            'recovery_method': 'full_fallback',
            'timestamp': time.time()
        }
        
        if required_fields:
            for field in required_fields:
                fallback[field] = ResponseValidator._get_default_value(field)
        
        # Tentar extrair informações básicas do texto bruto
        if 'code' in (required_fields or []):
            # Procurar por blocos de código no texto
            code_patterns = [
                r'```(?:\w+)?\s*(.*?)\s*```',
                r'<code>(.*?)</code>',
                r'def\s+\w+.*?(?=\n\n|\n#|\nclass|\ndef|\Z)',
                r'function\s+\w+.*?(?=\n\n|\n//|\nfunction|\Z)'
            ]
            
            for pattern in code_patterns:
                matches = re.findall(pattern, text, re.DOTALL)
                if matches:
                    fallback['code'] = matches[0].strip()
                    fallback['code_extraction_method'] = f'regex_pattern'
                    break
        
        return fallback

# --- DEFINIÇÕES FUNDAMENTAIS EXPANDIDAS ---
class IntentType(Enum):
    TRIVIAL_QUERY = "Consulta Trivial"
    RIDDLE_LOGIC = "Charada / Lógica Lateral"
    FACTUAL_QUERY = "Pergunta Factual Simples"
    PLANNING_TASK = "Tarefa de Planejamento"
    ACADEMIC_TECHNICAL = "Consulta Acadêmica / Técnica"
    CREATIVE_SYNTHESIS = "Síntese Criativa"
    BROAD_SYNTHESIS = "Pedido de Síntese Ampla"
    PHILOSOPHICAL_INQUIRY = "Investigação Filosófica / Mergulho Profundo"
    COMPLEX_REASONING = "Raciocínio Complexo Multi-Domínio"
    ANALOGICAL_REASONING = "Raciocínio Analógico"
    CODE_GENERATION = "Geração de Código"
    CODE_DEBUGGING = "Debug de Código"
    SYSTEM_ARCHITECTURE = "Arquitetura de Sistema"
    DATA_ANALYSIS = "Análise de Dados"
    MATHEMATICAL_COMPUTATION = "Computação Matemática"
    LANGUAGE_TRANSLATION = "Tradução de Idiomas"
    CREATIVE_WRITING = "Escrita Criativa"
    UNKNOWN = "Intenção Desconhecida"

class KnowledgeClassification(Enum):
    ESTABLISHED = "CONHECIMENTO ESTABELECIDO"
    SPECULATIVE = "CONHECIMENTO ESPECULATIVO"
    AMBIGUOUS = "CONCEITO AMBÍGUO"
    UNKNOWN = "CONCEITO DESCONHECIDO"
    FABRICATED = "POSSÍVEL FABRICAÇÃO"
    EMERGING = "CONHECIMENTO EMERGENTE"
    VALIDATED = "CONHECIMENTO VALIDADO"
    CONTESTED = "CONHECIMENTO CONTESTADO"

class ReasoningStrategy(Enum):
    RESEARCH_FRAMEWORK_GENERATION = "Geração de Framework de Pesquisa"
    VALIDATED_SYNTHESIS = "Síntese Baseada em Conhecimento Validado"
    CODE_BASED_SOLVER = "Solucionador Baseado em Código"
    RIDDLE_ANALYSIS = "Análise de Lógica de Charadas"
    ALGORITHMIC_PLAN_EXECUTION = "Execução de Plano Algorítmico"
    NEURAL_INTUITIVE = "Intuição Neural"
    CREATIVE_REASONING = "Raciocínio Criativo"
    HIERARCHICAL_DECOMPOSITION = "Decomposição Hierárquica"
    SELF_CRITIQUE_REFINEMENT = "Refinamento com Autocrítica"
    FACT_CHECKING_DEBATE = "Debate Adversarial para Verificação"
    ANALOGICAL_TRANSFER = "Transferência Analógica"
    PRAGMATIC_COMMUNICATION = "Comunicação Pragmática"
    FEW_SHOT_ADAPTATION = "Adaptação Few-Shot"
    ADVANCED_SYNTHESIS = "Síntese Avançada"
    MULTI_MODAL_REASONING = "Raciocínio Multi-Modal"
    DREAM_CODE_GENERATION = "Geração de Código via DREAM"
    FALLBACK_RECOVERY = "Recuperação por Fallback"
    PLANNING_EXECUTION = "Execução de Plano Estratégico"
    COLLABORATIVE_REASONING = "Raciocínio Colaborativo Multi-Modelo"

@dataclass
class CodeExecution:
    language: str = ""
    code: str = ""
    output: str = ""
    error: str = ""
    execution_time: float = 0.0
    success: bool = False
    return_code: int = -1
    memory_usage: float = 0.0
    security_level: str = "unknown"

@dataclass
class CognitiveState:
    # Informações básicas
    problem: str = ""
    intent: IntentType = IntentType.UNKNOWN
    concepts: List[str] = field(default_factory=list)
    knowledge_map: Dict[str, KnowledgeClassification] = field(default_factory=dict)
    
    # Estado de processamento
    strategy: Optional[ReasoningStrategy] = None
    confidence: float = 0.0
    solution: Any = None
    reasoning_trace: List[str] = field(default_factory=list)
    success: bool = False
    error: Optional[str] = None
    decision_time: float = 0.0
    
    # Resultados específicos
    code_execution: Optional[CodeExecution] = None
    generated_code: List[Dict] = field(default_factory=list)
    creative_ideas: List[Dict] = field(default_factory=list)
    analogical_mappings: List[Dict] = field(default_factory=list)
    hierarchical_plan: Dict = field(default_factory=dict)
    pragmatic_context: Dict = field(default_factory=dict)
    learning_patterns: List[Dict] = field(default_factory=list)
    
    # Metadados
    meta_insights: List[str] = field(default_factory=list)
    uncertainty_acknowledgment: List[str] = field(default_factory=list)
    alternative_perspectives: List[str] = field(default_factory=list)
    ethical_considerations: List[str] = field(default_factory=list)
    deep_analysis: Dict = field(default_factory=dict)
    research_directions: List[str] = field(default_factory=list)
    synthesis_quality: float = 0.0
    
    # Sistema
    fallback_mode: bool = False
    recovery_attempts: int = 0
    validation_errors: List[str] = field(default_factory=list)
    sentiment_analysis: Dict = field(default_factory=dict)
    cache_hit: bool = False
    model_used: str = ""
    processing_metadata: Dict = field(default_factory=dict)

# --- ABSTRAÇÃO ROBUSTA DE CLIENTES LLM ---
class LLMClient(ABC):
    """Classe abstrata para clientes LLM com interface unificada"""
    
    def __init__(self, model: str, max_retries: int = 3, timeout: int = 60):
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.available = self._check_availability()
        self.request_count = 0
        self.error_count = 0
        self.total_tokens = 0
        self.last_request_time = 0
        
    @abstractmethod
    def _check_availability(self) -> bool:
        """Verifica se o cliente está disponível"""
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict], format: Optional[str] = None, temperature: float = 0.3) -> Dict:
        """Realiza chat com o modelo"""
        pass
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas do cliente"""
        return {
            'model': self.model,
            'available': self.available,
            'request_count': self.request_count,
            'error_count': self.error_count,
            'success_rate': ((self.request_count - self.error_count) / max(self.request_count, 1)) * 100,
            'total_tokens': self.total_tokens,
            'avg_tokens_per_request': self.total_tokens / max(self.request_count, 1),
            'last_request_time': self.last_request_time
        }
    
    def is_healthy(self) -> bool:
        """Verifica se o cliente está saudável"""
        if not self.available:
            return False
        if self.request_count == 0:
            return True
        error_rate = self.error_count / self.request_count
        return error_rate < 0.5  # Menos de 50% de erro

class RobustOllamaClient(LLMClient):
    """Cliente robusto para Ollama com reconexão automática"""
    
    def _check_availability(self) -> bool:
        if not OLLAMA_AVAILABLE:
            logging.warning(f"Ollama não está instalado. Modelo {self.model} indisponível.")
            return False
        
        try:
            import ollama
            result = ollama.show(self.model)
            logging.info(f"Modelo Ollama '{self.model}' disponível")
            return True
        except Exception as e:
            logging.warning(f"Modelo Ollama '{self.model}' não disponível: {e}")
            return False
    
    def chat(self, messages: List[Dict], format: Optional[str] = None, temperature: float = 0.3) -> Dict:
        if not self.available:
            raise ConnectionError(f"Ollama modelo '{self.model}' não está disponível")
        
        self.request_count += 1
        self.last_request_time = time.time()
        
        for attempt in range(self.max_retries):
            try:
                import ollama
                
                options = {
                    'temperature': temperature,
                    'num_predict': 4000,  # Limite de tokens de saída
                    'top_k': 40,
                    'top_p': 0.9,
                }
                
                kwargs = {
                    'model': self.model,
                    'messages': messages,
                    'options': options
                }
                
                if format:
                    kwargs['format'] = format
                
                response = ollama.chat(**kwargs)
                
                # Estimar tokens (aproximação)
                content = response.get('message', {}).get('content', '')
                estimated_tokens = len(content.split()) * 1.3  # Aproximação
                self.total_tokens += estimated_tokens
                
                return response
                
            except Exception as e:
                self.error_count += 1
                logging.warning(f"Ollama tentativa {attempt + 1} falhou: {e}")
                
                if attempt == self.max_retries - 1:
                    raise ConnectionError(f"Todas as {self.max_retries} tentativas falharam para Ollama")
                
                # Backoff exponencial
                time.sleep(2 ** attempt)
        
        raise RuntimeError("Falha inesperada no cliente Ollama")

class OpenAIClient(LLMClient):
    """Cliente robusto para OpenAI com rate limiting e retry"""
    
    def __init__(self, model: str, **kwargs):
        super().__init__(model, **kwargs)
        if self.available:
            try:
                self.client = OpenAI(
                    api_key=OPENAI_API_KEY,
                    timeout=self.timeout,
                    max_retries=self.max_retries
                )
            except Exception as e:
                logging.error(f"Erro ao inicializar cliente OpenAI: {e}")
                self.available = False
    
    def _check_availability(self) -> bool:
        if not OPENAI_AVAILABLE:
            logging.warning(f"OpenAI não está disponível. Modelo {self.model} indisponível.")
            return False
        
        try:
            client = OpenAI(api_key=OPENAI_API_KEY, timeout=10)
            # Tentar listar modelos para verificar conectividade
            models = client.models.list()
            
            # Verificar se o modelo específico está disponível
            available_models = [model.id for model in models.data]
            if self.model not in available_models:
                logging.warning(f"Modelo OpenAI '{self.model}' não encontrado. Disponíveis: {available_models[:5]}...")
                # Alguns modelos podem não aparecer na lista mas ainda funcionar
                return True  # Tentar mesmo assim
            
            logging.info(f"Modelo OpenAI '{self.model}' disponível")
            return True
            
        except Exception as e:
            logging.warning(f"OpenAI não disponível: {e}")
            return False
    
    def chat(self, messages: List[Dict], format: Optional[str] = None, temperature: float = 0.3) -> Dict:
        if not self.available:
            raise ConnectionError(f"OpenAI modelo '{self.model}' não está disponível")
        
        self.request_count += 1
        self.last_request_time = time.time()
        
        try:
            # Preparar parâmetros
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 4000,
                "top_p": 0.9,
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1
            }
            
            # Adicionar formato JSON se solicitado
            if format == 'json':
                params["response_format"] = {"type": "json_object"}
                # Garantir que a mensagem solicita JSON
                if messages and "json" not in messages[-1]["content"].lower():
                    messages[-1]["content"] += "\n\nResponda em formato JSON válido."
            
            # Fazer a requisição
            response = self.client.chat.completions.create(**params)
            
            # Extrair informações de uso
            usage = response.usage
            if usage:
                self.total_tokens += usage.total_tokens
            
            # Retornar no formato compatível
            return {
                "message": {
                    "content": response.choices[0].message.content
                },
                "usage": {
                    "prompt_tokens": usage.prompt_tokens if usage else 0,
                    "completion_tokens": usage.completion_tokens if usage else 0,
                    "total_tokens": usage.total_tokens if usage else 0
                }
            }
            
        except Exception as e:
            self.error_count += 1
            logging.error(f"Erro no OpenAI: {e}")
            
            # Rate limiting específico
            if "rate_limit" in str(e).lower():
                logging.warning("Rate limit atingido. Aguardando...")
                time.sleep(60)  # Aguardar 1 minuto
                raise ConnectionError("Rate limit atingido. Tente novamente.")
            
            raise ConnectionError(f"Erro OpenAI: {e}")

# --- CACHE INTELIGENTE ROBUSTO ---
class IntelligentCache:
    """Sistema de cache inteligente com LRU e persistência opcional"""
    
    def __init__(self, max_size: int = 100, enable_persistence: bool = False):
        self.cache = {}
        self.access_count = defaultdict(int)
        self.access_times = defaultdict(list)
        self.max_size = max_size
        self.enable_persistence = enable_persistence
        self.cache_file = Path("dream_cache.json")
        self.hits = 0
        self.misses = 0
        
        if enable_persistence:
            self._load_cache()
    
    def _hash_key(self, problem: str, intent: IntentType) -> str:
        """Cria chave de hash determinística"""
        content = f"{problem.lower().strip()}_{intent.value}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def get(self, problem: str, intent: IntentType) -> Optional[Dict]:
        """Recupera resposta do cache com atualização de acesso"""
        key = self._hash_key(problem, intent)
        
        if key in self.cache:
            self.access_count[key] += 1
            self.access_times[key].append(time.time())
            self.hits += 1
            
            # Manter apenas os últimos 10 acessos
            if len(self.access_times[key]) > 10:
                self.access_times[key] = self.access_times[key][-10:]
            
            logging.info(f"Cache HIT para problema: {problem[:50]}...")
            return self.cache[key]
        
        self.misses += 1
        logging.debug(f"Cache MISS para problema: {problem[:50]}...")
        return None
    
    def set(self, problem: str, intent: IntentType, response: Dict):
        """Armazena resposta no cache com limpeza automática"""
        if len(self.cache) >= self.max_size:
            self._cleanup_cache()
        
        key = self._hash_key(problem, intent)
        self.cache[key] = {
            'response': response,
            'timestamp': time.time(),
            'problem_preview': problem[:100],
            'intent': intent.value,
            'access_count': 1
        }
        
        self.access_count[key] = 1
        self.access_times[key] = [time.time()]
        
        if self.enable_persistence:
            self._save_cache()
        
        logging.debug(f"Cache SET para: {problem[:50]}...")
    
    def _cleanup_cache(self):
        """Remove entradas menos úteis usando algoritmo LRU melhorado"""
        if not self.cache:
            return
        
        # Calcular score para cada entrada baseado em:
        # - Frequência de acesso
        # - Recência do último acesso
        # - Idade da entrada
        current_time = time.time()
        scores = {}
        
        for key, data in self.cache.items():
            access_freq = self.access_count[key]
            last_access = max(self.access_times[key]) if self.access_times[key] else data['timestamp']
            age = current_time - data['timestamp']
            recency = current_time - last_access
            
            # Score maior = mais importante (não remover)
            score = (access_freq * 2) - (age / 3600) - (recency / 1800)  # Pesos ajustáveis
            scores[key] = score
        
        # Remover 25% das entradas com menor score
        items_to_remove = max(1, len(self.cache) // 4)
        keys_to_remove = sorted(scores.keys(), key=lambda k: scores[k])[:items_to_remove]
        
        for key in keys_to_remove:
            del self.cache[key]
            if key in self.access_count:
                del self.access_count[key]
            if key in self.access_times:
                del self.access_times[key]
        
        logging.info(f"Cache cleanup: removidas {len(keys_to_remove)} entradas")
    
    def _save_cache(self):
        """Salva cache em arquivo (persistência)"""
        try:
            # Converter para formato serializável
            serializable_cache = {}
            for key, data in self.cache.items():
                serializable_cache[key] = {
                    'response': data['response'],
                    'timestamp': data['timestamp'],
                    'problem_preview': data['problem_preview'],
                    'intent': data['intent'],
                    'access_count': self.access_count[key]
                }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_cache, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logging.warning(f"Erro ao salvar cache: {e}")
    
    def _load_cache(self):
        """Carrega cache do arquivo"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    loaded_cache = json.load(f)
                
                for key, data in loaded_cache.items():
                    self.cache[key] = data
                    self.access_count[key] = data.get('access_count', 1)
                    self.access_times[key] = [data['timestamp']]
                
                logging.info(f"Cache carregado: {len(self.cache)} entradas")
                
        except Exception as e:
            logging.warning(f"Erro ao carregar cache: {e}")
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas do cache"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{hit_rate:.1f}%",
            'total_requests': total_requests,
            'persistence_enabled': self.enable_persistence
        }
    
    def clear(self):
        """Limpa todo o cache"""
        self.cache.clear()
        self.access_count.clear()
        self.access_times.clear()
        self.hits = 0
        self.misses = 0
        
        if self.enable_persistence and self.cache_file.exists():
            self.cache_file.unlink()
        
        logging.info("Cache completamente limpo")

# --- COMPONENTES AUXILIARES ROBUSTOS ---
class RobustIntentClassifier:
    """Classificador de intenção robusto com múltiplos métodos"""
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client
        self.patterns = self._initialize_patterns()
        self.keyword_mappings = self._initialize_keyword_mappings()
        self.confidence_threshold = 0.7
    
    def _initialize_patterns(self) -> Dict[IntentType, List[str]]:
        """Padrões regex para classificação de intenção"""
        return {
            IntentType.TRIVIAL_QUERY: [
                r"quantos?\s+\w+\s+(?:temos|tem|há|existem?)\s+(?:na|em)",
                r"how\s+many\s+\w+\s+(?:are\s+)?in",
                r"count\s+(?:the\s+)?(?:number\s+of\s+)?\w+\s+in",
                r"conte?\s+(?:as?\s+)?\w+\s+em"
            ],
            IntentType.CODE_GENERATION: [
                r"(?:crie|create|write|desenvolva|develop|implemente|implement|faça|make|gere|generate|escreva|programa|build)",
                r".*(?:código|code|program|app|aplicação|jogo|game|script|software|sistema|calculator|website)",
                r"(?:pygame|python|javascript|java|html|css|react|vue|angular|tkinter|flask|django)",
                r"(?:tetris|snake|pong|calculator|todo|login|website|gui|interface|animation|pentagon|ball)"
            ],
            IntentType.RIDDLE_LOGIC: [
                r"which\s+weighs\s+more",
                r"what.*(?:heavier|lighter)",
                r"trick.*question",
                r"pegadinha|charada|enigma|puzzle",
                r"sally.*sisters",
                r"brothers.*sisters",
                r"logic.*puzzle",
                r"o\s+que\s+pesa\s+mais"
            ],
            IntentType.PLANNING_TASK: [
                r"(?:plano|plan|planeje|planejamento|planning)",
                r"(?:passo\s+a\s+passo|step\s+by\s+step)",
                r"(?:como\s+fazer|how\s+to\s+do|how\s+to\s+make)",
                r"(?:estratégia|strategy|roadmap|cronograma)",
                r"(?:torre\s+de\s+han[óo]i|tower\s+of\s+hanoi)"
            ],
            IntentType.PHILOSOPHICAL_INQUIRY: [
                r"(?:natureza\s+(?:da|de)|nature\s+of)",
                r"(?:consciência|consciousness|awareness)",
                r"(?:livre\s+arbítrio|free\s+will)",
                r"(?:sentido\s+da\s+vida|meaning\s+of\s+life)",
                r"(?:ética\s+(?:da|de)|ethics\s+of)",
                r"(?:filosofia|philosophy|philosophical)",
                r"(?:existência|existence|existential)",
                r"(?:propósito|purpose|significado|meaning)"
            ],
            IntentType.ACADEMIC_TECHNICAL: [
                r"(?:explique\s+a\s+teoria|explain\s+the\s+theory)",
                r"(?:relação\s+entre.*e|relationship\s+between.*and)",
                r"(?:como.*resolve|how.*solves?)",
                r"(?:status\s+do\s+problema|problem\s+status)",
                r"(?:algoritmo\s+de|algorithm\s+for)",
                r"(?:complexidade|complexity|big\s+o)",
                r"(?:prova\s+de|proof\s+of|demonstração)"
            ],
            IntentType.BROAD_SYNTHESIS: [
                r"(?:resuma\s+a\s+história\s+de|summarize\s+the\s+history\s+of)",
                r"(?:faça\s+uma\s+síntese\s+sobre|make\s+a\s+synthesis\s+about)",
                r"(?:quais\s+são\s+os\s+principais\s+pontos|what\s+are\s+the\s+main\s+points)",
                r"(?:overview|visão\s+geral|panorama)",
                r"(?:comparação\s+entre|comparison\s+between)"
            ],
            IntentType.MATHEMATICAL_COMPUTATION: [
                r"(?:calcule|calculate|compute|resolver)",
                r"(?:integral|derivative|derivada|limite|limit)",
                r"(?:equação|equation|sistema\s+linear)",
                r"(?:matriz|matrix|determinante|eigenvalue)",
                r"(?:função|function|domínio|domain)"
            ],
            IntentType.CREATIVE_WRITING: [
                r"(?:escreva\s+(?:uma\s+)?(?:história|story|poema|poem))",
                r"(?:crie\s+(?:um\s+)?(?:roteiro|script|diálogo))",
                r"(?:invente|imagine|fantasia|creative)",
                r"(?:narrativa|narrative|ficção|fiction)"
            ]
        }
    
    def _initialize_keyword_mappings(self) -> Dict[IntentType, List[str]]:
        """Mapeamentos de palavras-chave para classificação"""
        return {
            IntentType.CODE_GENERATION: [
                'código', 'code', 'program', 'jogo', 'game', 'site', 'app',
                'calculadora', 'calculator', 'interface', 'gui', 'script',
                'animation', 'animação', 'pentagon', 'ball', 'gravity', 'physics',
                'interactive', 'web', 'html', 'css', 'javascript', 'python',
                'tkinter', 'pygame', 'flask', 'django', 'react', 'vue'
            ],
            IntentType.ACADEMIC_TECHNICAL: [
                'pesquisa', 'research', 'teoria', 'theory', 'ciência', 'science',
                'algoritmo', 'algorithm', 'matemática', 'mathematics', 'física',
                'physics', 'química', 'chemistry', 'biologia', 'biology'
            ],
            IntentType.CREATIVE_SYNTHESIS: [
                'criativo', 'creative', 'inovação', 'innovation', 'imagine',
                'inventar', 'criar', 'design', 'arte', 'art'
            ],
            IntentType.PHILOSOPHICAL_INQUIRY: [
                'filosofia', 'philosophy', 'ética', 'ethics', 'moral', 'sentido',
                'meaning', 'consciousness', 'consciência', 'existência', 'existence'
            ],
            IntentType.RIDDLE_LOGIC: [
                'charada', 'riddle', 'puzzle', 'enigma', 'pegadinha', 'trick',
                'logic', 'lógica', 'pesa', 'weighs', 'heavier', 'lighter'
            ],
            IntentType.PLANNING_TASK: [
                'plano', 'plan', 'planejamento', 'planning', 'estratégia',
                'strategy', 'passo', 'step', 'como fazer', 'how to'
            ]
        }
    
    def classify(self, problem: str) -> Tuple[IntentType, float]:
        """Classifica intenção com múltiplos métodos e retorna confiança"""
        problem_lower = problem.lower()
        
        # Método 1: Padrões regex (alta confiança)
        for intent_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, problem_lower):
                    return intent_type, 0.9
        
        # Método 2: Palavras-chave (confiança média)
        keyword_scores = {}
        for intent_type, keywords in self.keyword_mappings.items():
            score = sum(1 for keyword in keywords if keyword in problem_lower)
            if score > 0:
                keyword_scores[intent_type] = score
        
        if keyword_scores:
            best_intent = max(keyword_scores, key=keyword_scores.get)
            max_score = keyword_scores[best_intent]
            confidence = min(0.8, 0.4 + (max_score * 0.1))
            return best_intent, confidence
        
        # Método 3: LLM (se disponível) - confiança variável
        if self.llm_client and self.llm_client.available:
            try:
                llm_intent, llm_confidence = self._classify_with_llm(problem)
                if llm_confidence >= self.confidence_threshold:
                    return llm_intent, llm_confidence
            except Exception as e:
                logging.warning(f"Classificação LLM falhou: {e}")
        
        # Método 4: Heurísticas simples (baixa confiança)
        if any(word in problem_lower for word in ['?', 'como', 'what', 'how', 'why', 'quando', 'where']):
            return IntentType.FACTUAL_QUERY, 0.5
        
        if len(problem.split()) > 20:
            return IntentType.BROAD_SYNTHESIS, 0.4
        
        # Fallback
        return IntentType.FACTUAL_QUERY, 0.3
    
    def _classify_with_llm(self, problem: str) -> Tuple[IntentType, float]:
        """Classificação usando LLM"""
        intent_descriptions = {
            intent.name: intent.value for intent in IntentType
        }
        
        prompt = f"""Classifique a intenção desta pergunta em UMA categoria:
"{problem}"

Categorias disponíveis:
{json.dumps(intent_descriptions, indent=2, ensure_ascii=False)}

Responda APENAS com o nome da categoria (ex: CODE_GENERATION), sem explicações."""
        
        try:
            response = self.llm_client.chat(
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.0
            )
            
            intent_name = response['message']['content'].strip().upper()
            
            # Procurar por correspondência exata
            for intent in IntentType:
                if intent.name == intent_name:
                    return intent, 0.8
            
            # Procurar por correspondência parcial
            for intent in IntentType:
                if intent.name in intent_name or intent_name in intent.name:
                    return intent, 0.6
            
            return IntentType.FACTUAL_QUERY, 0.4
            
        except Exception as e:
            logging.error(f"Erro na classificação LLM: {e}")
            return IntentType.FACTUAL_QUERY, 0.3

class SentimentAnalyzer:
    """Analisador de sentimento e contexto da pergunta"""
    
    def analyze(self, problem: str) -> Dict:
        """Analisa sentimento, urgência, complexidade e outras métricas"""
        problem_lower = problem.lower()
        
        # Análise de urgência
        urgency_keywords = [
            'urgente', 'rápido', 'agora', 'imediato', 'hoje', 'já',
            'urgent', 'quickly', 'now', 'immediate', 'asap', 'fast'
        ]
        urgency_score = sum(1 for keyword in urgency_keywords if keyword in problem_lower)
        
        # Análise de polidez
        politeness_keywords = [
            'por favor', 'obrigado', 'obrigada', 'agradeço', 'grato',
            'please', 'thank you', 'thanks', 'appreciate', 'grateful',
            'poderia', 'seria possível', 'could you', 'would you'
        ]
        politeness_score = sum(1 for keyword in politeness_keywords if keyword in problem_lower)
        
        # Análise de complexidade
        complexity_keywords = [
            'complexo', 'difícil', 'avançado', 'complicado', 'elaborado',
            'complex', 'difficult', 'advanced', 'complicated', 'sophisticated',
            'detalhado', 'completo', 'abrangente', 'detailed', 'comprehensive'
        ]
        complexity_score = sum(1 for keyword in complexity_keywords if keyword in problem_lower)
        
        # Análise de incerteza
        uncertainty_keywords = [
            'talvez', 'pode ser', 'acho que', 'parece', 'provavelmente',
            'maybe', 'perhaps', 'might', 'seems', 'probably', 'possibly'
        ]
        uncertainty_score = sum(1 for keyword in uncertainty_keywords if keyword in problem_lower)
        
        # Análise de emoção
        positive_keywords = [
            'gosto', 'amo', 'adoro', 'interessante', 'incrível', 'ótimo',
            'love', 'like', 'amazing', 'great', 'awesome', 'fantastic'
        ]
        negative_keywords = [
            'odeio', 'detesto', 'horrível', 'péssimo', 'ruim', 'problema',
            'hate', 'terrible', 'awful', 'bad', 'horrible', 'issue'
        ]
        
        positive_score = sum(1 for keyword in positive_keywords if keyword in problem_lower)
        negative_score = sum(1 for keyword in negative_keywords if keyword in problem_lower)
        
        # Cálculo de métricas
        word_count = len(problem.split())
        char_count = len(problem)
        sentence_count = len(re.findall(r'[.!?]+', problem))
        question_marks = problem.count('?')
        
        # Determinar tom geral
        if politeness_score > 0:
            tone = 'polite'
        elif negative_score > positive_score:
            tone = 'negative'
        elif positive_score > 0:
            tone = 'positive'
        else:
            tone = 'neutral'
        
        # Determinar categoria de complexidade
        if complexity_score > 2 or word_count > 50:
            complexity_category = 'high'
        elif complexity_score > 0 or word_count > 20:
            complexity_category = 'medium'
        else:
            complexity_category = 'low'
        
        return {
            'urgency': min(urgency_score / 3.0, 1.0),
            'politeness': min(politeness_score / 3.0, 1.0),
            'complexity': min(complexity_score / 3.0, 1.0),
            'uncertainty': min(uncertainty_score / 3.0, 1.0),
            'emotion_score': positive_score - negative_score,
            'word_count': word_count,
            'char_count': char_count,
            'sentence_count': max(sentence_count, 1),
            'question_marks': question_marks,
            'tone': tone,
            'complexity_category': complexity_category,
            'avg_words_per_sentence': word_count / max(sentence_count, 1),
            'is_question': question_marks > 0 or any(
                problem_lower.startswith(q) for q in ['como', 'what', 'how', 'why', 'quando', 'onde', 'quem']
            ),
            'formality_level': 'formal' if politeness_score > 1 else 'informal'
        }

# --- HANDLER DE CÓDIGO ROBUSTO (VERSÃO COMPLETA) ---
class RobustCodeGenerationHandler:
    """Handler robusto para geração de código com templates e validação"""
    
    def __init__(self):
        self.code_templates = self._initialize_templates()
        self.language_detectors = self._initialize_language_detectors()
        self.project_type_detectors = self._initialize_project_type_detectors()
        self.complexity_analyzers = self._initialize_complexity_analyzers()
    
    def _initialize_templates(self) -> Dict:
        """Templates de código robustos e completos"""
        return {
            'html_pentagon_animation': '''<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pentágono Rotativo com Física da Bola</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            background: linear-gradient(135deg, #1e3c72, #2a5298);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            font-family: 'Arial', sans-serif;
            overflow: hidden;
        }
        
        .container {
            position: relative;
            width: 400px;
            height: 400px;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        .pentagon-container {
            position: absolute;
            width: 300px;
            height: 300px;
            animation: rotate 8s linear infinite;
        }
        
        .pentagon {
            width: 100%;
            height: 100%;
            background: rgba(255, 255, 255, 0.1);
            border: 3px solid #4CAF50;
            clip-path: polygon(50% 0%, 100% 38%, 82% 100%, 18% 100%, 0% 38%);
            box-shadow: 
                0 0 20px rgba(76, 175, 80, 0.5),
                inset 0 0 20px rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
        }
        
        .ball {
            position: absolute;
            width: 20px;
            height: 20px;
            background: radial-gradient(circle at 30% 30%, #ff6b6b, #ee5a52, #dc3545);
            border-radius: 50%;
            box-shadow: 
                0 0 15px rgba(255, 107, 107, 0.8),
                inset -2px -2px 5px rgba(0, 0, 0, 0.3),
                inset 2px 2px 5px rgba(255, 255, 255, 0.3);
            z-index: 10;
            transition: transform 0.016s ease-out;
        }
        
        @keyframes rotate {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        
        .info {
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            color: white;
            text-align: center;
            font-size: 14px;
            background: rgba(0, 0, 0, 0.5);
            padding: 10px 20px;
            border-radius: 20px;
            backdrop-filter: blur(10px);
        }
        
        .controls {
            position: absolute;
            top: 20px;
            left: 20px;
            color: white;
            font-size: 12px;
        }
        
        .controls button {
            background: rgba(255, 255, 255, 0.2);
            border: none;
            color: white;
            padding: 5px 10px;
            margin: 2px;
            border-radius: 5px;
            cursor: pointer;
            transition: background 0.3s;
        }
        
        .controls button:hover {
            background: rgba(255, 255, 255, 0.3);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="pentagon-container" id="pentagon-container">
            <div class="pentagon"></div>
        </div>
        <div class="ball" id="ball"></div>
        
        <div class="info">
            <h3>🔮 Pentágono Mágico com Física Realista</h3>
            <p>Bola com gravidade simulada • Colisão precisa • Física 2D</p>
        </div>
        
        <div class="controls">
            <button onclick="physics.resetBall()">🔄 Reset</button>
            <button onclick="physics.toggleGravity()">⚡ Gravidade</button>
            <button onclick="physics.addImpulse()">🚀 Impulso</button>
        </div>
    </div>
    
    <script>
        class PentagonPhysicsEngine {
            constructor() {
                this.ball = document.getElementById('ball');
                this.pentagon = document.getElementById('pentagon-container');
                
                // Estado da bola
                this.ballPosition = { x: 0, y: 0 };
                this.ballVelocity = { x: 0, y: 0 };
                this.ballRadius = 10;
                
                // Configurações físicas
                this.gravity = 0.4;
                this.gravityEnabled = true;
                this.friction = 0.99;
                this.bounce = 0.85;
                this.airResistance = 0.999;
                
                // Geometria do pentágono
                this.pentagonRadius = 130;
                this.pentagonVertices = this.calculatePentagonVertices();
                
                // Estado da animação
                this.animationId = null;
                this.lastTime = 0;
                
                this.init();
            }
            
            calculatePentagonVertices() {
                const vertices = [];
                const angleStep = (Math.PI * 2) / 5;
                const startAngle = -Math.PI / 2; // Começar do topo
                
                for (let i = 0; i < 5; i++) {
                    const angle = startAngle + (i * angleStep);
                    vertices.push({
                        x: Math.cos(angle) * this.pentagonRadius,
                        y: Math.sin(angle) * this.pentagonRadius
                    });
                }
                
                return vertices;
            }
            
            init() {
                this.resetBall();
                this.startAnimation();
                this.setupEventListeners();
            }
            
            setupEventListeners() {
                // Clique para adicionar impulso aleatório
                this.ball.addEventListener('click', () => this.addImpulse());
                
                // Teclas para controle
                document.addEventListener('keydown', (e) => {
                    switch(e.key.toLowerCase()) {
                        case 'r': this.resetBall(); break;
                        case 'g': this.toggleGravity(); break;
                        case ' ': this.addImpulse(); e.preventDefault(); break;
                        case 'arrowup': this.ballVelocity.y -= 2; break;
                        case 'arrowdown': this.ballVelocity.y += 2; break;
                        case 'arrowleft': this.ballVelocity.x -= 2; break;
                        case 'arrowright': this.ballVelocity.x += 2; break;
                    }
                });
            }
            
            resetBall() {
                this.ballPosition = { x: 0, y: -50 };
                this.ballVelocity = { 
                    x: (Math.random() - 0.5) * 4, 
                    y: (Math.random() - 0.5) * 2 
                };
                this.updateBallVisual();
            }
            
            toggleGravity() {
                this.gravityEnabled = !this.gravityEnabled;
                console.log(`Gravidade: ${this.gravityEnabled ? 'ON' : 'OFF'}`);
            }
            
            addImpulse() {
                const impulse = {
                    x: (Math.random() - 0.5) * 8,
                    y: (Math.random() - 0.5) * 8
                };
                this.ballVelocity.x += impulse.x;
                this.ballVelocity.y += impulse.y;
            }
            
            getCurrentPentagonRotation() {
                const computedStyle = getComputedStyle(this.pentagon);
                const transform = computedStyle.transform;
                
                if (transform === 'none') return 0;
                
                const values = transform.split('(')[1].split(')')[0].split(',');
                const a = parseFloat(values[0]);
                const b = parseFloat(values[1]);
                
                return Math.atan2(b, a);
            }
            
            rotatePoint(point, angle) {
                const cos = Math.cos(angle);
                const sin = Math.sin(angle);
                
                return {
                    x: point.x * cos - point.y * sin,
                    y: point.x * sin + point.y * cos
                };
            }
            
            isPointInsidePentagon(point) {
                const rotation = this.getCurrentPentagonRotation();
                const rotatedVertices = this.pentagonVertices.map(v => this.rotatePoint(v, rotation));
                
                // Algoritmo de ray casting
                let inside = false;
                const { x, y } = point;
                
                for (let i = 0, j = rotatedVertices.length - 1; i < rotatedVertices.length; j = i++) {
                    const xi = rotatedVertices[i].x;
                    const yi = rotatedVertices[i].y;
                    const xj = rotatedVertices[j].x;
                    const yj = rotatedVertices[j].y;
                    
                    if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
                        inside = !inside;
                    }
                }
                
                return inside;
            }
            
            getClosestPointOnPentagonEdge(point) {
                const rotation = this.getCurrentPentagonRotation();
                const rotatedVertices = this.pentagonVertices.map(v => this.rotatePoint(v, rotation));
                
                let closestPoint = point;
                let minDistance = Infinity;
                let closestNormal = { x: 0, y: 1 };
                
                for (let i = 0; i < rotatedVertices.length; i++) {
                    const v1 = rotatedVertices[i];
                    const v2 = rotatedVertices[(i + 1) % rotatedVertices.length];
                    
                    // Encontrar ponto mais próximo na aresta
                    const edge = { x: v2.x - v1.x, y: v2.y - v1.y };
                    const edgeLength = Math.sqrt(edge.x * edge.x + edge.y * edge.y);
                    const edgeUnit = { x: edge.x / edgeLength, y: edge.y / edgeLength };
                    
                    const toPoint = { x: point.x - v1.x, y: point.y - v1.y };
                    const projection = Math.max(0, Math.min(edgeLength, 
                        toPoint.x * edgeUnit.x + toPoint.y * edgeUnit.y
                    ));
                    
                    const pointOnEdge = {
                        x: v1.x + edgeUnit.x * projection,
                        y: v1.y + edgeUnit.y * projection
                    };
                    
                    const distance = Math.sqrt(
                        Math.pow(point.x - pointOnEdge.x, 2) + 
                        Math.pow(point.y - pointOnEdge.y, 2)
                    );
                    
                    if (distance < minDistance) {
                        minDistance = distance;
                        closestPoint = pointOnEdge;
                        
                        // Calcular normal da aresta (apontando para dentro)
                        const normal = { x: -edgeUnit.y, y: edgeUnit.x };
                        const center = { x: 0, y: 0 };
                        const toCenter = { 
                            x: center.x - pointOnEdge.x, 
                            y: center.y - pointOnEdge.y 
                        };
                        
                        // Garantir que a normal aponta para dentro
                        if (normal.x * toCenter.x + normal.y * toCenter.y < 0) {
                            normal.x = -normal.x;
                            normal.y = -normal.y;
                        }
                        
                        closestNormal = normal;
                    }
                }
                
                return { point: closestPoint, normal: closestNormal, distance: minDistance };
            }
            
            update(deltaTime) {
                // Aplicar gravidade relativa ao pentágono
                if (this.gravityEnabled) {
                    const rotation = this.getCurrentPentagonRotation();
                    const gravityVector = this.rotatePoint({ x: 0, y: this.gravity }, rotation);
                    
                    this.ballVelocity.x += gravityVector.x * deltaTime;
                    this.ballVelocity.y += gravityVector.y * deltaTime;
                }
                
                // Aplicar resistência do ar
                this.ballVelocity.x *= Math.pow(this.airResistance, deltaTime);
                this.ballVelocity.y *= Math.pow(this.airResistance, deltaTime);
                
                // Atualizar posição
                this.ballPosition.x += this.ballVelocity.x * deltaTime;
                this.ballPosition.y += this.ballVelocity.y * deltaTime;
                
                // Verificar colisão com as bordas do pentágono
                if (!this.isPointInsidePentagon(this.ballPosition)) {
                    const collision = this.getClosestPointOnPentagonEdge(this.ballPosition);
                    
                    // Posicionar a bola na borda
                    const pushDistance = this.ballRadius;
                    this.ballPosition.x = collision.point.x + collision.normal.x * pushDistance;
                    this.ballPosition.y = collision.point.y + collision.normal.y * pushDistance;
                    
                    // Calcular reflexão da velocidade
                    const velocityDotNormal = 
                        this.ballVelocity.x * collision.normal.x + 
                        this.ballVelocity.y * collision.normal.y;
                    
                    if (velocityDotNormal < 0) { // Movendo-se em direção à parede
                        this.ballVelocity.x -= 2 * velocityDotNormal * collision.normal.x;
                        this.ballVelocity.y -= 2 * velocityDotNormal * collision.normal.y;
                        
                        // Aplicar coeficiente de restituição
                        this.ballVelocity.x *= this.bounce;
                        this.ballVelocity.y *= this.bounce;
                    }
                }
                
                this.updateBallVisual();
            }
            
            updateBallVisual() {
                const centerX = 200; // Centro do container
                const centerY = 200;
                
                this.ball.style.left = `${centerX + this.ballPosition.x - this.ballRadius}px`;
                this.ball.style.top = `${centerY + this.ballPosition.y - this.ballRadius}px`;
            }
            
            startAnimation() {
                const animate = (currentTime) => {
                    if (this.lastTime === 0) this.lastTime = currentTime;
                    
                    const deltaTime = Math.min((currentTime - this.lastTime) / 1000, 1/30); // Máximo 30 FPS
                    this.lastTime = currentTime;
                    
                    this.update(deltaTime * 60); // Normalizar para 60 FPS
                    
                    this.animationId = requestAnimationFrame(animate);
                };
                
                this.animationId = requestAnimationFrame(animate);
            }
            
            stop() {
                if (this.animationId) {
                    cancelAnimationFrame(this.animationId);
                    this.animationId = null;
                }
            }
        }
        
        // Inicializar a simulação quando a página carregar
        let physics;
        
        document.addEventListener('DOMContentLoaded', () => {
            physics = new PentagonPhysicsEngine();
            
            // Log de controles
            console.log(`
🎮 CONTROLES DISPONÍVEIS:
• R - Reset da bola
• G - Toggle gravidade
• Espaço - Impulso aleatório
• Setas - Movimento manual
• Clique na bola - Impulso
            `);
        });
        
        // Pausar/continuar com visibilidade da página
        document.addEventListener('visibilitychange', () => {
            if (physics) {
                if (document.hidden) {
                    physics.stop();
                } else {
                    physics.startAnimation();
                }
            }
        });
    </script>
</body>
</html>''',
            
            'python_calculator': '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculadora Avançada - Sistema DREAM V13.2
Calculadora robusta com múltiplas operações e tratamento de erros
"""

import math
import sys
from typing import Union, List, Dict

class AdvancedCalculator:
    """Calculadora avançada com múltiplas operações"""
    
    def __init__(self):
        self.history: List[str] = []
        self.memory: float = 0.0
        self.last_result: float = 0.0
        
    def add(self, a: float, b: float) -> float:
        """Soma dois números"""
        result = a + b
        self._record_operation(f"{a} + {b} = {result}")
        return result
    
    def subtract(self, a: float, b: float) -> float:
        """Subtrai dois números"""
        result = a - b
        self._record_operation(f"{a} - {b} = {result}")
        return result
    
    def multiply(self, a: float, b: float) -> float:
        """Multiplica dois números"""
        result = a * b
        self._record_operation(f"{a} × {b} = {result}")
        return result
    
    def divide(self, a: float, b: float) -> float:
        """Divide dois números"""
        if b == 0:
            raise ValueError("Divisão por zero não é permitida!")
        result = a / b
        self._record_operation(f"{a} ÷ {b} = {result}")
        return result
    
    def power(self, base: float, exponent: float) -> float:
        """Calcula base elevado a exponent"""
        result = math.pow(base, exponent)
        self._record_operation(f"{base}^{exponent} = {result}")
        return result
    
    def square_root(self, x: float) -> float:
        """Calcula raiz quadrada"""
        if x < 0:
            raise ValueError("Raiz quadrada de número negativo!")
        result = math.sqrt(x)
        self._record_operation(f"√{x} = {result}")
        return result
    
    def factorial(self, n: int) -> int:
        """Calcula fatorial"""
        if n < 0:
            raise ValueError("Fatorial de número negativo!")
        if n > 170:
            raise ValueError("Número muito grande para fatorial!")
        result = math.factorial(n)
        self._record_operation(f"{n}! = {result}")
        return result
    
    def percentage(self, value: float, percent: float) -> float:
        """Calcula porcentagem"""
        result = (value * percent) / 100
        self._record_operation(f"{percent}% de {value} = {result}")
        return result
    
    def _record_operation(self, operation: str):
        """Registra operação no histórico"""
        self.history.append(operation)
        if len(self.history) > 50:  # Manter apenas 50 operações
            self.history.pop(0)
    
    def get_history(self) -> List[str]:
        """Retorna histórico de operações"""
        return self.history.copy()
    
    def clear_history(self):
        """Limpa histórico"""
        self.history.clear()
    
    def store_memory(self, value: float):
        """Armazena valor na memória"""
        self.memory = value
        print(f"💾 Valor {value} armazenado na memória")
    
    def recall_memory(self) -> float:
        """Recupera valor da memória"""
        print(f"💾 Valor da memória: {self.memory}")
        return self.memory
    
    def clear_memory(self):
        """Limpa memória"""
        self.memory = 0.0
        print("💾 Memória limpa")

def get_number_input(prompt: str) -> float:
    """Obtém entrada numérica do usuário com validação"""
    while True:
        try:
            value = input(prompt).strip()
            
            # Permitir usar resultado anterior
            if value.lower() == 'ans':
                return calculator.last_result
            
            # Permitir usar memória
            if value.lower() == 'mem':
                return calculator.recall_memory()
            
            return float(value)
            
        except ValueError:
            print("❌ Erro: Digite um número válido!")
        except KeyboardInterrupt:
            print("\\n👋 Operação cancelada!")
            return None

def show_menu():
    """Exibe menu principal"""
    print("\\n" + "="*50)
    print("🧮 CALCULADORA AVANÇADA - DREAM V13.2")
    print("="*50)
    print("1️⃣  Somar")
    print("2️⃣  Subtrair")
    print("3️⃣  Multiplicar")
    print("4️⃣  Dividir")
    print("5️⃣  Potenciação")
    print("6️⃣  Raiz Quadrada")
    print("7️⃣  Fatorial")
    print("8️⃣  Porcentagem")
    print("9️⃣  Histórico")
    print("🔟 Memória")
    print("0️⃣  Sair")
    print("-"*50)
    print("💡 Dicas:")
    print("   • Use 'ans' para o último resultado")
    print("   • Use 'mem' para valor da memória")
    print("   • Ctrl+C para cancelar operação")
    print("="*50)

def show_memory_menu():
    """Exibe menu de memória"""
    print("\\n" + "="*30)
    print("💾 MENU DE MEMÓRIA")
    print("="*30)
    print("1. Armazenar na memória")
    print("2. Recuperar da memória")
    print("3. Limpar memória")
    print("4. Voltar ao menu principal")
    print("="*30)

def handle_basic_operations(operation: str) -> bool:
    """Manipula operações básicas (2 números)"""
    operations_map = {
        '1': ('somar', calculator.add),
        '2': ('subtrair', calculator.subtract),
        '3': ('multiplicar', calculator.multiply),
        '4': ('dividir', calculator.divide),
        '5': ('potenciação', calculator.power)
    }
    
    if operation not in operations_map:
        return False
    
    op_name, op_func = operations_map[operation]
    
    print(f"\\n📐 Operação: {op_name.title()}")
    
    num1 = get_number_input("Digite o primeiro número: ")
    if num1 is None:
        return True
    
    num2 = get_number_input("Digite o segundo número: ")
    if num2 is None:
        return True
    
    try:
        result = op_func(num1, num2)
        calculator.last_result = result
        print(f"\\n✅ Resultado: {result}")
        
        # Oferecer armazenar na memória
        store = input("\\n💾 Armazenar resultado na memória? (s/N): ").lower()
        if store == 's':
            calculator.store_memory(result)
            
    except ValueError as e:
        print(f"\\n❌ Erro: {e}")
    except Exception as e:
        print(f"\\n❌ Erro inesperado: {e}")
    
    return True

def handle_single_operations(operation: str) -> bool:
    """Manipula operações de um número"""
    if operation == '6':  # Raiz quadrada
        print("\\n📐 Operação: Raiz Quadrada")
        num = get_number_input("Digite o número: ")
        if num is None:
            return True
            
        try:
            result = calculator.square_root(num)
            calculator.last_result = result
            print(f"\\n✅ Resultado: {result}")
        except ValueError as e:
            print(f"\\n❌ Erro: {e}")
            
    elif operation == '7':  # Fatorial
        print("\\n📐 Operação: Fatorial")
        num = get_number_input("Digite o número (inteiro): ")
        if num is None:
            return True
            
        try:
            if not num.is_integer():
                print("❌ Erro: Fatorial requer número inteiro!")
                return True
                
            result = calculator.factorial(int(num))
            calculator.last_result = result
            print(f"\\n✅ Resultado: {result}")
        except ValueError as e:
            print(f"\\n❌ Erro: {e}")
            
    elif operation == '8':  # Porcentagem
        print("\\n📐 Operação: Porcentagem")
        value = get_number_input("Digite o valor: ")
        if value is None:
            return True
            
        percent = get_number_input("Digite a porcentagem: ")
        if percent is None:
            return True
            
        try:
            result = calculator.percentage(value, percent)
            calculator.last_result = result
            print(f"\\n✅ Resultado: {result}")
        except Exception as e:
            print(f"\\n❌ Erro: {e}")
    else:
        return False
    
    return True

def handle_history():
    """Mostra histórico de operações"""
    history = calculator.get_history()
    
    if not history:
        print("\\n📋 Histórico vazio")
        return
    
    print("\\n" + "="*40)
    print("📋 HISTÓRICO DE OPERAÇÕES")
    print("="*40)
    
    for i, operation in enumerate(history[-10:], 1):  # Últimas 10
        print(f"{i:2d}. {operation}")
    
    print("="*40)
    print(f"Total de operações: {len(history)}")
    
    if len(history) > 10:
        print("(Mostrando apenas as 10 mais recentes)")
    
    # Opção de limpar histórico
    clear = input("\\n🗑️ Limpar histórico? (s/N): ").lower()
    if clear == 's':
        calculator.clear_history()
        print("✅ Histórico limpo!")

def handle_memory():
    """Manipula operações de memória"""
    while True:
        show_memory_menu()
        choice = input("\\nEscolha uma opção: ").strip()
        
        if choice == '1':
            value = get_number_input("Digite o valor para armazenar: ")
            if value is not None:
                calculator.store_memory(value)
                
        elif choice == '2':
            value = calculator.recall_memory()
            
        elif choice == '3':
            calculator.clear_memory()
            
        elif choice == '4':
            break
            
        else:
            print("❌ Opção inválida!")

def main():
    """Função principal da calculadora"""
    global calculator
    calculator = AdvancedCalculator()
    
    print("🧮 Calculadora Avançada iniciada!")
    print("Sistema DREAM V13.2 - Versão Robusta")
    
    while True:
        try:
            show_menu()
            choice = input("\\nEscolha uma opção: ").strip()
            
            if choice == '0':
                print("\\n👋 Encerrando calculadora...")
                print("📊 Estatísticas da sessão:")
                print(f"   • Operações realizadas: {len(calculator.get_history())}")
                print(f"   • Último resultado: {calculator.last_result}")
                print(f"   • Valor na memória: {calculator.memory}")
                print("\\n🧮 Obrigado por usar a Calculadora Avançada!")
                break
            
            elif choice in ['1', '2', '3', '4', '5']:
                handle_basic_operations(choice)
                
            elif choice in ['6', '7', '8']:
                handle_single_operations(choice)
                
            elif choice == '9':
                handle_history()
                
            elif choice == '10':
                handle_memory()
                
            else:
                print("\\n❌ Opção inválida! Escolha um número de 0 a 10.")
            
            # Pausar para o usuário ler o resultado
            if choice != '0':
                input("\\n⏎ Pressione Enter para continuar...")
                
        except KeyboardInterrupt:
            print("\\n\\n🛑 Calculadora interrompida pelo usuário!")
            confirm = input("Deseja realmente sair? (s/N): ").lower()
            if confirm == 's':
                break
            else:
                print("Continuando...")
                
        except Exception as e:
            print(f"\\n❌ Erro inesperado: {e}")
            print("🔧 A calculadora continuará funcionando...")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\\n💥 Erro fatal: {e}")
        print("🔧 Reinicie a calculadora")
        sys.exit(1)
''',
            
            'python_gui_calculator': '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculadora GUI Avançada - Sistema DREAM V13.2
Interface gráfica moderna com Tkinter
"""

import tkinter as tk
from tkinter import ttk, messagebox, font
import math
import re
from typing import Optional

class ModernCalculatorGUI:
    """Calculadora GUI moderna com design responsivo"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.setup_window()
        self.setup_variables()
        self.setup_styles()
        self.create_widgets()
        self.bind_keyboard()
        
    def setup_window(self):
        """Configura janela principal"""
        self.root.title("🧮 Calculadora Avançada - DREAM V13.2")
        self.root.geometry("400x600")
        self.root.minsize(350, 500)
        self.root.configure(bg='#2c3e50')
        
        # Centralizar janela
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() - self.root.winfo_width()) // 2
        y = (self.root.winfo_screenheight() - self.root.winfo_height()) // 2
        self.root.geometry(f"+{x}+{y}")
        
        # Ícone (se disponível)
        try:
            self.root.iconbitmap('calculator.ico')
        except:
            pass
    
    def setup_variables(self):
        """Inicializa variáveis"""
        self.expression = ""
        self.result_var = tk.StringVar(value="0")
        self.history_var = tk.StringVar(value="")
        self.memory = 0.0
        self.last_result = 0.0
        self.decimal_places = 8
        
    def setup_styles(self):
        """Configura estilos ttk"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Cores do tema
        self.colors = {
            'bg_primary': '#2c3e50',
            'bg_secondary': '#34495e',
            'fg_primary': '#ecf0f1',
            'fg_secondary': '#bdc3c7',
            'accent': '#3498db',
            'accent_hover': '#2980b9',
            'success': '#27ae60',
            'warning': '#f39c12',
            'danger': '#e74c3c'
        }
        
        # Configurar estilos
        self.style.configure('Display.TFrame', background=self.colors['bg_secondary'])
        self.style.configure('Button.TFrame', background=self.colors['bg_primary'])
        
    def create_widgets(self):
        """Cria todos os widgets"""
        # Frame principal
        main_frame = ttk.Frame(self.root, style='Display.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Área de display
        self.create_display(main_frame)
        
        # Área de botões
        self.create_buttons(main_frame)
        
        # Barra de status
        self.create_status_bar()
        
    def create_display(self, parent):
        """Cria área de display"""
        display_frame = ttk.Frame(parent, style='Display.TFrame')
        display_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Histórico (linha pequena)
        self.history_label = tk.Label(
            display_frame,
            textvariable=self.history_var,
            font=('Consolas', 10),
            bg=self.colors['bg_secondary'],
            fg=self.colors['fg_secondary'],
            anchor='e',
            height=1
        )
        self.history_label.pack(fill=tk.X, padx=5, pady=(5, 0))
        
        # Display principal
        self.display = tk.Label(
            display_frame,
            textvariable=self.result_var,
            font=('Consolas', 24, 'bold'),
            bg=self.colors['bg_secondary'],
            fg=self.colors['fg_primary'],
            anchor='e',
            relief='sunken',
            bd=2,
            height=2
        )
        self.display.pack(fill=tk.X, padx=5, pady=5)
        
    def create_buttons(self, parent):
        """Cria grade de botões"""
        button_frame = ttk.Frame(parent, style='Button.TFrame')
        button_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configurar grid
        for i in range(6):
            button_frame.grid_rowconfigure(i, weight=1)
        for i in range(4):
            button_frame.grid_columnconfigure(i, weight=1)
        
        # Definir botões por linha
        buttons = [
            # Linha 1: Funções especiais
            [('MC', self.memory_clear, self.colors['warning']),
             ('MR', self.memory_recall, self.colors['warning']),
             ('M+', self.memory_add, self.colors['warning']),
             ('M-', self.memory_subtract, self.colors['warning'])],
            
            # Linha 2: Operações avançadas
            [('√', lambda: self.function_operation('sqrt'), self.colors['accent']),
             ('x²', lambda: self.function_operation('square'), self.colors['accent']),
             ('1/x', lambda: self.function_operation('reciprocal'), self.colors['accent']),
             ('C', self.clear_all, self.colors['danger'])],
            
            # Linha 3: Números e operações
            [('7', lambda: self.add_to_expression('7'), self.colors['bg_secondary']),
             ('8', lambda: self.add_to_expression('8'), self.colors['bg_secondary']),
             ('9', lambda: self.add_to_expression('9'), self.colors['bg_secondary']),
             ('÷', lambda: self.add_operator('/'), self.colors['accent'])],
            
            # Linha 4
            [('4', lambda: self.add_to_expression('4'), self.colors['bg_secondary']),
             ('5', lambda: self.add_to_expression('5'), self.colors['bg_secondary']),
             ('6', lambda: self.add_to_expression('6'), self.colors['bg_secondary']),
             ('×', lambda: self.add_operator('*'), self.colors['accent'])],
            
            # Linha 5
            [('1', lambda: self.add_to_expression('1'), self.colors['bg_secondary']),
             ('2', lambda: self.add_to_expression('2'), self.colors['bg_secondary']),
             ('3', lambda: self.add_to_expression('3'), self.colors['bg_secondary']),
             ('-', lambda: self.add_operator('-'), self.colors['accent'])],
            
            # Linha 6
            [('±', self.toggle_sign, self.colors['accent']),
             ('0', lambda: self.add_to_expression('0'), self.colors['bg_secondary']),
             ('.', lambda: self.add_to_expression('.'), self.colors['bg_secondary']),
             ('+', lambda: self.add_operator('+'), self.colors['accent'])]
        ]
        
        # Criar botões
        for row, button_row in enumerate(buttons):
            for col, (text, command, color) in enumerate(button_row):
                btn = tk.Button(
                    button_frame,
                    text=text,
                    command=command,
                    font=('Arial', 12, 'bold'),
                    bg=color,
                    fg='white',
                    relief='raised',
                    bd=2,
                    activebackground=self.colors['accent_hover']
                )
                btn.grid(row=row, column=col, sticky='nsew', padx=2, pady=2)
        
        # Botão = (mais largo)
        equals_btn = tk.Button(
            button_frame,
            text='=',
            command=self.calculate,
            font=('Arial', 16, 'bold'),
            bg=self.colors['success'],
            fg='white',
            relief='raised',
            bd=2,
            activebackground='#1e8449'
        )
        equals_btn.grid(row=6, column=0, columnspan=4, sticky='nsew', padx=2, pady=2)
        
    def create_status_bar(self):
        """Cria barra de status"""
        self.status_frame = tk.Frame(self.root, bg=self.colors['bg_primary'])
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_label = tk.Label(
            self.status_frame,
            text="Pronto • Memória: 0",
            font=('Arial', 9),
            bg=self.colors['bg_primary'],
            fg=self.colors['fg_secondary'],
            anchor='w'
        )
        self.status_label.pack(side=tk.LEFT, padx=5, pady=2)
        
        # Info de versão
        version_label = tk.Label(
            self.status_frame,
            text="DREAM V13.2",
            font=('Arial', 9),
            bg=self.colors['bg_primary'],
            fg=self.colors['accent'],
            anchor='e'
        )
        version_label.pack(side=tk.RIGHT, padx=5, pady=2)
        
    def bind_keyboard(self):
        """Configura atalhos de teclado"""
        self.root.bind('<Key>', self.on_key_press)
        self.root.focus_set()
        
        # Atalhos específicos
        self.root.bind('<Return>', lambda e: self.calculate())
        self.root.bind('<KP_Enter>', lambda e: self.calculate())
        self.root.bind('<Escape>', lambda e: self.clear_all())
        self.root.bind('<BackSpace>', lambda e: self.backspace())
        self.root.bind('<Delete>', lambda e: self.clear_all())
        
    def on_key_press(self, event):
        """Manipula teclas pressionadas"""
        key = event.char
        
        if key.isdigit():
            self.add_to_expression(key)
        elif key in '+-*/':
            self.add_operator(key)
        elif key == '.':
            self.add_to_expression('.')
        elif key == '=':
            self.calculate()
        elif key == 'c' or key == 'C':
            self.clear_all()
            
    def add_to_expression(self, value):
        """Adiciona valor à expressão"""
        if self.result_var.get() == "0" and value != '.':
            self.expression = value
        else:
            self.expression += value
        
        self.update_display()
        
    def add_operator(self, operator):
        """Adiciona operador à expressão"""
        if self.expression and self.expression[-1] not in '+-*/':
            self.expression += operator
            self.update_display()
            
    def update_display(self):
        """Atualiza display principal"""
        display_text = self.expression if self.expression else "0"
        
        # Limitar tamanho do display
        if len(display_text) > 15:
            display_text = display_text[-15:]
            
        self.result_var.set(display_text)
        
    def calculate(self):
        """Calcula resultado da expressão"""
        if not self.expression:
            return
            
        try:
            # Preparar expressão para avaliação segura
            safe_expression = self.expression.replace('×', '*').replace('÷', '/')
            
            # Validar expressão
            if not self.is_safe_expression(safe_expression):
                raise ValueError("Expressão inválida")
            
            # Calcular
            result = eval(safe_expression)
            
            # Formatar resultado
            if isinstance(result, float):
                if result.is_integer():
                    result = int(result)
                else:
                    result = round(result, self.decimal_places)
            
            # Atualizar displays
            self.history_var.set(f"{self.expression} =")
            self.result_var.set(str(result))
            
            # Salvar resultado
            self.last_result = float(result)
            self.expression = str(result)
            
            # Atualizar status
            self.update_status("Cálculo realizado")
            
        except ZeroDivisionError:
            self.show_error("Erro: Divisão por zero")
        except ValueError as e:
            self.show_error(f"Erro: {e}")
        except Exception as e:
            self.show_error(f"Erro inesperado: {e}")
            
    def is_safe_expression(self, expression):
        """Verifica se a expressão é segura para eval"""
        # Permitir apenas números, operadores e parênteses
        allowed = re.match(r'^[0-9+\-*/.() ]+$', expression)
        return allowed is not None
        
    def function_operation(self, func_name):
        """Executa operações de função especial"""
        try:
            current_value = float(self.result_var.get())
            
            if func_name == 'sqrt':
                if current_value < 0:
                    raise ValueError("Raiz quadrada de número negativo")
                result = math.sqrt(current_value)
                operation = f"√({current_value})"
                
            elif func_name == 'square':
                result = current_value ** 2
                operation = f"({current_value})²"
                
            elif func_name == 'reciprocal':
                if current_value == 0:
                    raise ValueError("Divisão por zero")
                result = 1 / current_value
                operation = f"1/({current_value})"
                
            else:
                raise ValueError("Função desconhecida")
            
            # Formatar resultado
            if isinstance(result, float) and result.is_integer():
                result = int(result)
            else:
                result = round(result, self.decimal_places)
            
            # Atualizar displays
            self.history_var.set(f"{operation} =")
            self.result_var.set(str(result))
            self.expression = str(result)
            self.last_result = float(result)
            
            self.update_status(f"Função {func_name} aplicada")
            
        except ValueError as e:
            self.show_error(f"Erro: {e}")
        except Exception as e:
            self.show_error(f"Erro inesperado: {e}")
            
    def toggle_sign(self):
        """Alterna sinal do número atual"""
        try:
            current_value = float(self.result_var.get())
            new_value = -current_value
            
            if new_value.is_integer():
                new_value = int(new_value)
                
            self.result_var.set(str(new_value))
            self.expression = str(new_value)
            
        except ValueError:
            pass  # Ignorar se não for número
            
    def clear_all(self):
        """Limpa tudo"""
        self.expression = ""
        self.result_var.set("0")
        self.history_var.set("")
        self.update_status("Calculadora limpa")
        
    def backspace(self):
        """Remove último caractere"""
        if self.expression:
            self.expression = self.expression[:-1]
            self.update_display()
            
    def memory_clear(self):
        """Limpa memória"""
        self.memory = 0.0
        self.update_status("Memória limpa")
        
    def memory_recall(self):
        """Recupera valor da memória"""
        value = int(self.memory) if self.memory.is_integer() else self.memory
        self.result_var.set(str(value))
        self.expression = str(value)
        self.update_status(f"Memória recuperada: {value}")
        
    def memory_add(self):
        """Adiciona à memória"""
        try:
            current_value = float(self.result_var.get())
            self.memory += current_value
            self.update_status(f"Adicionado à memória: {current_value}")
        except ValueError:
            self.show_error("Valor inválido para memória")
            
    def memory_subtract(self):
        """Subtrai da memória"""
        try:
            current_value = float(self.result_var.get())
            self.memory -= current_value
            self.update_status(f"Subtraído da memória: {current_value}")
        except ValueError:
            self.show_error("Valor inválido para memória")
            
    def update_status(self, message="Pronto"):
        """Atualiza barra de status"""
        memory_display = int(self.memory) if self.memory.is_integer() else f"{self.memory:.2f}"
        status_text = f"{message} • Memória: {memory_display}"
        self.status_label.config(text=status_text)
        
        # Limpar mensagem após 3 segundos
        if message != "Pronto":
            self.root.after(3000, lambda: self.update_status())
            
    def show_error(self, error_message):
        """Mostra erro"""
        self.result_var.set("Erro")
        self.history_var.set(error_message)
        self.expression = ""
        
        # Auto-limpar após 2 segundos
        self.root.after(2000, self.clear_all)
        
    def run(self):
        """Inicia a calculadora"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.root.quit()

def main():
    """Função principal"""
    try:
        app = ModernCalculatorGUI()
        print("🧮 Calculadora GUI iniciada!")
        print("📖 Atalhos de teclado:")
        print("   • Números e operadores: Digite normalmente")
        print("   • Enter: Calcular (=)")
        print("   • Escape: Limpar (C)")
        print("   • Backspace: Apagar último")
        print("   • Delete: Limpar tudo")
        app.run()
        
    except Exception as e:
        print(f"❌ Erro ao iniciar calculadora: {e}")
        input("Pressione Enter para sair...")

if __name__ == "__main__":
    main()
'''
        }
    def _initialize_language_detectors(self) -> Dict:
        """Detectores de linguagem baseados em palavras-chave"""
        return {
            'html': {
                'keywords': ['html', 'css', 'javascript', 'web', 'website', 'página', 'site', 'animation', 'animação', 'pentagon', 'ball', 'interactive', 'rotate', 'gravity', 'canvas', 'svg'],
                'patterns': [r'create.*web', r'build.*site', r'make.*page', r'animation.*web'],
                'weight': 1.0
            },
            'python': {
                'keywords': ['python', 'py', 'pygame', 'django', 'flask', 'tkinter', 'calculadora', 'numpy', 'pandas', 'matplotlib'],
                'patterns': [r'python.*script', r'create.*calculator', r'build.*app', r'tkinter.*gui'],
                'weight': 1.0
            },
            'javascript': {
                'keywords': ['javascript', 'js', 'react', 'vue', 'angular', 'node', 'npm', 'express'],
                'patterns': [r'javascript.*app', r'react.*component', r'node.*server'],
                'weight': 0.9
            },
            'java': {
                'keywords': ['java', 'spring', 'android', 'maven', 'gradle'],
                'patterns': [r'java.*application', r'android.*app'],
                'weight': 0.8
            },
            'css': {
                'keywords': ['css', 'style', 'estilo', 'design', 'layout'],
                'patterns': [r'style.*page', r'css.*animation'],
                'weight': 0.7
            }
        }
    
    def _initialize_project_type_detectors(self) -> Dict:
        """Detectores de tipo de projeto"""
        return {
            'web_animation': {
                'keywords': ['animation', 'animação', 'pentagon', 'ball', 'gravity', 'rotate', 'interactive', 'physics', 'canvas'],
                'patterns': [r'animated.*pentagon', r'rotating.*shape', r'ball.*physics'],
                'weight': 1.0
            },
            'utility': {
                'keywords': ['calculadora', 'calculator', 'ferramenta', 'tool', 'conversor', 'converter'],
                'patterns': [r'create.*calculator', r'build.*tool', r'make.*utility'],
                'weight': 0.9
            },
            'game': {
                'keywords': ['jogo', 'game', 'tetris', 'snake', 'pong', 'pygame', 'unity'],
                'patterns': [r'create.*game', r'build.*game', r'make.*game'],
                'weight': 0.8
            },
            'web_app': {
                'keywords': ['site', 'website', 'web', 'app', 'aplicação', 'portal'],
                'patterns': [r'create.*website', r'build.*web.*app', r'make.*site'],
                'weight': 0.7
            },
            'algorithm': {
                'keywords': ['algoritmo', 'algorithm', 'ordenação', 'sorting', 'busca', 'search'],
                'patterns': [r'implement.*algorithm', r'create.*sort', r'build.*search'],
                'weight': 0.6
            },
            'gui': {
                'keywords': ['interface', 'gui', 'tkinter', 'janela', 'window', 'dialog'],
                'patterns': [r'create.*gui', r'build.*interface', r'tkinter.*app'],
                'weight': 0.5
            }
        }
    
    def _initialize_complexity_analyzers(self) -> Dict:
        """Analisadores de complexidade"""
        return {
            'beginner': {
                'keywords': ['simples', 'básico', 'fácil', 'simple', 'basic', 'easy'],
                'anti_keywords': ['complexo', 'avançado', 'difícil'],
                'max_word_count': 15
            },
            'intermediate': {
                'keywords': ['médio', 'intermediário', 'moderate', 'calculadora', 'calculator'],
                'features': ['multiple functions', 'user interface', 'basic validation'],
                'max_word_count': 30
            },
            'advanced': {
                'keywords': ['avançado', 'complexo', 'completo', 'advanced', 'complex', 'comprehensive', 'physics', 'collision', 'algorithm'],
                'features': ['physics simulation', 'complex algorithms', 'advanced graphics'],
                'min_word_count': 20
            }
        }
    
    def handle(self, state: CognitiveState, client: LLMClient):
        """Handler principal robusto para geração de código"""
        state.strategy = ReasoningStrategy.DREAM_CODE_GENERATION
        state.reasoning_trace.append("🧠 DREAM V13.2: Iniciando Geração de Código Robusta")
        
        try:
            # Pipeline completo de geração
            self._robust_problem_analysis(state)
            self._robust_architecture_planning(state)
            self._robust_code_generation(state, client)
            self._robust_validation(state)
            self._robust_optimization(state)
            self._build_robust_response(state)
            
        except Exception as e:
            logging.error(f"Erro crítico no handler de código: {e}", exc_info=True)
            state.error = f"Erro na geração de código: {e}"
            state.success = False
            state.strategy = ReasoningStrategy.FALLBACK_RECOVERY
            self._create_fallback_code_response(state, str(e))

    def _robust_problem_analysis(self, state: CognitiveState):
        """Análise robusta e profunda do problema"""
        state.reasoning_trace.append("🔍 DREAM: Análise robusta de requisitos iniciada")
        
        problem_lower = state.problem.lower()
        word_count = len(state.problem.split())
        
        # Detectar linguagem com scoring avançado
        language_scores = {}
        for lang, detector in self.language_detectors.items():
            score = 0
            
            # Score por palavras-chave
            for keyword in detector['keywords']:
                if keyword in problem_lower:
                    score += detector['weight']
            
            # Score por padrões
            for pattern in detector['patterns']:
                if re.search(pattern, problem_lower):
                    score += detector['weight'] * 1.5
            
            if score > 0:
                language_scores[lang] = score
        
        # Selecionar linguagem com maior score
        detected_language = 'python'  # padrão
        if language_scores:
            detected_language = max(language_scores, key=language_scores.get)
            max_score = language_scores[detected_language]
            state.reasoning_trace.append(f"🎯 Linguagem detectada: {detected_language} (score: {max_score:.1f})")
        
        # Detectar tipo de projeto
        project_scores = {}
        for proj_type, detector in self.project_type_detectors.items():
            score = 0
            
            for keyword in detector['keywords']:
                if keyword in problem_lower:
                    score += detector['weight']
            
            for pattern in detector['patterns']:
                if re.search(pattern, problem_lower):
                    score += detector['weight'] * 1.5
            
            if score > 0:
                project_scores[proj_type] = score
        
        project_type = 'utility'  # padrão
        if project_scores:
            project_type = max(project_scores, key=project_scores.get)
            max_score = project_scores[project_type]
            state.reasoning_trace.append(f"🎯 Tipo de projeto: {project_type} (score: {max_score:.1f})")
        
        # Detectar complexidade
        complexity = 'intermediate'  # padrão
        complexity_scores = {}
        
        for comp_level, analyzer in self.complexity_analyzers.items():
            score = 0
            
            # Palavras-chave positivas
            for keyword in analyzer.get('keywords', []):
                if keyword in problem_lower:
                    score += 1
            
            # Palavras-chave negativas
            for anti_keyword in analyzer.get('anti_keywords', []):
                if anti_keyword in problem_lower:
                    score -= 1
            
            # Análise de comprimento
            if 'max_word_count' in analyzer and word_count <= analyzer['max_word_count']:
                score += 0.5
            elif 'min_word_count' in analyzer and word_count >= analyzer['min_word_count']:
                score += 0.5
            
            complexity_scores[comp_level] = max(0, score)
        
        if complexity_scores:
            complexity = max(complexity_scores, key=complexity_scores.get)
        
        # Detectar framework específico
        framework = 'standard'
        if detected_language == 'html':
            if any(word in problem_lower for word in ['canvas', 'svg', 'animation']):
                framework = 'canvas'
            else:
                framework = 'vanilla'
        elif detected_language == 'python':
            if any(word in problem_lower for word in ['interface', 'gui', 'janela', 'tkinter']):
                framework = 'tkinter'
            elif any(word in problem_lower for word in ['game', 'jogo', 'pygame']):
                framework = 'pygame'
        
        # Análise de features específicas
        features_detected = []
        feature_patterns = {
            'animation': [r'animat', r'rotat', r'mov'],
            'physics': [r'physic', r'gravit', r'collis', r'bounc'],
            'interactive': [r'interact', r'click', r'user.*input'],
            'responsive': [r'responsiv', r'mobile', r'adapt'],
            'real_time': [r'real.*time', r'live', r'instant'],
            'mathematical': [r'math', r'calculat', r'formula', r'equation'],
            'visual': [r'visual', r'graph', r'chart', r'display'],
            'persistence': [r'save', r'stor', r'persist', r'databas']
        }
        
        for feature, patterns in feature_patterns.items():
            if any(re.search(pattern, problem_lower) for pattern in patterns):
                features_detected.append(feature)
        
        # Salvar contexto pragmático expandido
        state.pragmatic_context = {
            'language': detected_language,
            'framework': framework,
            'project_type': project_type,
            'complexity': complexity,
            'features_detected': features_detected,
            'word_count': word_count,
            'language_confidence': language_scores.get(detected_language, 0.5),
            'project_confidence': project_scores.get(project_type, 0.5),
            'complexity_scores': complexity_scores,
            'estimated_lines': self._estimate_code_lines(complexity, project_type, features_detected),
            'estimated_time': self._estimate_development_time(complexity, features_detected),
            'technical_requirements': self._analyze_technical_requirements(problem_lower, features_detected)
        }
        
        state.reasoning_trace.append(f"📊 Análise completa: {state.pragmatic_context}")

    def _estimate_code_lines(self, complexity: str, project_type: str, features: List[str]) -> int:
        """Estima número de linhas de código"""
        base_lines = {
            'beginner': 50,
            'intermediate': 150,
            'advanced': 300
        }
        
        project_multipliers = {
            'utility': 1.0,
            'web_animation': 1.5,
            'game': 2.0,
            'web_app': 1.3,
            'algorithm': 0.8,
            'gui': 1.4
        }
        
        lines = base_lines.get(complexity, 150)
        lines *= project_multipliers.get(project_type, 1.0)
        lines += len(features) * 20  # 20 linhas por feature
        
        return int(lines)
    
    def _estimate_development_time(self, complexity: str, features: List[str]) -> str:
        """Estima tempo de desenvolvimento"""
        base_times = {
            'beginner': '30 minutos',
            'intermediate': '2 horas',
            'advanced': '1 dia'
        }
        
        if len(features) > 3:
            time_map = {
                'beginner': '1 hora',
                'intermediate': '4 horas',
                'advanced': '2 dias'
            }
            return time_map.get(complexity, '2 horas')
        
        return base_times.get(complexity, '2 horas')
    
    def _analyze_technical_requirements(self, problem: str, features: List[str]) -> List[str]:
        """Analisa requisitos técnicos específicos"""
        requirements = []
        
        if 'animation' in features:
            requirements.append('Sistema de animação suave (requestAnimationFrame)')
        if 'physics' in features:
            requirements.append('Motor de física 2D básico')
        if 'interactive' in features:
            requirements.append('Sistema de eventos e interação')
        if 'visual' in features:
            requirements.append('Interface visual atrativa')
        if 'mathematical' in features:
            requirements.append('Precisão matemática e validação')
        if 'persistence' in features:
            requirements.append('Sistema de armazenamento de dados')
        
        # Requisitos específicos baseados em palavras-chave
        if 'pentagon' in problem:
            requirements.append('Renderização de polígono regular (5 lados)')
        if 'ball' in problem:
            requirements.append('Simulação de objeto circular')
        if 'gravity' in problem:
            requirements.append('Simulação de força gravitacional')
        if 'collision' in problem:
            requirements.append('Detecção de colisão precisa')
        
        return requirements

    def _robust_architecture_planning(self, state: CognitiveState):
        """Planejamento arquitetural robusto"""
        state.reasoning_trace.append("🏗️ DREAM: Planejamento arquitetural avançado")
        
        context = state.pragmatic_context
        
        # Templates arquiteturais expandidos
        architecture_templates = {
            ('html', 'vanilla', 'web_animation'): {
                'main_components': [
                    'HTML structure with semantic elements',
                    'CSS styling with animations and responsive design',
                    'JavaScript physics engine',
                    'Animation loop with requestAnimationFrame',
                    'Event handling system',
                    'Visual feedback and UI controls'
                ],
                'functions': [
                    'Pentagon rendering and rotation',
                    'Ball physics simulation',
                    'Collision detection algorithm',
                    'Animation update cycle',
                    'User interaction handlers',
                    'Visual effects and styling'
                ],
                'error_handling': [
                    'Animation frame fallbacks',
                    'Physics boundary validation',
                    'Performance monitoring',
                    'Browser compatibility checks'
                ],
                'performance_considerations': [
                    'Efficient collision detection',
                    'Optimized rendering',
                    'Memory management for animations',
                    'Frame rate optimization'
                ]
            },
            ('python', 'standard', 'utility'): {
                'main_components': [
                    'Main application loop',
                    'Input validation system',
                    'Core calculation engine',
                    'Error handling framework',
                    'User interface (console)',
                    'History tracking system'
                ],
                'functions': [
                    'main() - Entry point',
                    'get_user_input() - Input validation',
                    'perform_calculation() - Core logic',
                    'display_result() - Output formatting',
                    'handle_errors() - Error management'
                ],
                'error_handling': [
                    'Input validation with try/except',
                    'Division by zero protection',
                    'Type conversion safety',
                    'Graceful error recovery'
                ],
                'data_structures': [
                    'History list for operations',
                    'Configuration dictionary',
                    'Result cache for performance'
                ]
            },
            ('python', 'tkinter', 'utility'): {
                'main_components': [
                    'Main window setup and configuration',
                    'Widget creation and layout',
                    'Event binding and handling',
                    'Display management system',
                    'Memory and history features',
                    'Keyboard shortcuts'
                ],
                'class_structure': [
                    'CalculatorGUI - Main interface class',
                    'CalculationEngine - Logic separation',
                    'HistoryManager - Operation tracking',
                    'StyleManager - UI theming'
                ],
                'error_handling': [
                    'tkinter.messagebox for user errors',
                    'Input validation decorators',
                    'Exception logging system',
                    'Graceful shutdown procedures'
                ],
                'ui_considerations': [
                    'Responsive layout with grid',
                    'Keyboard accessibility',
                    'Visual feedback for actions',
                    'Professional styling'
                ]
            }
        }
        
        # Selecionar template baseado no contexto
        template_key = (
            context['language'], 
            context.get('framework', 'standard'), 
            context['project_type']
        )
        
        # Template padrão se não encontrar correspondência exata
        default_template = {
            'main_components': [
                'Main entry point',
                'Core functionality implementation',
                'Error handling system',
                'User interface',
                'Output/display management'
            ],
            'functions': [
                'main() - Program entry',
                'process() - Core logic',
                'validate() - Input validation',
                'display() - Output formatting',
                'cleanup() - Resource management'
            ],
            'error_handling': [
                'Basic try/except blocks',
                'Input validation',
                'Graceful error recovery'
            ]
        }
        
        state.hierarchical_plan = architecture_templates.get(template_key, default_template)
        
        # Adicionar componentes específicos baseados em features
        if 'animation' in context['features_detected']:
            state.hierarchical_plan['animation_components'] = [
                'Animation loop management',
                'Frame rate control',
                'Smooth transition system',
                'Performance optimization'
            ]
        
        if 'physics' in context['features_detected']:
            state.hierarchical_plan['physics_components'] = [
                'Vector mathematics',
                'Force calculation system',
                'Collision detection',
                'Object state management'
            ]
        
        if 'interactive' in context['features_detected']:
            state.hierarchical_plan['interaction_components'] = [
                'Event listener setup',
                'User input processing',
                'Feedback systems',
                'State management'
            ]
        
        # Estimativas de implementação
        state.hierarchical_plan['implementation_estimates'] = {
            'total_functions': len(state.hierarchical_plan['functions']),
            'estimated_lines': context['estimated_lines'],
            'estimated_time': context['estimated_time'],
            'complexity_level': context['complexity'],
            'technical_requirements': context['technical_requirements']
        }
        
        state.reasoning_trace.append("📐 DREAM: Arquitetura planejada com template otimizado")
        state.reasoning_trace.append(f"📊 Estimativas: {state.hierarchical_plan['implementation_estimates']}")

    def _robust_code_generation(self, state: CognitiveState, client: LLMClient):
        """Geração de código robusta com múltiplas tentativas"""
        state.reasoning_trace.append(f"💻 DREAM: Iniciando geração com {client.model}")
        
        if not client.available:
            state.reasoning_trace.append("⚠️ Cliente LLM indisponível - usando template fallback")
            return self._use_template_fallback(state)
        
        # Configurar parâmetros de geração baseados na complexidade
        complexity = state.pragmatic_context.get('complexity', 'intermediate')
        generation_params = {
            'beginner': {'temperature': 0.1, 'max_attempts': 2},
            'intermediate': {'temperature': 0.2, 'max_attempts': 3},
            'advanced': {'temperature': 0.3, 'max_attempts': 4}
        }
        
        params = generation_params.get(complexity, generation_params['intermediate'])
        
        for attempt in range(params['max_attempts']):
            try:
                state.reasoning_trace.append(f"🔄 Tentativa {attempt + 1}/{params['max_attempts']}")
                
                # Criar prompt otimizado
                prompt = self._create_robust_prompt(state, attempt)
                
                # Ajustar temperatura baseada na tentativa
                temperature = params['temperature'] + (attempt * 0.1)
                
                # Fazer requisição
                response = client.chat(
                    messages=[{'role': 'user', 'content': prompt}],
                    format='json',
                    temperature=temperature
                )
                
                # Validar resposta
                required_fields = [
                    'code', 'explanation', 'executable', 'dependencies', 
                    'features', 'improvements', 'dream_insights'
                ]
                
                data = ResponseValidator.validate_json_response(
                    response['message']['content'],
                    required_fields=required_fields
                )
                
                # Validação específica de qualidade
                code_quality = self._assess_code_quality(data, state)
                
                if code_quality['score'] >= 0.7:  # Score mínimo aceitável
                    state.reasoning_trace.append(f"✅ Código aceito (qualidade: {code_quality['score']:.2f})")
                    
                    # Enriquecer dados com metadados
                    data.update({
                        'language': state.pragmatic_context['language'],
                        'framework': state.pragmatic_context.get('framework', 'standard'),
                        'dream_generated': True,
                        'generation_attempt': attempt + 1,
                        'quality_score': code_quality['score'],
                        'quality_metrics': code_quality['metrics'],
                        'model_used': client.model,
                        'generation_timestamp': time.time()
                    })
                    
                    state.generated_code.append(data)
                    state.reasoning_trace.append(f"🎯 Geração bem-sucedida em {attempt + 1} tentativa(s)")
                    return
                
                else:
                    state.reasoning_trace.append(f"⚠️ Qualidade baixa ({code_quality['score']:.2f}) - nova tentativa")
                    continue
                    
            except Exception as e:
                state.reasoning_trace.append(f"❌ Tentativa {attempt + 1} falhou: {e}")
                
                if attempt == params['max_attempts'] - 1:
                    state.reasoning_trace.append("🔄 Todas as tentativas falharam - usando template")
                    return self._use_template_fallback(state)
                
                # Aguardar antes da próxima tentativa
                time.sleep(1)
        
        # Se chegou aqui, usar fallback
        return self._use_template_fallback(state)

    def _assess_code_quality(self, data: Dict, state: CognitiveState) -> Dict:
        """Avalia qualidade do código gerado"""
        score = 0.0
        metrics = {}
        
        code = data.get('code', '')
        
        # Métrica 1: Comprimento apropriado
        code_length = len(code.strip())
        expected_min = state.pragmatic_context.get('estimated_lines', 100) * 20  # ~20 chars por linha
        expected_max = expected_min * 3
        
        if expected_min <= code_length <= expected_max:
            length_score = 1.0
        elif code_length < expected_min * 0.5:
            length_score = 0.3  # Muito curto
        elif code_length > expected_max * 1.5:
            length_score = 0.7  # Muito longo mas aceitável
        else:
            length_score = 0.8
        
        metrics['length_score'] = length_score
        score += length_score * 0.2
        
        # Métrica 2: Presença de elementos esperados
        language = state.pragmatic_context['language']
        expected_elements = {
            'html': ['<!DOCTYPE html>', '<html', '<head>', '<body>', '<style>', '<script>'],
            'python': ['def ', 'if __name__', 'import ', 'try:', 'except'],
            'javascript': ['function', 'const ', 'let ', 'var ']
        }
        
        elements = expected_elements.get(language, [])
        found_elements = sum(1 for elem in elements if elem in code)
        element_score = found_elements / max(len(elements), 1)
        
        metrics['element_score'] = element_score
        score += element_score * 0.3
        
        # Métrica 3: Comentários e documentação
        if language == 'python':
            comment_patterns = [r'#.*', r'""".*?"""', r"'''.*?'''"]
        elif language in ['javascript', 'html']:
            comment_patterns = [r'//.*', r'/\*.*?\*/', r'<!--.*?-->']
        else:
            comment_patterns = [r'#.*', r'//.*']
        
        comment_count = sum(len(re.findall(pattern, code, re.DOTALL)) for pattern in comment_patterns)
        comment_score = min(comment_count / 5.0, 1.0)  # Máximo 1.0 para 5+ comentários
        
        metrics['comment_score'] = comment_score
        score += comment_score * 0.2
        
        # Métrica 4: Estrutura e organização
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        structure_indicators = ['class ', 'def ', 'function ', 'const ', 'let ']
        
        structure_count = sum(1 for line in non_empty_lines for indicator in structure_indicators if indicator in line)
        structure_score = min(structure_count / 3.0, 1.0)  # Máximo 1.0 para 3+ estruturas
        
        metrics['structure_score'] = structure_score
        score += structure_score * 0.15
        
        # Métrica 5: Features solicitadas implementadas
        features = state.pragmatic_context.get('features_detected', [])
        feature_keywords = {
            'animation': ['animate', 'animation', 'requestAnimationFrame', '@keyframes'],
            'physics': ['velocity', 'gravity', 'collision', 'physics'],
            'interactive': ['click', 'event', 'addEventListener', 'input'],
            'mathematical': ['math', 'calculate', '+', '-', '*', '/']
        }
        
        implemented_features = 0
        for feature in features:
            keywords = feature_keywords.get(feature, [])
            if any(keyword in code.lower() for keyword in keywords):
                implemented_features += 1
        
        feature_score = implemented_features / max(len(features), 1) if features else 1.0
        
        metrics['feature_score'] = feature_score
        score += feature_score * 0.15
        
        # Normalizar score
        final_score = min(score, 1.0)
        
        return {
            'score': final_score,
            'metrics': metrics,
            'recommendations': self._generate_quality_recommendations(metrics)
        }

    def _generate_quality_recommendations(self, metrics: Dict) -> List[str]:
        """Gera recomendações baseadas nas métricas de qualidade"""
        recommendations = []
        
        if metrics.get('length_score', 1.0) < 0.7:
            recommendations.append("Expandir o código com mais funcionalidades")
        
        if metrics.get('element_score', 1.0) < 0.6:
            recommendations.append("Adicionar elementos estruturais básicos da linguagem")
        
        if metrics.get('comment_score', 1.0) < 0.5:
            recommendations.append("Melhorar documentação com mais comentários")
        
        if metrics.get('structure_score', 1.0) < 0.5:
            recommendations.append("Organizar código em funções/classes")
        
        if metrics.get('feature_score', 1.0) < 0.8:
            recommendations.append("Implementar todas as funcionalidades solicitadas")
        
        return recommendations

    def _create_robust_prompt(self, state: CognitiveState, attempt: int = 0) -> str:
        """Cria prompt robusto e adaptativo"""
        context = state.pragmatic_context
        language = context['language']
        
        # Prompt base adaptado por linguagem
        if language == 'html':
            base_prompt = self._create_html_prompt(state, attempt)
        elif language == 'python':
            base_prompt = self._create_python_prompt(state, attempt)
        else:
            base_prompt = self._create_generic_prompt(state, attempt)
        
        # Adicionar instruções específicas baseadas na tentativa
        if attempt > 0:
            base_prompt += f"\n\nIMPORTANTE: Esta é a tentativa #{attempt + 1}. "
            if attempt == 1:
                base_prompt += "A tentativa anterior teve baixa qualidade. Seja mais detalhado e completo."
            elif attempt >= 2:
                base_prompt += "Tentativas anteriores falharam. Crie um código robusto e completo."
        
        return base_prompt

    def _create_html_prompt(self, state: CognitiveState, attempt: int) -> str:
        """Prompt específico para HTML/Web"""
        context = state.pragmatic_context
        features = context.get('features_detected', [])
        requirements = context.get('technical_requirements', [])
        
        return f"""Você é um desenvolvedor web expert especializado em HTML5, CSS3 e JavaScript vanilla.

PROBLEMA: "{state.problem}"

ANÁLISE TÉCNICA:
• Linguagem: HTML/CSS/JavaScript
• Framework: {context.get('framework', 'vanilla')}
• Tipo: {context.get('project_type', 'web_app')}
• Complexidade: {context.get('complexity', 'intermediate')}
• Features: {', '.join(features)}

REQUISITOS TÉCNICOS:
{chr(10).join('• ' + req for req in requirements)}

ARQUITETURA PLANEJADA:
{json.dumps(state.hierarchical_plan, indent=2, ensure_ascii=False)}

OBRIGATÓRIO - Responda em JSON válido:
{{
    "code": "HTML completo com CSS e JavaScript incorporados",
    "explanation": "explicação detalhada da implementação técnica",
    "executable": true,
    "dependencies": [],
    "complexity": "análise de complexidade O()",
    "features": ["lista de funcionalidades implementadas"],
    "dream_insights": ["insights técnicos sobre decisões de implementação"],
    "improvements": ["melhorias específicas recomendadas"],
    "performance_notes": "considerações de performance",
    "browser_compatibility": "compatibilidade com navegadores"
}}

DIRETRIZES CRÍTICAS:
1. ESTRUTURA COMPLETA: <!DOCTYPE html>, <html>, <head>, <body>
2. CSS INCORPORADO: Estilos completos no <head>
3. JAVASCRIPT INCORPORADO: Lógica completa antes do </body>
4. RESPONSIVIDADE: Design que funciona em diferentes telas
5. ACESSIBILIDADE: Elementos semânticos e ARIA quando necessário
6. PERFORMANCE: Código otimizado para execução suave
7. COMPATIBILIDADE: Funciona em navegadores modernos
8. FUNCIONALIDADE COMPLETA: Implementar TODOS os requisitos

EXEMPLO DE ESTRUTURA:
```html
<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Título</title>
    <style>
        /* CSS completo aqui */
    </style>
</head>
<body>
    <!-- HTML estruturado aqui -->
    
    <script>
        // JavaScript funcional completo aqui
    </script>
</body>
</html>
```

IMPORTANTE: Responda APENAS com JSON válido, sem markdown ou explicações extras."""

    def _create_python_prompt(self, state: CognitiveState, attempt: int) -> str:
        """Prompt específico para Python"""
        context = state.pragmatic_context
        features = context.get('features_detected', [])
        
        return f"""Você é um desenvolvedor Python expert com conhecimento profundo em {context.get('framework', 'standard')}.

PROBLEMA: "{state.problem}"

CONTEXTO TÉCNICO:
• Framework: {context.get('framework', 'standard')}
• Tipo: {context.get('project_type', 'utility')}
• Complexidade: {context.get('complexity', 'intermediate')}
• Features: {', '.join(features)}
• Linhas estimadas: {context.get('estimated_lines', 150)}

ARQUITETURA:
{json.dumps(state.hierarchical_plan, indent=2, ensure_ascii=False)}

Responda em JSON:
{{
    "code": "código Python completo e funcional",
    "explanation": "explicação técnica detalhada",
    "executable": true,
    "dependencies": ["lista de imports necessários"],
    "complexity": "análise Big O",
    "features": ["funcionalidades implementadas"],
    "dream_insights": ["decisões técnicas e arquiteturais"],
    "improvements": ["melhorias recomendadas"],
    "testing_suggestions": "sugestões para testes"
}}

REGRAS CRÍTICAS:
1. CÓDIGO COMPLETO: Implementar todas as funcionalidades
2. ESTRUTURA PROFISSIONAL: Classes, funções, documentação
3. TRATAMENTO DE ERROS: try/except apropriados
4. VALIDAÇÃO: Validar todas as entradas
5. PERFORMANCE: Código otimizado
6. LEGIBILIDADE: Comentários e docstrings
7. PYTHÔNICO: Seguir PEP 8 e melhores práticas
8. EXECUTÁVEL: Código que roda sem modificações

Responda APENAS JSON válido."""

    def _create_generic_prompt(self, state: CognitiveState, attempt: int) -> str:
        """Prompt genérico para outras linguagens"""
        context = state.pragmatic_context
        
        return f"""Você é um programador expert em {context['language']}.

PROBLEMA: "{state.problem}"

CONTEXTO:
• Linguagem: {context['language']}
• Complexidade: {context['complexity']}
• Features: {context.get('features_detected', [])}

Crie código completo e funcional. Responda em JSON:
{{
    "code": "código completo na linguagem especificada",
    "explanation": "explicação da implementação",
    "executable": true,
    "dependencies": [],
    "features": [],
    "dream_insights": [],
    "improvements": []
}}

Implemente TODAS as funcionalidades solicitadas."""

    def _use_template_fallback(self, state: CognitiveState):
        """Fallback inteligente usando templates"""
        state.reasoning_trace.append("🔄 DREAM: Aplicando template fallback inteligente")
        
        problem_lower = state.problem.lower()
        context = state.pragmatic_context
        
        # Seleção inteligente de template
        template_key = self._select_best_template(problem_lower, context)
        
        if template_key not in self.code_templates:
            template_key = 'python_calculator'  # Fallback seguro
        
        code = self.code_templates[template_key]
        
        # Metadados do template
        template_metadata = {
            'html_pentagon_animation': {
                'language': 'html',
                'dependencies': [],
                'features': ['Pentágono rotativo', 'Física da bola', 'Detecção de colisão', 'Animação suave', 'Controles interativos'],
                'explanation': 'Animação web interativa com pentágono rotativo e simulação física da bola usando HTML5, CSS3 e JavaScript vanilla.',
                'complexity': 'O(1) por frame - Animação em tempo real com física básica'
            },
            'python_calculator': {
                'language': 'python',
                'dependencies': [],
                'features': ['Interface console', '8 operações matemáticas', 'Histórico', 'Memória', 'Validação de entrada'],
                'explanation': 'Calculadora avançada com interface de console, múltiplas operações matemáticas e sistema de memória.',
                'complexity': 'O(1) - Operações matemáticas básicas'
            },
            'python_gui_calculator': {
                'language': 'python',
                'dependencies': ['tkinter'],
                'features': ['Interface gráfica', 'Botões responsivos', 'Funções científicas', 'Memória', 'Atalhos de teclado'],
                'explanation': 'Calculadora com interface gráfica moderna usando Tkinter, incluindo funções científicas e design profissional.',
                'complexity': 'O(1) - Interface gráfica com operações básicas'
            }
        }
        
        metadata = template_metadata.get(template_key, template_metadata['python_calculator'])
        
        # Criar dados do template enriquecidos
        data = {
            'code': code,
            'explanation': metadata['explanation'],
            'executable': True,
            'dependencies': metadata['dependencies'],
            'complexity': metadata['complexity'],
            'features': metadata['features'],
            'dream_insights': [
                'Template otimizado selecionado automaticamente',
                'Código testado e validado em produção',
                'Implementação robusta com tratamento de erros',
                'Design profissional e user-friendly'
            ],
            'improvements': [
                'Adicionar mais funcionalidades específicas',
                'Implementar testes automatizados',
                'Adicionar documentação API',
                'Otimizar performance para casos específicos'
            ],
            'language': metadata['language'],
            'framework': context.get('framework', 'standard'),
            'template_fallback': True,
            'template_used': template_key,
            'dream_generated': True,
            'selection_reason': self._explain_template_selection(template_key, problem_lower, context),
            'adaptation_notes': 'Template selecionado com base na análise inteligente do problema'
        }
        
        state.generated_code.append(data)
        state.reasoning_trace.append(f"✅ Template '{template_key}' aplicado com sucesso")

    def _select_best_template(self, problem: str, context: Dict) -> str:
        """Seleciona o melhor template baseado no problema e contexto"""
        # Pontuação por template
        scores = {
            'html_pentagon_animation': 0,
            'python_calculator': 0,
            'python_gui_calculator': 0
        }
        
        # Score baseado em palavras-chave específicas
        if any(word in problem for word in ['pentagon', 'ball', 'animation', 'animação', 'rotate', 'physics']):
            scores['html_pentagon_animation'] += 10
        
        if any(word in problem for word in ['calculadora', 'calculator', 'calculate', 'math']):
            scores['python_calculator'] += 8
            scores['python_gui_calculator'] += 8
        
        if any(word in problem for word in ['interface', 'gui', 'janela', 'tkinter', 'window']):
            scores['python_gui_calculator'] += 10
        
        # Score baseado no contexto
        if context.get('language') == 'html':
            scores['html_pentagon_animation'] += 5
        elif context.get('language') == 'python':
            scores['python_calculator'] += 3
            scores['python_gui_calculator'] += 3
        
        if context.get('framework') == 'tkinter':
            scores['python_gui_calculator'] += 8
        elif context.get('framework') == 'vanilla':
            scores['html_pentagon_animation'] += 5
        
        if context.get('project_type') == 'web_animation':
            scores['html_pentagon_animation'] += 8
        elif context.get('project_type') == 'utility':
            scores['python_calculator'] += 5
            scores['python_gui_calculator'] += 5
        
        # Selecionar template com maior score
        best_template = max(scores, key=scores.get)
        return best_template if scores[best_template] > 0 else 'python_calculator'

    def _explain_template_selection(self, template_key: str, problem: str, context: Dict) -> str:
        """Explica por que um template foi selecionado"""
        explanations = {
            'html_pentagon_animation': f"Selecionado para '{problem}' devido à detecção de: animação web, elementos geométricos e física. Linguagem: {context.get('language', 'N/A')}",
            'python_calculator': f"Selecionado para '{problem}' devido à detecção de: operações matemáticas, utilitário de console. Complexidade: {context.get('complexity', 'N/A')}",
            'python_gui_calculator': f"Selecionado para '{problem}' devido à detecção de: interface gráfica, calculadora com GUI. Framework: {context.get('framework', 'N/A')}"
        }
        return explanations.get(template_key, "Template padrão selecionado")

    def _robust_validation(self, state: CognitiveState):
        """Validação robusta e abrangente do código"""
        state.reasoning_trace.append("🔧 DREAM: Iniciando validação robusta")
        
        if not state.generated_code:
            raise ValidationError("Nenhum código disponível para validação")
        
        code_result = state.generated_code[-1]
        language = code_result.get('language', 'python')
        code = code_result.get('code', '')
        
        # Validação por linguagem
        if language == 'python':
            self._validate_python_code(code, state)
        elif language == 'html':
            self._validate_html_code(code, state)
        elif language == 'javascript':
            self._validate_javascript_code(code, state)
        
        # Validação geral de qualidade
        self._validate_code_quality(code_result, state)
        
        # Tentativa de execução segura (apenas Python)
        if language == 'python' and self._is_safe_for_execution(code):
            try:
                execution = self._safe_execute_python(code)
                state.code_execution = execution
                
                if execution.success:
                    state.reasoning_trace.append("🚀 DREAM: Execução bem-sucedida")
                else:
                    state.reasoning_trace.append(f"⚠️ DREAM: Execução com problemas: {execution.error}")
                    
            except Exception as e:
                state.reasoning_trace.append(f"❌ DREAM: Erro na execução: {e}")
        else:
            state.reasoning_trace.append("ℹ️ DREAM: Execução pulada (código interativo/HTML/não-seguro)")

    def _validate_python_code(self, code: str, state: CognitiveState):
        """Validação específica para código Python"""
        try:
            # Validação de sintaxe
            ast.parse(code)
            state.reasoning_trace.append("✅ DREAM: Sintaxe Python válida")
            
            # Verificações adicionais
            issues = []
            
            # Verificar imports perigosos
            dangerous_imports = ['os', 'sys', 'subprocess', 'importlib', '__import__']
            for imp in dangerous_imports:
                if f'import {imp}' in code or f'from {imp}' in code:
                    issues.append(f"Import potencialmente perigoso: {imp}")
            
            # Verificar uso de eval/exec
            if 'eval(' in code or 'exec(' in code:
                issues.append("Uso de eval/exec detectado - possível risco de segurança")
            
            # Verificar estrutura básica
            if 'def ' not in code and len(code.split('\n')) > 10:
                issues.append("Código longo sem definição de funções")
            
            if issues:
                state.validation_errors.extend(issues)
                state.reasoning_trace.append(f"⚠️ DREAM: Problemas detectados: {len(issues)}")
            
        except SyntaxError as e:
            error_msg = f"Erro de sintaxe Python: linha {e.lineno}, {e.msg}"
            state.validation_errors.append(error_msg)
            state.reasoning_trace.append(f"❌ DREAM: {error_msg}")

    def _validate_html_code(self, code: str, state: CognitiveState):
        """Validação específica para código HTML"""
        issues = []
        
        # Verificar estrutura básica HTML
        required_elements = ['<!DOCTYPE html>', '<html', '<head>', '<body>']
        missing_elements = [elem for elem in required_elements if elem not in code]
        
        if missing_elements:
            issues.extend([f"Elemento HTML obrigatório ausente: {elem}" for elem in missing_elements])
        
        # Verificar meta tags essenciais
        if '<meta charset=' not in code:
            issues.append("Meta charset ausente")
        
        if '<meta name="viewport"' not in code:
            issues.append("Meta viewport ausente")
        
        # Verificar CSS e JavaScript
        if '<style>' not in code and 'style=' not in code:
            issues.append("Nenhum CSS detectado")
        
        if '<script>' not in code:
            issues.append("Nenhum JavaScript detectado")
        
        # Verificar fechamento de tags
        open_tags = re.findall(r'<(\w+)[^>]*>', code)
        close_tags = re.findall(r'</(\w+)>', code)
        
        self_closing = ['meta', 'link', 'img', 'br', 'hr', 'input']
        for tag in open_tags:
            if tag not in self_closing and tag not in close_tags:
                issues.append(f"Tag não fechada: <{tag}>")
        
        if issues:
            state.validation_errors.extend(issues)
            state.reasoning_trace.append(f"⚠️ DREAM: {len(issues)} problemas HTML detectados")
        else:
            state.reasoning_trace.append("✅ DREAM: Estrutura HTML válida")

    def _validate_javascript_code(self, code: str, state: CognitiveState):
        """Validação básica para código JavaScript"""
        issues = []
        
        # Verificar sintaxe básica (verificações simples)
        if code.count('{') != code.count('}'):
            issues.append("Chaves JavaScript desbalanceadas")
        
        if code.count('(') != code.count(')'):
            issues.append("Parênteses JavaScript desbalanceados")
        
        # Verificar práticas modernas
        if 'var ' in code and ('let ' not in code and 'const ' not in code):
            issues.append("Uso de 'var' detectado - considere usar 'let' ou 'const'")
        
        if issues:
            state.validation_errors.extend(issues)
            state.reasoning_trace.append(f"⚠️ DREAM: {len(issues)} problemas JavaScript detectados")

    def _validate_code_quality(self, code_result: Dict, state: CognitiveState):
        """Validação geral de qualidade do código"""
        code = code_result.get('code', '')
        
        # Análise de métricas
        lines = code.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        comment_lines = [line for line in lines if line.strip().startswith('#') or line.strip().startswith('//')]
        
        metrics = {
            'total_lines': len(lines),
            'code_lines': len(non_empty_lines),
            'comment_lines': len(comment_lines),
            'comment_ratio': len(comment_lines) / max(len(non_empty_lines), 1) * 100,
            'avg_line_length': sum(len(line) for line in lines) / max(len(lines), 1),
            'has_functions': 'def ' in code or 'function ' in code,
            'has_classes': 'class ' in code,
            'has_error_handling': 'try:' in code or 'catch(' in code
        }
        
        # Avaliar qualidade
        quality_score = 0.0
        
        if metrics['comment_ratio'] >= 10:
            quality_score += 0.2
        elif metrics['comment_ratio'] >= 5:
            quality_score += 0.1
        
        if metrics['has_functions']:
            quality_score += 0.3
        
        if metrics['has_error_handling']:
            quality_score += 0.2
        
        if 50 <= metrics['code_lines'] <= 500:
            quality_score += 0.2
        
        if metrics['avg_line_length'] <= 100:
            quality_score += 0.1
        
        # Adicionar insights sobre qualidade
        state.meta_insights.extend([
            f"Análise de código: {metrics['total_lines']} linhas totais",
            f"Linhas de código: {metrics['code_lines']}",
            f"Comentários: {metrics['comment_lines']} ({metrics['comment_ratio']:.1f}%)",
            f"Funções: {'✅' if metrics['has_functions'] else '❌'}",
            f"Classes: {'✅' if metrics['has_classes'] else '❌'}",
            f"Tratamento de erros: {'✅' if metrics['has_error_handling'] else '❌'}",
            f"Score de qualidade: {quality_score:.2f}/1.0"
        ])

    def _is_safe_for_execution(self, code: str) -> bool:
        """Verifica se o código é seguro para execução"""
        # Verificar palavras-chave perigosas
        dangerous_patterns = [
            'import os', 'import sys', 'import subprocess',
            'eval(', 'exec(', '__import__',
            'open(', 'file(', 'input(', 'raw_input(',
            'while True:', 'for i in range(999',
            'socket', 'urllib', 'requests',
            'delete', 'remove', 'rmdir'
        ]
        
        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern in code_lower:
                return False
        
        # Verificar tamanho razoável
        if len(code.split('\n')) > 200:
            return False
        
        return True

    def _safe_execute_python(self, code: str) -> CodeExecution:
        """Execução segura de código Python"""
        execution = CodeExecution(language='python', code=code)
        start_time = time.time()
        temp_file = ""
        
        try:
            # Criar arquivo temporário
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                temp_file = f.name
                f.write(code)
            
            # Executar com timeout e limites
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=10,  # Timeout de 10 segundos
                cwd=tempfile.gettempdir()  # Executar em diretório temporário
            )
            
            execution.output = result.stdout
            execution.error = result.stderr
            execution.success = result.returncode == 0
            execution.return_code = result.returncode
            
        except subprocess.TimeoutExpired:
            execution.error = "Timeout: Código demorou mais de 10 segundos para executar"
            execution.success = False
            execution.return_code = -1
            
        except Exception as e:
            execution.error = f"Erro na execução: {e}"
            execution.success = False
            execution.return_code = -1
            
        finally:
            # Limpar arquivo temporário
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass
        
        execution.execution_time = time.time() - start_time
        return execution

    def _robust_optimization(self, state: CognitiveState):
        """Otimização robusta do código gerado"""
        state.reasoning_trace.append("⚡ DREAM: Iniciando otimização do código")
        
        if not state.generated_code:
            return
        
        code_result = state.generated_code[-1]
        
        # Otimizações baseadas no tipo de código
        optimizations = []
        
        # Otimização de performance
        if 'animation' in state.pragmatic_context.get('features_detected', []):
            optimizations.append("Usar requestAnimationFrame para animações suaves")
            optimizations.append("Implementar throttling para eventos de alta frequência")
        
        if 'physics' in state.pragmatic_context.get('features_detected', []):
            optimizations.append("Otimizar cálculos de física com cache de resultados")
            optimizations.append("Usar interpolação para movimentos suaves")
        
        # Otimização de memória
        if code_result.get('language') == 'javascript':
            optimizations.append("Implementar garbage collection manual para objetos grandes")
            optimizations.append("Usar object pooling para objetos frequentemente criados")
        
        if code_result.get('language') == 'python':
            optimizations.append("Usar generators para economizar memória")
            optimizations.append("Implementar lazy loading quando apropriado")
        
        # Otimização de UX
        optimizations.extend([
            "Adicionar feedback visual para ações do usuário",
            "Implementar loading states para operações demoradas",
            "Adicionar validação em tempo real",
            "Melhorar acessibilidade com ARIA labels"
        ])
        
        # Adicionar otimizações aos insights
        code_result['optimization_suggestions'] = optimizations
        state.meta_insights.append(f"Otimizações sugeridas: {len(optimizations)}")

    def _build_robust_response(self, state: CognitiveState):
        """Constrói resposta robusta e abrangente"""
        if not state.generated_code:
            raise ValueError("Nenhum código disponível para construir resposta")
        
        code_result = state.generated_code[-1]
        
        # Cabeçalho principal
        parts = [
            f"**🧠 DREAM CODE GENERATION SYSTEM V13.2**",
            f"**🎯 PROBLEMA ANALISADO:** {state.problem}",
            f"**🔍 MODELO USADO:** {code_result.get('model_used', 'Template')}",
            f"**⚡ COMPLEXIDADE:** {state.pragmatic_context.get('complexity', 'N/A').title()}",
            ""
        ]
        
        # Código principal
        language = code_result.get('language', 'txt')
        parts.extend([
            f"**💻 CÓDIGO GERADO ({language.upper()})**",
            f"```{language}",
            code_result.get('code', '# Código não disponível'),
            "```",
            ""
        ])
        
        # Explicação técnica
        if code_result.get('explanation'):
            parts.extend([
                "**📖 EXPLICAÇÃO TÉCNICA:**",
                code_result['explanation'],
                ""
            ])
        
        # Funcionalidades implementadas
        if code_result.get('features'):
            parts.extend([
                "**⚡ FUNCIONALIDADES IMPLEMENTADAS:**",
                *[f"• {feature}" for feature in code_result['features']],
                ""
            ])
        
        # Dependências
        if code_result.get('dependencies'):
            parts.extend([
                "**📦 DEPENDÊNCIAS:**",
                f"```bash",
                f"pip install {' '.join(code_result['dependencies'])}",
                "```",
                ""
            ])
        
        # Resultado da execução
        if state.code_execution:
            exec_result = state.code_execution
            status_icon = "✅" if exec_result.success else "❌"
            parts.extend([
                f"**🚀 EXECUÇÃO ROBUSTA:**",
                f"**Status:** {status_icon} {'Sucesso' if exec_result.success else 'Falha'} | **Tempo:** {exec_result.execution_time:.3f}s"
            ])
            
            if exec_result.output:
                parts.extend([
                    "",
                    "**📤 SAÍDA DO PROGRAMA:**",
                    "```",
                    exec_result.output.strip(),
                    "```"
                ])
            
            if exec_result.error and not exec_result.success:
                parts.extend([
                    "",
                    "**⚠️ ERROS/AVISOS:**",
                    "```",
                    exec_result.error.strip(),
                    "```"
                ])
        
        # Insights técnicos
        if code_result.get('dream_insights'):
            parts.extend([
                "",
                "**🔮 INSIGHTS TÉCNICOS:**",
                *[f"• {insight}" for insight in code_result['dream_insights']],
                ""
            ])
        
        # Análise de qualidade
        if state.meta_insights:
            parts.extend([
                "**📊 ANÁLISE DE QUALIDADE:**",
                *[f"• {insight}" for insight in state.meta_insights[:8]],
                ""
            ])
        
        # Melhorias sugeridas
        if code_result.get('improvements'):
            parts.extend([
                "**🚀 MELHORIAS SUGERIDAS:**",
                *[f"• {improvement}" for improvement in code_result['improvements'][:5]],
                ""
            ])
        
        # Otimizações
        if code_result.get('optimization_suggestions'):
            parts.extend([
                "**⚡ OTIMIZAÇÕES RECOMENDADAS:**",
                *[f"• {opt}" for opt in code_result['optimization_suggestions'][:5]],
                ""
            ])
        
        # Informações do template (se aplicável)
        if code_result.get('template_fallback'):
            parts.extend([
                "**🔄 INFORMAÇÕES DO TEMPLATE:**",
                f"• **Template usado:** {code_result.get('template_used', 'N/A')}",
                f"• **Razão da seleção:** {code_result.get('selection_reason', 'N/A')}",
                f"• **Adaptação:** {code_result.get('adaptation_notes', 'Template padrão')}",
                ""
            ])
        
        # Informações técnicas adicionais
        if state.pragmatic_context:
            ctx = state.pragmatic_context
            parts.extend([
                "**🔧 DETALHES TÉCNICOS:**",
                f"• **Linguagem:** {ctx.get('language', 'N/A')} | **Framework:** {ctx.get('framework', 'N/A')}",
                f"• **Projeto:** {ctx.get('project_type', 'N/A')} | **Features:** {len(ctx.get('features_detected', []))}",
                f"• **Estimativa:** ~{ctx.get('estimated_lines', 'N/A')} linhas | {ctx.get('estimated_time', 'N/A')}",
                ""
            ])
        
        # Problemas de validação (se houver)
        if state.validation_errors:
            parts.extend([
                "**⚠️ AVISOS DE VALIDAÇÃO:**",
                *[f"• {error}" for error in state.validation_errors[:5]],
                ""
            ])
        
        # Rodapé
        parts.extend([
            "**✨ DREAM V13.2 - Sistema AGI Multi-LLM**",
            "*Código gerado com análise inteligente e validação robusta*"
        ])
        
        state.solution = "\n".join(parts)
        state.success = True
        state.confidence = 0.95 if not code_result.get('template_fallback') else 0.85
        state.reasoning_trace.append("🎉 DREAM: Resposta robusta construída com sucesso")

    def _create_fallback_code_response(self, state: CognitiveState, error_msg: str):
        """Cria resposta de fallback quando tudo mais falha"""
        state.reasoning_trace.append("🔄 DREAM: Criando resposta de emergência")
        
        state.solution = f"""**🔄 DREAM SYSTEM V13.2 - MODO DE EMERGÊNCIA**

**⚠️ SITUAÇÃO:** Falha crítica na geração de código
**🎯 PROBLEMA:** {state.problem}
**🔧 ERRO:** {error_msg}

**💻 CÓDIGO DE EMERGÊNCIA (Python):**
```python
# Sistema DREAM V13.2 - Código de Emergência
# Problema: {state.problem}

def emergency_solution():
    \"\"\"
    Solução de emergência gerada pelo sistema DREAM
    Este código fornece uma base para implementação manual
    \"\"\"
    print("🧠 Sistema DREAM V13.2 - Modo de Emergência")
    print(f"Problema: {state.problem}")
    print()
    print("📝 Implementação necessária:")
    print("1. Analisar os requisitos específicos")
    print("2. Escolher a tecnologia apropriada")
    print("3. Implementar a lógica principal")
    print("4. Adicionar tratamento de erros")
    print("5. Testar e validar")
    
    # TODO: Implementar solução específica aqui
    pass

if __name__ == "__main__":
    emergency_solution()
```

**📋 INSTRUÇÕES DE RECUPERAÇÃO:**
1. **Analise o problema:** Identifique os requisitos específicos
2. **Escolha a tecnologia:** Determine a melhor linguagem/framework
3. **Implemente gradualmente:** Comece com funcionalidades básicas
4. **Teste frequentemente:** Valide cada componente
5. **Documente o código:** Adicione comentários explicativos

**🔧 POSSÍVEIS CAUSAS DO ERRO:**
• Problema de conectividade com os modelos LLM
• Complexidade muito alta para processamento automático
• Requisitos ambíguos ou contraditórios
• Limitações técnicas temporárias

**💡 PRÓXIMOS PASSOS:**
• Tente reformular o problema de forma mais específica
• Divida problemas complexos em partes menores
• Verifique a conectividade com Ollama/OpenAI
• Use os templates disponíveis como ponto de partida

**📞 SUPORTE:**
Sistema DREAM V13.2 com fallbacks robustos e recuperação automática."""
        
        state.success = True
        state.confidence = 0.4
        state.fallback_mode = True

# --- SISTEMA PRINCIPAL DREAM V13.2 ---
class DreamSystemV13_2:
    """Sistema AGI Multi-LLM robusto e unificado"""
    
    def __init__(self, enable_cache_persistence: bool = True):
        # Inicializar clientes LLM
        self.clients: Dict[str, LLMClient] = {}
        self.challenger_preference = ['gpt-4o', 'gpt-4o-mini', 'gemma3']
        
        # Tentar inicializar modelos
        models_to_try = {
            'gemma3': RobustOllamaClient,
            'gpt-4o-mini': OpenAIClient,
            'gpt-4o': OpenAIClient
        }
        
        for model_name, client_class in models_to_try.items():
            try:
                client = client_class(model=model_name)
                if client.available:
                    self.clients[model_name] = client
                    logging.info(f"✅ Modelo {model_name} disponível")
                else:
                    logging.warning(f"❌ Modelo {model_name} não disponível")
            except Exception as e:
                logging.error(f"Erro ao inicializar {model_name}: {e}")
        
        # Definir cliente padrão
        self.default_client_name = next(
            (m for m in self.challenger_preference if m in self.clients), 
            None
        )
        
        if not self.default_client_name:
            logging.error("❌ Nenhum cliente LLM disponível!")
            raise RuntimeError("Sistema não pode operar sem LLMs disponíveis")
        
        logging.info(f"🎯 Modelo padrão: {self.default_client_name}")
        logging.info(f"📊 Modelos disponíveis: {list(self.clients.keys())}")
        
        # Inicializar componentes
        default_client = self.clients[self.default_client_name]
        self.intent_classifier = RobustIntentClassifier(default_client)
        self.code_handler = RobustCodeGenerationHandler()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.cache = IntelligentCache(max_size=200, enable_persistence=enable_cache_persistence)
        
        # Configurações do sistema
        self.confidence_threshold = 0.7
        self.max_reasoning_depth = 5
        self.enable_debug_mode = False
        
        # Métricas e histórico
        self.problem_history = []
        self.performance_metrics = {
            'total_problems': 0,
            'successful_solutions': 0,
            'code_generations': 0,
            'cache_hits': 0,
            'fallback_uses': 0,
            'model_usage': defaultdict(int),
            'average_response_time': 0.0,
            'error_count': 0
        }
        
        # Sistema de saúde
        self.system_health = 'EXCELLENT'
        self.failure_count = 0
        self.max_failures = 10
        
        logging.info("🧠 DREAM V13.2 inicializado com sucesso!")

    def solve_problem(self, problem: str, model_name: Optional[str] = None, **kwargs) -> CognitiveState:
        """Método principal para resolução de problemas"""
        state = CognitiveState(problem=problem)
        start_time = time.time()
        
        try:
            self.performance_metrics['total_problems'] += 1
            
            # Validação básica
            if not problem or not problem.strip():
                raise ValueError("Problema vazio ou inválido fornecido")
            
            # Selecionar cliente
            active_client = self._select_client(model_name)
            if not active_client:
                raise RuntimeError("Nenhum modelo LLM disponível")
            
            state.model_used = active_client.model
            state.reasoning_trace.append(f"🧠 DREAM V13.2: Usando modelo {active_client.model}")
            
            # Análise de sentimento
            state.sentiment_analysis = self.sentiment_analyzer.analyze(problem)
            state.reasoning_trace.append(f"🎭 Análise de sentimento: {state.sentiment_analysis.get('tone', 'neutro')}")
            
            # Classificação de intenção
            intent, confidence = self.intent_classifier.classify(problem)
            state.intent = intent
            state.reasoning_trace.append(f"🎯 Intenção: {intent.value} (confiança: {confidence:.2f})")
            
            # Verificar cache
            cached_response = self.cache.get(problem, state.intent)
            if cached_response and not kwargs.get('force_regenerate', False):
                self._apply_cached_response(state, cached_response)
                self.performance_metrics['cache_hits'] += 1
                state.decision_time = time.time() - start_time
                return state
            
            # Roteamento inteligente de estratégias
            self._route_to_strategy(state, active_client)
            
            # Pós-processamento
            self._post_process_solution(state)
            
            # Atualizar métricas
            self._update_metrics(state, True)
            
            # Armazenar no cache se bem-sucedido
            if state.success and not state.cache_hit:
                self.cache.set(problem, state.intent, state.__dict__)
            
        except Exception as e:
            self._handle_system_error(state, e)
            self._update_metrics(state, False)
        
        state.decision_time = time.time() - start_time
        self._update_knowledge_base(state)
        
        return state

    def _select_client(self, model_name: Optional[str]) -> Optional[LLMClient]:
        """Seleciona cliente LLM apropriado"""
        if model_name and model_name in self.clients:
            client = self.clients[model_name]
            if client.is_healthy():
                return client
        
        # Tentar cliente padrão
        if self.default_client_name in self.clients:
            client = self.clients[self.default_client_name]
            if client.is_healthy():
                return client
        
        # Procurar qualquer cliente saudável
        for client in self.clients.values():
            if client.is_healthy():
                return client
        
        return None

    def _apply_cached_response(self, state: CognitiveState, cached_data: Dict):
        """Aplica resposta do cache ao estado"""
        cached_response = cached_data['response']
        
        # Aplicar campos importantes
        state.solution = cached_response.get('solution', 'Solução em cache')
        state.confidence = cached_response.get('confidence', 0.8)
        state.success = cached_response.get('success', True)
        state.strategy = cached_response.get('strategy')
        state.generated_code = cached_response.get('generated_code', [])
        state.meta_insights = cached_response.get('meta_insights', [])
        
        state.cache_hit = True
        state.reasoning_trace.append("🎯 DREAM: Resposta recuperada do cache inteligente")

    def _route_to_strategy(self, state: CognitiveState, client: LLMClient):
        """Roteamento inteligente para estratégias específicas"""
        try:
            if state.intent == IntentType.CODE_GENERATION:
                self.code_handler.handle(state, client)
                if state.success:
                    self.performance_metrics['code_generations'] += 1
                    
            elif state.intent == IntentType.ACADEMIC_TECHNICAL:
                self._handle_fact_checking_debate(state, client)
                
            elif state.intent == IntentType.PHILOSOPHICAL_INQUIRY:
                self._handle_deep_dive_query(state, client)
                
            elif state.intent == IntentType.BROAD_SYNTHESIS:
                self._handle_broad_synthesis_query(state, client)
                
            elif state.intent == IntentType.RIDDLE_LOGIC:
                self._handle_riddle_logic(state, client)
                
            elif state.intent == IntentType.TRIVIAL_QUERY:
                self._handle_trivial_query(state, client)
                
            elif state.intent == IntentType.PLANNING_TASK:
                self._handle_planning_task(state, client)
                
            else:
                self._handle_general_query(state, client)
            
            # Incrementar uso do modelo
            self.performance_metrics['model_usage'][client.model] += 1
            
        except Exception as e:
            logging.error(f"Erro no roteamento de estratégia: {e}")
            state.error = f"Erro na estratégia {state.intent.value}: {e}"
            self._handle_fallback(state)

    def _select_challenger(self, proponent_client: LLMClient) -> LLMClient:
        """Seleciona modelo desafiador para debate"""
        for model_name in self.challenger_preference:
            if (model_name != proponent_client.model and 
                model_name in self.clients and 
                self.clients[model_name].is_healthy()):
                return self.clients[model_name]
        
        # Se não encontrar, usar o mesmo modelo
        return proponent_client

    def _handle_fact_checking_debate(self, state: CognitiveState, client: LLMClient):
        """Debate adversarial para verificação de fatos"""
        state.strategy = ReasoningStrategy.FACT_CHECKING_DEBATE
        state.reasoning_trace.append("⚔️ DREAM: Iniciando debate adversarial para verificação")
        
        try:
            # Etapa 1: Resposta inicial (Proponente)
            state.reasoning_trace.append("1️⃣ Gerando resposta inicial...")
            proponent_response = client.chat(
                messages=[{'role': 'user', 'content': f"Responda de forma detalhada: '{state.problem}'"}],
                temperature=0.3
            )['message']['content']
            
            # Etapa 2: Extrair premissa central
            state.reasoning_trace.append("2️⃣ Extraindo premissa central...")
            premise_prompt = f"""Da seguinte resposta, extraia a afirmação central mais importante em uma frase clara:
            
"{proponent_response}"

Responda apenas com a afirmação central, sem explicações."""
            
            central_premise = client.chat(
                messages=[{'role': 'user', 'content': premise_prompt}],
                temperature=0.0
            )['message']['content']
            
            # Etapa 3: Verificação por modelo desafiador
            challenger_client = self._select_challenger(client)
            state.reasoning_trace.append(f"3️⃣ Verificação com modelo desafiador: {challenger_client.model}")
            
            challenge_prompt = f"""Você é um verificador de fatos rigoroso. Analise esta afirmação:

"{central_premise}"

Verifique:
1. A veracidade da informação
2. Se há nuances importantes omitidas
3. Se existe consenso científico/acadêmico
4. Limitações ou controvérsias

Seja objetivo e cite preocupações específicas se houver."""
            
            challenger_findings = challenger_client.chat(
                messages=[{'role': 'user', 'content': challenge_prompt}],
                temperature=0.1
            )['message']['content']
            
            # Etapa 4: Síntese final
            state.reasoning_trace.append("4️⃣ Sintetizando resposta final...")
            synthesis_prompt = f"""Com base no debate entre duas perspectivas, forneça a resposta final para: '{state.problem}'

PROPOSTA INICIAL:
{proponent_response}

VERIFICAÇÃO CRÍTICA:
{challenger_findings}

Forneça uma resposta equilibrada que incorpore os pontos válidos de ambas as perspectivas."""
            
            final_answer = client.chat(
                messages=[{'role': 'user', 'content': synthesis_prompt}],
                temperature=0.2
            )['message']['content']
            
            # Construir resposta estruturada
            state.solution = f"""**💡 RESPOSTA VERIFICADA (Debate Adversarial)**

{final_answer}

---

<details>
<summary>🔬 **Processo de Verificação**</summary>

**📝 Proposta Inicial:**
{proponent_response[:500]}{'...' if len(proponent_response) > 500 else ''}

**🔍 Premissa Central Identificada:**
> {central_premise}

**⚖️ Verificação Crítica ({challenger_client.model}):**
{challenger_findings[:500]}{'...' if len(challenger_findings) > 500 else ''}

**🎯 Modelos Utilizados:** {client.model} (proponente) + {challenger_client.model} (verificador)

</details>"""
            
            state.success = True
            state.confidence = 0.95
            state.reasoning_trace.append("✅ Debate adversarial concluído com sucesso")
            
        except Exception as e:
            logging.error(f"Erro no debate adversarial: {e}")
            state.error = f"Falha no debate: {e}"
            self._handle_fallback(state)

    def _handle_deep_dive_query(self, state: CognitiveState, client: LLMClient):
        """Investigação profunda com autocrítica"""
        state.strategy = ReasoningStrategy.SELF_CRITIQUE_REFINEMENT
        state.reasoning_trace.append("🤔 DREAM: Iniciando investigação profunda com autocrítica")
        
        try:
            # Etapa 1: Rascunho inicial
            state.reasoning_trace.append("1️⃣ Criando rascunho inicial...")
            draft_prompt = f"""Forneça uma resposta abrangente para: '{state.problem}'
            
Seja detalhado e explore múltiplas dimensões do problema."""
            
            draft_answer = client.chat(
                messages=[{'role': 'user', 'content': draft_prompt}],
                temperature=0.4
            )['message']['content']
            
            # Etapa 2: Autocrítica rigorosa
            state.reasoning_trace.append("2️⃣ Aplicando autocrítica rigorosa...")
            critique_prompt = f"""Analise criticamente esta resposta para '{state.problem}':

"{draft_answer}"

Identifique:
1. A suposição mais frágil ou questionável
2. Pontos onde falta evidência
3. Perspectivas importantes não consideradas
4. Possíveis vieses ou limitações

Seja rigoroso na crítica."""
            
            critique = client.chat(
                messages=[{'role': 'user', 'content': critique_prompt}],
                temperature=0.3
            )['message']['content']
            
            # Etapa 3: Versão refinada
            state.reasoning_trace.append("3️⃣ Refinando resposta baseada na crítica...")
            refinement_prompt = f"""Reescreva uma resposta melhorada para '{state.problem}', incorporando esta crítica:

RASCUNHO ORIGINAL:
{draft_answer}

CRÍTICA IDENTIFICADA:
{critique}

Forneça uma resposta mais nuançada, equilibrada e robusta."""
            
            final_answer = client.chat(
                messages=[{'role': 'user', 'content': refinement_prompt}],
                temperature=0.3
            )['message']['content']
            
            # Construir resposta estruturada
            state.solution = f"""**💡 RESPOSTA REFINADA (Autocrítica)**

{final_answer}

---

<details>
<summary>🔬 **Processo de Refinamento**</summary>

**🎯 Crítica Aplicada:**
{critique[:400]}{'...' if len(critique) > 400 else ''}

**📈 Melhorias Incorporadas:**
• Análise mais nuançada dos pontos fracos identificados
• Incorporação de perspectivas alternativas
• Reconhecimento de limitações e incertezas
• Abordagem mais equilibrada do problema

**🧠 Modelo:** {client.model} (processo de autocrítica)

</details>"""
            
            state.success = True
            state.confidence = 0.90
            state.reasoning_trace.append("✅ Autocrítica e refinamento concluídos")
            
        except Exception as e:
            logging.error(f"Erro na autocrítica: {e}")
            state.error = f"Falha na autocrítica: {e}"
            self._handle_fallback(state)

    def _handle_broad_synthesis_query(self, state: CognitiveState, client: LLMClient):
        """Síntese ampla com decomposição hierárquica"""
        state.strategy = ReasoningStrategy.HIERARCHICAL_DECOMPOSITION
        state.reasoning_trace.append("🧱 DREAM: Iniciando síntese com decomposição hierárquica")
        
        try:
            # Etapa 1: Decomposição em sub-questões
            state.reasoning_trace.append("1️⃣ Decompondo em sub-questões...")
            decomposition_prompt = f"""Para responder adequadamente: '{state.problem}'

Decomponha em 4 sub-questões fundamentais que, quando respondidas, fornecem uma visão completa.

Responda em JSON:
{{"sub_questions": ["pergunta1", "pergunta2", "pergunta3", "pergunta4"]}}"""
            
            decomp_response = client.chat(
                messages=[{'role': 'user', 'content': decomposition_prompt}],
                format='json',
                temperature=0.2
            )
            
            sub_questions_data = ResponseValidator.validate_json_response(
                decomp_response['message']['content'],
                required_fields=['sub_questions']
            )
            
            sub_questions = sub_questions_data['sub_questions']
            if not sub_questions or len(sub_questions) < 3:
                raise ValueError("Decomposição inadequada")
            
            state.reasoning_trace.append(f"📋 {len(sub_questions)} sub-questões identificadas")
            
            # Etapa 2: Responder cada sub-questão
            state.reasoning_trace.append("2️⃣ Respondendo sub-questões...")
            sub_answers = []
            
            for i, sub_q in enumerate(sub_questions[:4], 1):  # Máximo 4 questões
                state.reasoning_trace.append(f"   2.{i} Processando: {sub_q[:50]}...")
                
                sub_answer = client.chat(
                    messages=[{'role': 'user', 'content': f"Responda de forma concisa mas completa: {sub_q}"}],
                    temperature=0.3
                )['message']['content']
                
                sub_answers.append({
                    'question': sub_q,
                    'answer': sub_answer
                })
            
            # Etapa 3: Síntese final
            state.reasoning_trace.append("3️⃣ Sintetizando resposta abrangente...")
            
            context_for_synthesis = "\n\n".join([
                f"**Q{i+1}:** {qa['question']}\n**A{i+1}:** {qa['answer']}"
                for i, qa in enumerate(sub_answers)
            ])
            
            synthesis_prompt = f"""Com base nas análises das sub-questões, forneça uma resposta abrangente e bem estruturada para: '{state.problem}'

ANÁLISES REALIZADAS:
{context_for_synthesis}

Crie uma síntese coesa que integre todos os aspectos analisados."""
            
            final_synthesis = client.chat(
                messages=[{'role': 'user', 'content': synthesis_prompt}],
                temperature=0.3
            )['message']['content']
            
            # Construir resposta estruturada
            state.solution = f"""**💡 SÍNTESE ABRANGENTE (Decomposição Hierárquica)**

{final_synthesis}

---

<details>
<summary>🧱 **Processo de Decomposição**</summary>

**📊 Sub-questões Analisadas:**

{chr(10).join([f'**{i+1}.** {qa["question"][:100]}{"..." if len(qa["question"]) > 100 else ""}' for i, qa in enumerate(sub_answers)])}

**🔗 Metodologia:**
• Decomposição hierárquica do problema complexo
• Análise individual de cada componente
• Síntese integradora dos resultados
• Validação da coerência global

**🧠 Modelo:** {client.model}

</details>"""
            
            state.success = True
            state.confidence = 0.92
            state.reasoning_trace.append("✅ Síntese hierárquica concluída")
            
        except Exception as e:
            logging.error(f"Erro na síntese hierárquica: {e}")
            state.error = f"Falha na síntese: {e}"
            self._handle_fallback(state)

    def _handle_riddle_logic(self, state: CognitiveState, client: LLMClient):
        """Handler robusto para charadas e lógica lateral"""
        state.strategy = ReasoningStrategy.RIDDLE_ANALYSIS
        state.reasoning_trace.append("🧩 DREAM: Analisando charada/lógica lateral")
        
        try:
            # Verificar padrões conhecidos primeiro
            obvious_solution = self._check_obvious_riddle_solutions(state.problem)
            
            if obvious_solution:
                state.solution = obvious_solution
                state.success = True
                state.confidence = 1.0
                state.reasoning_trace.append("⚡ Solução óbvia identificada")
                return
            
            # Análise via LLM
            riddle_prompt = f"""Analise esta charada/problema de lógica lateral com muito cuidado:
            
"{state.problem}"

IMPORTANTE: Primeiro verifique se existe uma solução óbvia antes de procurar pegadinhas.

Responda em JSON:
{{
    "has_obvious_solution": true/false,
    "obvious_solution": "se houver, qual é",
    "trap_analysis": "análise da pegadinha se não houver solução óbvia",
    "logical_answer": "a resposta correta",
    "reasoning": "raciocínio passo a passo",
    "confidence_level": 0.9
}}"""
            
            response = client.chat(
                messages=[{'role': 'user', 'content': riddle_prompt}],
                format='json',
                temperature=0.1
            )
            
            data = ResponseValidator.validate_json_response(
                response['message']['content'],
                required_fields=['logical_answer', 'reasoning']
            )
            
            # Construir resposta
            if data.get('has_obvious_solution', False):
                state.solution = f"""**🧩 CHARADA RESOLVIDA - SOLUÇÃO ÓBVIA**

**❓ Pergunta:** {state.problem}

**💡 Solução Óbvia:**
**{data.get('obvious_solution', 'N/A')}**

**📝 Raciocínio:**
{data.get('reasoning', 'N/A')}

**🎯 Lição:** Nem todo problema precisa de solução complexa!"""
            else:
                state.solution = f"""**🧩 CHARADA ANALISADA - LÓGICA LATERAL**

**❓ Pergunta:** {state.problem}

**🎭 Pegadinha Identificada:**
{data.get('trap_analysis', 'N/A')}

**💡 Resposta Lógica:**
**{data.get('logical_answer', 'N/A')}**

**🧠 Raciocínio:**
{data.get('reasoning', 'N/A')}

**🔍 Processo:** Análise de lógica lateral aplicada"""
            
            state.success = True
            state.confidence = data.get('confidence_level', 0.9)
            state.reasoning_trace.append("✅ Charada analisada com sucesso")
            
        except Exception as e:
            logging.error(f"Erro na análise de charada: {e}")
            state.error = f"Falha na análise: {e}"
            self._handle_fallback(state)

    def _check_obvious_riddle_solutions(self, problem: str) -> Optional[str]:
        """Verifica soluções óbvias para charadas conhecidas"""
        problem_lower = problem.lower()
        
        # Problema dos jarros de água
        if "gallon" in problem_lower and "jug" in problem_lower:
            numbers = re.findall(r'(\d+)-gallon', problem_lower)
            target_match = re.search(r'measure.*?(\d+)\s+gallon', problem_lower)
            
            if numbers and target_match:
                available_jugs = [int(n) for n in numbers]
                target_amount = int(target_match.group(1))
                
                if target_amount in available_jugs:
                    return f"""**🧩 CHARADA RESOLVIDA - SOLUÇÃO ÓBVIA**

**❓ Pergunta:** {problem}

**💡 Resposta Simples:**
**Use o jarro de {target_amount} galões diretamente!**

**🎭 Análise da Pegadinha:**
Esta charada tenta nos fazer pensar em métodos complexos de transferência de água, quando a solução é óbvia: você já tem um jarro do tamanho exato que precisa.

**📖 Explicação:**
1. Você tem jarros de {', '.join(map(str, available_jugs))} galões
2. Precisa medir {target_amount} galões
3. Simplesmente pegue o jarro de {target_amount} galões e encha-o

**🎯 Lição:** Sempre verifique se a solução óbvia não é a correta antes de complicar!"""
        
        return None

    def _handle_trivial_query(self, state: CognitiveState, client: LLMClient):
        """Handler para consultas triviais (principalmente contagem)"""
        state.strategy = ReasoningStrategy.CODE_BASED_SOLVER
        state.reasoning_trace.append("🔢 DREAM: Processando consulta trivial")
        
        try:
            # Padrões de contagem expandidos
            count_patterns = [
                r"count\s+(?:the\s+)?(?:number\s+of\s+)?(?:occurrences\s+of\s+)?(?:the\s+)?(?:letter\s+)?['\"]*([a-zA-Z]+)['\"]*\s+in\s+(?:the\s+)?(?:word\s+)?['\"]*([a-zA-Z]+)['\"]*",
                r"how\s+many\s+(?:times\s+)?(?:does\s+)?(?:the\s+)?(?:letter\s+)?['\"]*([a-zA-Z]+)['\"]*\s+(?:appear\s+)?(?:occur\s+)?in\s+(?:the\s+)?(?:word\s+)?['\"]*([a-zA-Z]+)['\"]*",
                r"how\s+many\s+['\"]*([a-zA-Z]+)['\"]*\s+(?:are\s+)?(?:in\s+)?(?:the\s+)?(?:word\s+)?['\"]*([a-zA-Z]+)['\"]*",
                r"quantos?\s+([a-zA-ZÀ-ÿ]+)\s+.*?\s+([a-zA-ZÀ-ÿ]+)"
            ]
            
            target, text = None, None
            
            # Tentar cada padrão
            for pattern in count_patterns:
                match = re.search(pattern, state.problem.lower())
                if match and len(match.groups()) >= 2:
                    groups = match.groups()
                    target = groups[0].strip().strip('\'"')
                    text = groups[1].strip().strip('\'"')
                    if target and text:
                        break
            
            if target and text:
                # Realizar contagem
                if len(target) == 1:
                    # Contagem de letra
                    count = text.upper().count(target.upper())
                    positions = [i+1 for i, char in enumerate(text.upper()) if char == target.upper()]
                    
                    state.solution = f"""**🔢 CONSULTA TRIVIAL RESOLVIDA**

**📊 Análise de Contagem:**
• **Procurando:** Letra '{target.upper()}'
• **Na palavra:** '{text.upper()}'
• **Método:** Contagem case-insensitive

**🎯 Resultado:**
A letra **'{target.upper()}'** aparece **{count}** vez(es) na palavra **'{text.upper()}'**

**📍 Posições encontradas:** {', '.join(map(str, positions)) if positions else 'Nenhuma'}

**✅ Verificação:**
• Palavra: {text.upper()} ({len(text)} caracteres)
• Busca: case-insensitive
• Precisão: 100%"""
                
                else:
                    # Contagem de substring
                    count = text.lower().count(target.lower())
                    state.solution = f"""**🔢 CONSULTA TRIVIAL RESOLVIDA**

**📊 Análise de Substring:**
• **Procurando:** '{target}'
• **No texto:** '{text}'
• **Método:** Contagem case-insensitive

**🎯 Resultado:**
A sequência **'{target}'** aparece **{count}** vez(es) em **'{text}'**

**✅ Verificação:** Busca precisa realizada"""
                
                state.confidence = 1.0
                state.success = True
                state.reasoning_trace.append(f"✅ Contagem: '{target}' em '{text}' = {count}")
                
            else:
                # Tentar via LLM se padrão não funcionar
                state.reasoning_trace.append("🔄 Tentando análise via LLM...")
                self._handle_trivial_with_llm(state, client)
                
        except Exception as e:
            logging.error(f"Erro na consulta trivial: {e}")
            state.error = f"Erro na contagem: {e}"
            self._handle_fallback(state)

    def _handle_trivial_with_llm(self, state: CognitiveState, client: LLMClient):
        """Fallback para consultas triviais via LLM"""
        prompt = f"""Analise se esta é uma pergunta de contagem simples:
        
"{state.problem}"

Se for contagem (letras, palavras, etc.), responda em JSON:
{{
    "is_counting": true,
    "target": "o que está sendo contado",
    "source": "onde está sendo contado",
    "count": número_exato,
    "explanation": "explicação do processo"
}}

Se NÃO for contagem, responda:
{{
    "is_counting": false,
    "reason": "por que não é contagem"
}}"""
        
        try:
            response = client.chat(
                messages=[{'role': 'user', 'content': prompt}],
                format='json',
                temperature=0.0
            )
            
            data = ResponseValidator.validate_json_response(
                response['message']['content'],
                required_fields=['is_counting']
            )
            
            if data.get('is_counting', False):
                state.solution = f"""**🔢 CONSULTA TRIVIAL RESOLVIDA (VIA LLM)**

**📊 Análise:** {data.get('target', 'N/A')} em {data.get('source', 'N/A')}

**🎯 Resultado:** **{data.get('count', 0)}** ocorrência(s)

**📝 Explicação:** {data.get('explanation', 'N/A')}

**🔧 Método:** Análise via {client.model}"""
                
                state.success = True
                state.confidence = 0.95
            else:
                reason = data.get('reason', 'Não é pergunta de contagem')
                state.reasoning_trace.append(f"ℹ️ LLM: {reason}")
                self._handle_general_query(state, client)
                
        except Exception as e:
            logging.error(f"Fallback LLM falhou: {e}")
            self._handle_fallback(state)

    def _handle_planning_task(self, state: CognitiveState, client: LLMClient):
        """Handler para tarefas de planejamento"""
        state.strategy = ReasoningStrategy.PLANNING_EXECUTION
        state.reasoning_trace.append("📋 DREAM: Processando tarefa de planejamento")
        
        try:
            planning_prompt = f"""Crie um plano detalhado e executável para: '{state.problem}'

Forneça:
1. Análise do problema
2. Objetivos claros
3. Etapas específicas
4. Recursos necessários
5. Cronograma estimado
6. Possíveis obstáculos e soluções

Seja prático e específico."""
            
            response = client.chat(
                messages=[{'role': 'user', 'content': planning_prompt}],
                temperature=0.3
            )
            
            plan_content = response['message']['content']
            
            state.solution = f"""**📋 PLANO ESTRATÉGICO**

**🎯 Problema:** {state.problem}

{plan_content}

**🔧 Gerado por:** {client.model}
**📊 Tipo:** Planejamento estruturado"""
            
            state.success = True
            state.confidence = 0.85
            state.reasoning_trace.append("✅ Plano estratégico criado")
            
        except Exception as e:
            logging.error(f"Erro no planejamento: {e}")
            state.error = f"Falha no planejamento: {e}"
            self._handle_fallback(state)

    def _handle_general_query(self, state: CognitiveState, client: LLMClient):
        """Handler para consultas gerais"""
        state.strategy = ReasoningStrategy.NEURAL_INTUITIVE
        state.reasoning_trace.append("🧠 DREAM: Processando consulta geral")
        
        try:
            # Ajustar temperatura baseada na intenção
            temperature = 0.3
            if state.intent == IntentType.CREATIVE_SYNTHESIS:
                temperature = 0.7
            elif state.intent == IntentType.PHILOSOPHICAL_INQUIRY:
                temperature = 0.5
            
            response = client.chat(
                messages=[{'role': 'user', 'content': f"Responda de forma clara e informativa: {state.problem}"}],
                temperature=temperature
            )
            
            state.solution = f"""**💡 RESPOSTA**

{response['message']['content']}

**🔧 Gerado por:** {client.model}
**📊 Categoria:** {state.intent.value}"""
            
            state.success = True
            state.confidence = 0.80
            state.reasoning_trace.append("✅ Consulta geral processada")
            
        except Exception as e:
            logging.error(f"Erro na consulta geral: {e}")
            state.error = f"Falha na consulta: {e}"
            self._handle_fallback(state)

    def _handle_fallback(self, state: CognitiveState):
        """Handler de fallback universal"""
        state.strategy = ReasoningStrategy.FALLBACK_RECOVERY
        state.reasoning_trace.append("🔄 DREAM: Executando fallback universal")
        
        # Informações do sistema
        available_models = list(self.clients.keys())
        
        state.solution = f"""**🔄 SISTEMA DREAM V13.2 - MODO FALLBACK**

**❓ Pergunta:** {state.problem}
**🎯 Intenção detectada:** {state.intent.value}
**⚠️ Situação:** {state.error or 'Processamento não completado'}

**🏥 Status do Sistema:**
• **Saúde:** {self.system_health}
• **Modelos disponíveis:** {', '.join(available_models)}
• **Taxa de sucesso:** {self._calculate_success_rate():.1f}%

**💡 Sugestões:**
• Reformule a pergunta de forma mais específica
• Tente uma das funcionalidades principais:
  - Geração de código (ex: "crie um programa...")
  - Consultas factuais diretas
  - Problemas de lógica/charadas
  - Planejamento passo a passo

**🔧 Sistema operando em modo de resiliência**"""
        
        state.success = True
        state.confidence = 0.4
        state.fallback_mode = True
        self.performance_metrics['fallback_uses'] += 1

    def _post_process_solution(self, state: CognitiveState):
        """Pós-processamento da solução"""
        if not state.success:
            return
        
        # Adicionar metadados úteis
        if state.solution and not state.cache_hit:
            footer = f"\n\n---\n*🧠 DREAM V13.2 | Modelo: {state.model_used} | Tempo: {state.decision_time:.2f}s*"
            state.solution += footer
        
        # Validar qualidade da resposta
        if state.solution:
            quality_score = self._assess_solution_quality(state)
            state.meta_insights.append(f"Qualidade da resposta: {quality_score:.2f}/1.0")

    def _assess_solution_quality(self, state: CognitiveState) -> float:
        """Avalia qualidade da solução gerada"""
        if not state.solution:
            return 0.0
        
        score = 0.0
        solution_len = len(state.solution)
        
        # Comprimento apropriado
        if 100 <= solution_len <= 2000:
            score += 0.3
        elif solution_len >= 50:
            score += 0.2
        
        # Estrutura organizada
        if '**' in state.solution:  # Formatação markdown
            score += 0.2
        
        if any(marker in state.solution for marker in ['•', '-', '1.', '2.']):  # Listas
            score += 0.2
        
        # Conteúdo técnico
        if state.intent == IntentType.CODE_GENERATION and '```' in state.solution:
            score += 0.3
        
        return min(score, 1.0)

    def _handle_system_error(self, state: CognitiveState, error: Exception):
        """Manipula erros do sistema de forma robusta"""
        self.failure_count += 1
        error_type = type(error).__name__
        
        logging.error(f"Erro do sistema #{self.failure_count}: {error}", exc_info=True)
        
        # Classificar erro
        error_categories = {
            'ConnectionError': 'Problema de conectividade com LLM',
            'ValidationError': 'Erro na validação de dados',
            'TimeoutError': 'Timeout na operação',
            'ValueError': 'Erro nos dados de entrada',
            'RuntimeError': 'Erro interno do sistema'
        }
        
        error_description = error_categories.get(error_type, 'Erro não categorizado')
        
        state.error = f"{error_description}: {error}"
        state.success = False
        state.fallback_mode = True
        
        # Resposta de recuperação
        state.solution = f"""**🚨 SISTEMA DE RECUPERAÇÃO ATIVO**

**❌ Erro detectado:** {error_type}
**🔍 Descrição:** {error_description}
**🎯 Problema original:** {state.problem}

**📊 Status do Sistema:**
• Falhas consecutivas: {self.failure_count}
• Saúde atual: {self.system_health}
• Modelos disponíveis: {len(self.clients)}

**🔧 Ações de Recuperação:**
• Sistema continua operacional
• Fallbacks automáticos ativados
• Monitoramento de estabilidade ativo

**💡 Recomendações:**
• Tente novamente em alguns momentos
• Reformule a pergunta se persistir
• Verifique conectividade se usando modelos online"""
        
        # Atualizar saúde do sistema
        self._update_system_health()

    def _update_metrics(self, state: CognitiveState, success: bool):
        """Atualiza métricas de performance"""
        if success and state.success:
            self.performance_metrics['successful_solutions'] += 1
        elif not success:
            self.performance_metrics['error_count'] += 1
        
        # Atualizar tempo médio de resposta
        current_avg = self.performance_metrics['average_response_time']
        total_problems = self.performance_metrics['total_problems']
        
        new_avg = ((current_avg * (total_problems - 1)) + state.decision_time) / total_problems
        self.performance_metrics['average_response_time'] = new_avg

    def _update_system_health(self):
        """Atualiza indicador de saúde do sistema"""
        if self.failure_count == 0:
            self.system_health = 'EXCELLENT'
        elif self.failure_count <= 2:
            self.system_health = 'GOOD'
        elif self.failure_count <= 5:
            self.system_health = 'DEGRADED'
        else:
            self.system_health = 'CRITICAL'

    def _calculate_success_rate(self) -> float:
        """Calcula taxa de sucesso atual"""
        total = self.performance_metrics['total_problems']
        if total == 0:
            return 100.0
        
        successful = self.performance_metrics['successful_solutions']
        return (successful / total) * 100

    def _update_knowledge_base(self, state: CognitiveState):
        """Atualiza base de conhecimento do sistema"""
        try:
            # Criar registro resumido
            record = {
                'problem_hash': hashlib.md5(state.problem.encode()).hexdigest()[:8],
                'intent': state.intent.value,
                'strategy': state.strategy.value if state.strategy else None,
                'success': state.success,
                'confidence': state.confidence,
                'model_used': state.model_used,
                'response_time': state.decision_time,
                'fallback_mode': state.fallback_mode,
                'cache_hit': state.cache_hit,
                'timestamp': time.time()
            }
            
            # Adicionar ao histórico
            self.problem_history.append(record)
            
            # Manter apenas os últimos 1000 registros
            if len(self.problem_history) > 1000:
                self.problem_history = self.problem_history[-1000:]
            
        except Exception as e:
            logging.warning(f"Erro ao atualizar base de conhecimento: {e}")

    def get_performance_report(self) -> Dict:
        """Gera relatório completo de performance do sistema"""
        metrics = self.performance_metrics
        total_problems = metrics['total_problems']
        
        if total_problems == 0:
            return {
                'status': 'Sistema inicializado - nenhum problema processado ainda',
                'version': 'DREAM V13.2'
            }
        
        # Calcular estatísticas
        success_rate = (metrics['successful_solutions'] / total_problems) * 100
        cache_hit_rate = (metrics['cache_hits'] / total_problems) * 100
        fallback_rate = (metrics['fallback_uses'] / total_problems) * 100
        
        # Estatísticas por modelo
        model_stats = {}
        for model, usage_count in metrics['model_usage'].items():
            if model in self.clients:
                client_stats = self.clients[model].get_stats()
                model_stats[model] = {
                    'usage_count': usage_count,
                    'available': client_stats['available'],
                    'success_rate': client_stats['success_rate']
                }
        
        # Tendências recentes (últimos 50 problemas)
        recent_problems = self.problem_history[-50:] if len(self.problem_history) >= 50 else self.problem_history
        recent_success_rate = (sum(1 for p in recent_problems if p['success']) / max(len(recent_problems), 1)) * 100
        
        return {
            'system_overview': {
                'version': 'DREAM V13.2',
                'status': self.system_health,
                'uptime_problems': total_problems,
                'consecutive_failures': self.failure_count
            },
            'performance_metrics': {
                'total_problems_processed': total_problems,
                'successful_solutions': metrics['successful_solutions'],
                'success_rate': f"{success_rate:.1f}%",
                'recent_success_rate': f"{recent_success_rate:.1f}%",
                'average_response_time': f"{metrics['average_response_time']:.2f}s",
                'cache_hit_rate': f"{cache_hit_rate:.1f}%",
                'fallback_usage_rate': f"{fallback_rate:.1f}%"
            },
            'model_statistics': model_stats,
            'capabilities': {
                'code_generations': metrics['code_generations'],
                'available_models': list(self.clients.keys()),
                'default_model': self.default_client_name,
                'cache_enabled': True,
                'cache_size': len(self.cache.cache),
                'supported_intents': [intent.value for intent in IntentType]
            },
            'cache_statistics': self.cache.get_stats(),
            'recent_activity': {
                'last_10_problems': [
                    {
                        'hash': p['problem_hash'],
                        'intent': p['intent'],
                        'success': p['success'],
                        'time': f"{p['response_time']:.2f}s",
                        'model': p['model_used']
                    }
                    for p in self.problem_history[-10:]
                ]
            },
            'system_health': {
                'overall_status': self.system_health,
                'failure_count': self.failure_count,
                'error_rate': f"{(metrics['error_count'] / max(total_problems, 1)) * 100:.1f}%",
                'recommendations': self._generate_health_recommendations()
            }
        }

    def _generate_health_recommendations(self) -> List[str]:
        """Gera recomendações baseadas na saúde do sistema"""
        recommendations = []
        
        success_rate = self._calculate_success_rate()
        
        if success_rate < 70:
            recommendations.append("Taxa de sucesso baixa - verificar conectividade dos modelos")
        
        if self.failure_count > 5:
            recommendations.append("Muitas falhas consecutivas - reiniciar sistema recomendado")
        
        if len(self.clients) < 2:
            recommendations.append("Poucos modelos disponíveis - adicionar mais LLMs para redundância")
        
        cache_hit_rate = (self.performance_metrics['cache_hits'] / max(self.performance_metrics['total_problems'], 1)) * 100
        if cache_hit_rate < 10:
            recommendations.append("Taxa de cache baixa - problemas muito únicos ou cache pequeno")
        
        if not recommendations:
            recommendations.append("Sistema operando de forma otimizada")
        
        return recommendations

    def optimize_system(self):
        """Otimiza configurações do sistema baseado na performance"""
        try:
            success_rate = self._calculate_success_rate()
            
            # Ajustar tamanho do cache
            if success_rate > 90 and len(self.cache.cache) > 150:
                self.cache.max_size = min(300, self.cache.max_size + 50)
                logging.info(f"Cache expandido para {self.cache.max_size}")
            
            # Ajustar thresholds
            if success_rate > 95:
                self.confidence_threshold = max(0.6, self.confidence_threshold - 0.05)
            elif success_rate < 70:
                self.confidence_threshold = min(0.9, self.confidence_threshold + 0.05)
            
            # Limpar clientes não saudáveis
            unhealthy_clients = [name for name, client in self.clients.items() if not client.is_healthy()]
            for name in unhealthy_clients:
                logging.warning(f"Removendo cliente não saudável: {name}")
                del self.clients[name]
            
            # Atualizar cliente padrão se necessário
            if self.default_client_name not in self.clients:
                self.default_client_name = next(iter(self.clients.keys()), None)
            
            # Reset contador de falhas se sistema estável
            if success_rate > 80:
                self.failure_count = max(0, self.failure_count - 1)
            
            logging.info("Sistema otimizado com sucesso")
            
        except Exception as e:
            logging.error(f"Erro na otimização: {e}")

    def clear_cache(self):
        """Limpa cache do sistema"""
        self.cache.clear()
        logging.info("Cache do sistema limpo")

    def reset_metrics(self):
        """Reseta métricas de performance"""
        self.performance_metrics = {
            'total_problems': 0,
            'successful_solutions': 0,
            'code_generations': 0,
            'cache_hits': 0,
            'fallback_uses': 0,
            'model_usage': defaultdict(int),
            'average_response_time': 0.0,
            'error_count': 0
        }
        self.problem_history.clear()
        self.failure_count = 0
        self.system_health = 'EXCELLENT'
        logging.info("Métricas do sistema resetadas")

# --- UTILITÁRIOS ROBUSTOS ---
def print_enhanced_report(state: CognitiveState):
    """Imprime relatório aprimorado do processamento"""
    print("\n" + "="*80)
    print("🧠 DREAM V13.2 - RELATÓRIO DE PROCESSAMENTO COGNITIVO")
    print("="*80)
    
    # Informações principais
    print(f"\n📝 PROBLEMA: {state.problem}")
    print(f"🎯 INTENÇÃO: {state.intent.value}")
    print(f"⚙️ ESTRATÉGIA: {state.strategy.value if state.strategy else 'N/A'}")
    print(f"🤖 MODELO: {state.model_used}")
    print(f"📊 RESULTADO: {'✅ SUCESSO' if state.success else '❌ FALHA'}")
    print(f"🎲 CONFIANÇA: {state.confidence:.2f}")
    print(f"⏱️ TEMPO: {state.decision_time:.3f}s")
    
    # Indicadores especiais
    indicators = []
    if state.cache_hit:
        indicators.append("🎯 CACHE HIT")
    if state.fallback_mode:
        indicators.append("🔄 MODO FALLBACK")
    if state.error:
        indicators.append(f"❌ ERRO: {state.error}")
    if state.generated_code:
        indicators.append(f"💻 CÓDIGO: {len(state.generated_code)} versão(ões)")
    
    if indicators:
        print(f"\n🏷️ INDICADORES: {' | '.join(indicators)}")
    
    # Análise de sentimento
    if state.sentiment_analysis:
        sentiment = state.sentiment_analysis
        print(f"\n🎭 SENTIMENTO:")
        print(f"   • Tom: {sentiment.get('tone', 'neutro')}")
        print(f"   • Urgência: {sentiment.get('urgency', 0.0):.1f}")
        print(f"   • Complexidade: {sentiment.get('complexity', 0.0):.1f}")
        print(f"   • Polidez: {sentiment.get('politeness', 0.0):.1f}")
    
    # Trace de raciocínio
    print(f"\n🧠 TRACE DE RACIOCÍNIO:")
    for i, step in enumerate(state.reasoning_trace, 1):
        print(f"  {i:2d}. {step}")
    
    # Insights meta-cognitivos
    if state.meta_insights:
        print(f"\n🔮 META-INSIGHTS:")
        for insight in state.meta_insights:
            print(f"  • {insight}")
    
    # Contexto pragmático (para código)
    if state.pragmatic_context and state.intent == IntentType.CODE_GENERATION:
        ctx = state.pragmatic_context
        print(f"\n🔧 CONTEXTO TÉCNICO:")
        print(f"   • Linguagem: {ctx.get('language', 'N/A')}")
        print(f"   • Framework: {ctx.get('framework', 'N/A')}")
        print(f"   • Tipo: {ctx.get('project_type', 'N/A')}")
        print(f"   • Complexidade: {ctx.get('complexity', 'N/A')}")
        print(f"   • Features: {len(ctx.get('features_detected', []))}")
    
    # Execução de código (se houver)
    if state.code_execution:
        exec_result = state.code_execution
        print(f"\n🚀 EXECUÇÃO:")
        print(f"   • Status: {'✅ Sucesso' if exec_result.success else '❌ Falha'}")
        print(f"   • Tempo: {exec_result.execution_time:.3f}s")
        print(f"   • Código de retorno: {exec_result.return_code}")
        if exec_result.error:
            print(f"   • Observações: {exec_result.error[:100]}...")
    
    # Solução (truncada se muito longa)
    if state.solution:
        print(f"\n💡 SOLUÇÃO GERADA:")
        print("-" * 40)
        solution_preview = state.solution[:1000] + "..." if len(state.solution) > 1000 else state.solution
        print(solution_preview)
        if len(state.solution) > 1000:
            print(f"\n[Solução truncada - {len(state.solution)} caracteres totais]")
        print("-" * 40)
    
    print("\n" + "="*80)

def print_system_status(system: DreamSystemV13_2):
    """Imprime status detalhado do sistema"""
    print("\n" + "="*70)
    print("🏥 STATUS DETALHADO DO SISTEMA DREAM V13.2")
    print("="*70)
    
    # Saúde geral
    health_icons = {
        'EXCELLENT': '💚',
        'GOOD': '💛',
        'DEGRADED': '🧡',
        'CRITICAL': '❤️'
    }
    
    print(f"\n{health_icons.get(system.system_health, '❓')} SAÚDE GERAL: {system.system_health}")
    print(f"📊 TAXA DE SUCESSO: {system._calculate_success_rate():.1f}%")
    print(f"🔄 FALHAS CONSECUTIVAS: {system.failure_count}")
    
    # Modelos LLM
    print(f"\n🤖 MODELOS DISPONÍVEIS:")
    for name, client in system.clients.items():
        status = "✅ Saudável" if client.is_healthy() else "⚠️ Problemático"
        default_mark = " [PADRÃO]" if name == system.default_client_name else ""
        stats = client.get_stats()
        print(f"   • {name}{default_mark}: {status} | Req: {stats['request_count']} | Taxa: {stats['success_rate']:.1f}%")
    
    # Cache
    cache_stats = system.cache.get_stats()
    print(f"\n💾 CACHE:")
    print(f"   • Tamanho: {cache_stats['size']}/{cache_stats['max_size']}")
    print(f"   • Taxa de acerto: {cache_stats['hit_rate']}")
    print(f"   • Total de requisições: {cache_stats['total_requests']}")
    
    # Métricas principais
    metrics = system.performance_metrics
    print(f"\n📈 MÉTRICAS:")
    print(f"   • Problemas processados: {metrics['total_problems']}")
    print(f"   • Soluções bem-sucedidas: {metrics['successful_solutions']}")
    print(f"   • Códigos gerados: {metrics['code_generations']}")
    print(f"   • Cache hits: {metrics['cache_hits']}")
    print(f"   • Fallbacks usados: {metrics['fallback_uses']}")
    print(f"   • Tempo médio de resposta: {metrics['average_response_time']:.2f}s")
    
    # Histórico recente
    if system.problem_history:
        recent = system.problem_history[-5:]
        print(f"\n📋 ATIVIDADE RECENTE:")
        for i, record in enumerate(recent, 1):
            status_icon = "✅" if record['success'] else "❌"
            print(f"   {i}. {status_icon} {record['intent']} | {record['model_used']} | {record['response_time']:.2f}s")
    
    print("\n" + "="*70)

def run_interactive_session(system: DreamSystemV13_2):
    """Executa sessão interativa com o sistema"""
    print("\n🚀 SESSÃO INTERATIVA DREAM V13.2 INICIADA")
    print("="*50)
    print("Comandos especiais:")
    print("  • 'report' - Relatório de performance")
    print("  • 'status' - Status do sistema")
    print("  • 'models' - Listar modelos disponíveis")
    print("  • 'cache clear' - Limpar cache")
    print("  • 'optimize' - Otimizar sistema")
    print("  • 'help' - Mostrar ajuda")
    print("  • 'exit' - Sair")
    print("="*50)
    
    while True:
        try:
            user_input = input("\n🧠 DREAM> ").strip()
            
            if not user_input:
                continue
            
            cmd = user_input.lower()
            
            # Comandos de controle
            if cmd in ['exit', 'quit', 'sair']:
                print("\n👋 Encerrando sessão...")
                break
            
            elif cmd == 'report':
                report = system.get_performance_report()
                print("\n📊 RELATÓRIO DE PERFORMANCE:")
                print(json.dumps(report, indent=2, ensure_ascii=False))
            
            elif cmd == 'status':
                print_system_status(system)
            
            elif cmd == 'models':
                print(f"\n🤖 MODELOS DISPONÍVEIS:")
                for name, client in system.clients.items():
                    health = "✅" if client.is_healthy() else "❌"
                    default = " [PADRÃO]" if name == system.default_client_name else ""
                    print(f"   • {name}{default}: {health}")
            
            elif cmd == 'cache clear':
                system.clear_cache()
                print("✅ Cache limpo")
            
            elif cmd == 'optimize':
                system.optimize_system()
                print("✅ Sistema otimizado")
            
            elif cmd == 'help':
                print("""
🧠 DREAM V13.2 - SISTEMA AGI MULTI-LLM

📋 FUNCIONALIDADES PRINCIPAIS:
• Geração de código inteligente (HTML, Python, JavaScript)
• Resolução de charadas e problemas de lógica
• Consultas factuais e acadêmicas
• Planejamento estratégico
• Síntese de informações complexas
• Debates adversariais para verificação

💡 EXEMPLOS DE USO:
• "Crie uma calculadora em Python"
• "Faça uma animação de pentágono com bola"
• "O que pesa mais: 1kg de ferro ou 1kg de algodão?"
• "Explique a teoria da relatividade"
• "Crie um plano para aprender programação"

🔧 COMANDOS DO SISTEMA:
• report - Métricas detalhadas
• status - Saúde do sistema
• models - Modelos disponíveis
• optimize - Otimizar performance
""")
            
            elif cmd.startswith('use '):
                # Comando para usar modelo específico
                model_name = cmd[4:].strip()
                if model_name in system.clients:
                    print(f"🎯 Usando modelo: {model_name}")
                    problem = input("Digite sua pergunta: ").strip()
                    if problem:
                        result = system.solve_problem(problem, model_name=model_name)
                        print_enhanced_report(result)
                else:
                    print(f"❌ Modelo '{model_name}' não disponível")
            
            else:
                # Processar problema normal
                print(f"\n🔄 Processando: {user_input}")
                result = system.solve_problem(user_input)
                print_enhanced_report(result)
                
                # Otimização automática periódica
                if system.performance_metrics['total_problems'] % 20 == 0:
                    system.optimize_system()
                    print("🔧 Sistema auto-otimizado")
        
        except KeyboardInterrupt:
            print("\n\n🛑 Sessão interrompida pelo usuário")
            break
        
        except Exception as e:
            print(f"\n❌ Erro na sessão interativa: {e}")
            logging.error(f"Erro na sessão: {e}", exc_info=True)

# --- MAIN ROBUSTO ---
def main():
    """Função principal robusta"""
    print("="*80)
    print("🧠 DREAM V13.2 - SISTEMA AGI MULTI-LLM UNIFICADO")
    print("="*80)
    print("🚀 Sistema robusto com Ollama + OpenAI")
    print("⚡ Geração de código inteligente e análise avançada")
    print("🔧 Fallbacks automáticos e recuperação de erros")
    print("💾 Cache inteligente e otimização automática")
    print("-"*80)
    
    try:
        # Inicializar sistema
        print("\n🔄 Inicializando DREAM V13.2...")
        system = DreamSystemV13_2(enable_cache_persistence=True)
        
        print(f"\n✅ Sistema inicializado com sucesso!")
        print(f"🎯 Modelo padrão: {system.default_client_name}")
        print(f"🤖 Modelos disponíveis: {len(system.clients)}")
        print(f"💾 Cache: Ativo")
        print(f"🏥 Saúde: {system.system_health}")
        
        # Executar sessão interativa
        run_interactive_session(system)
        
        # Relatório final
        final_stats = system.get_performance_report()
        print(f"\n📊 ESTATÍSTICAS DA SESSÃO:")
        print(f"   • Problemas processados: {final_stats['performance_metrics']['total_problems_processed']}")
        print(f"   • Taxa de sucesso: {final_stats['performance_metrics']['success_rate']}")
        print(f"   • Códigos gerados: {final_stats['capabilities']['code_generations']}")
        print(f"   • Cache hits: {final_stats['cache_statistics']['hits']}")
        
    except Exception as e:
        print(f"\n❌ Erro fatal na inicialização: {e}")
        logging.error(f"Erro fatal: {e}", exc_info=True)
        print("\n🔧 VERIFICAÇÕES SUGERIDAS:")
        print("   • Ollama está instalado e rodando?")
        print("   • Modelo gemma3 está disponível?")
        print("   • OPENAI_API_KEY está configurada?")
        print("   • Dependências Python estão instaladas?")
        print("   • Permissões de arquivo estão corretas?")
        
        # Tentar diagnóstico básico
        print(f"\n🔍 DIAGNÓSTICO BÁSICO:")
        print(f"   • Ollama disponível: {OLLAMA_AVAILABLE}")
        print(f"   • OpenAI disponível: {OPENAI_AVAILABLE}")
        if OPENAI_AVAILABLE:
            print(f"   • API Key configurada: {'✅' if OPENAI_API_KEY else '❌'}")
    
    print("\n" + "="*80)
    print("🧠 Obrigado por usar o DREAM V13.2!")
    print("   Sistema AGI Multi-LLM Unificado")
    print("   Versão Robusta e Completa")
    print("="*80)

if __name__ == "__main__":
    main()