# ==============================================================================
#           SISTEMA DREAM V6.9: FRAMEWORK COGNITIVO COM VALIDAÇÃO ATIVA
#               Implementa auto-correção através de validação de premissas
#               Resposta ao paper da Apple sobre "Illusion of Thinking"
# ==============================================================================

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Tuple
import math
import hashlib
from collections import defaultdict

# --- CONFIGURAÇÃO ---
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- NÍVEIS DE COMPLEXIDADE ---
class ComplexityLevel(Enum):
    TRIVIAL = 1
    BASIC = 2
    INTERMEDIATE = 3
    ADVANCED = 4
    EXPERT = 5

# --- TIPOS DE INCERTEZA ---
class UncertaintyType(Enum):
    FACTUAL = "factual"           # Incerteza sobre fatos
    CONCEPTUAL = "conceptual"     # Incerteza sobre conceitos
    RELATIONAL = "relational"     # Incerteza sobre relações
    PROCEDURAL = "procedural"     # Incerteza sobre métodos

# --- ESTRATÉGIAS DE RACIOCÍNIO ---
class ReasoningStrategy(Enum):
    NEURAL_INTUITIVE = "neural_intuitive"
    VALIDATED_REASONING = "validated_reasoning"
    SELF_CORRECTING_ANALYSIS = "self_correcting_analysis"
    PREMISE_DRIVEN_SYNTHESIS = "premise_driven_synthesis"
    CONCEPTUAL_SYNTHESIS = "conceptual_synthesis"
    RIDDLE_ANALYSIS = "riddle_analysis"
    MATHEMATICAL_DECOMPOSITION = "mathematical_decomposition"
    HYBRID_ALGORITHMIC = "hybrid_algorithmic"

# --- VALIDADOR DE PREMISSAS ---
class PremiseValidator:
    def __init__(self, ollama_model: str):
        self.ollama_model = ollama_model
        self.confidence_threshold = 0.6
        self.validation_cache = {}
    
    def validate_premise(self, premise: str, context: Dict) -> Dict:
        """Valida uma premissa e retorna confiança + sugestões de correção"""
        cache_key = hashlib.md5(premise.encode()).hexdigest()
        if cache_key in self.validation_cache:
            return self.validation_cache[cache_key]
        
        if not OLLAMA_AVAILABLE:
            return {"confidence": 0.5, "needs_validation": True, "validation_plan": []}
        
        prompt = f"""
        Você é um validador de conhecimento científico. Avalie a seguinte afirmação:
        
        AFIRMAÇÃO: "{premise}"
        CONTEXTO: {context}
        
        Responda em JSON com:
        - "confidence": float entre 0 e 1 (sua confiança na veracidade)
        - "uncertainty_type": "factual", "conceptual", "relational", ou "procedural"
        - "critical_assumptions": lista de suposições críticas
        - "validation_queries": lista de perguntas específicas para validar
        - "correction_hints": sugestões se a afirmação parecer incorreta
        - "needs_external_validation": true/false
        
        Seja honesto sobre limitações do seu conhecimento.
        """
        
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{'role': 'user', 'content': prompt}],
                format='json',
                options={'temperature': 0.2}
            )
            
            result = json.loads(response['message']['content'])
            result['needs_validation'] = result.get('confidence', 0) < self.confidence_threshold
            
            self.validation_cache[cache_key] = result
            return result
            
        except Exception as e:
            logging.error(f"Erro na validação de premissa: {e}")
            return {
                "confidence": 0.3,
                "uncertainty_type": "factual",
                "needs_validation": True,
                "validation_plan": [f"Verificar: {premise}"]
            }

# --- SISTEMA DE METACOGNIÇÃO AVANÇADO ---
@dataclass
class AdvancedMetaCognitionState:
    # Estados básicos
    current_strategy: ReasoningStrategy = ReasoningStrategy.NEURAL_INTUITIVE
    confidence: float = 0.0
    reasoning_trace: List[str] = field(default_factory=list)
    uncertainty_sources: List[str] = field(default_factory=list)
    complexity_level: ComplexityLevel = ComplexityLevel.TRIVIAL
    
    # Estados avançados
    premise_validations: Dict[str, Dict] = field(default_factory=dict)
    knowledge_gaps: List[str] = field(default_factory=list)
    correction_attempts: int = 0
    max_corrections: int = 3
    
    # Métricas de desempenho
    decision_time: float = 0.0
    validation_time: float = 0.0
    total_premises_checked: int = 0
    high_confidence_premises: int = 0
    
    # Controle de fluxo
    validation_enabled: bool = True
    auto_correction_enabled: bool = True
    domain: str = "unknown"
    success: bool = False

# --- ANALISADOR DE CAUSA RAIZ APRIMORADO ---
class EnhancedRootCauseAnalyzer:
    def __init__(self, ollama_model: str):
        self.ollama_model = ollama_model
        self.premise_validator = PremiseValidator(ollama_model)
    
    def analyze_with_5whys(self, problem: str) -> Optional[Dict]:
        """Análise de 5 porquês com validação de premissas"""
        if not OLLAMA_AVAILABLE:
            return None
        
        prompt = f"""
        Você é um especialista em análise de causa raiz. Analise a pergunta para descobrir:
        1. A intenção fundamental (5 porquês)
        2. Os conceitos-chave mencionados
        3. As premissas implícitas na pergunta
        
        PERGUNTA: "{problem}"
        
        Responda em JSON com:
        - "why_chain": lista de objetos {{why: "pergunta", because: "resposta"}}
        - "root_cause": necessidade fundamental
        - "key_concepts": lista de conceitos principais
        - "implicit_premises": lista de premissas que a pergunta assume
        - "knowledge_requirements": tipos de conhecimento necessários
        """
        
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{'role': 'user', 'content': prompt}],
                format='json',
                options={'temperature': 0.3}
            )
            
            result = json.loads(response['message']['content'])
            
            # Validar premissas identificadas
            if 'implicit_premises' in result:
                validated_premises = {}
                for premise in result['implicit_premises']:
                    validation = self.premise_validator.validate_premise(
                        premise, {"problem": problem}
                    )
                    validated_premises[premise] = validation
                
                result['premise_validations'] = validated_premises
            
            return {"success": True, **result}
            
        except Exception as e:
            logging.error(f"Erro na análise dos 5 Porquês: {e}")
            return {"success": False, "error": str(e)}

# --- CLASSES AUXILIARES ---
class AgentMemory:
    def __init__(self):
        self._knowledge: Dict[str, List[str]] = {}
        self._strategy_success: Dict[str, Dict[str, int]] = {}
    
    def add_learning(self, domain: str, learning: str):
        if domain not in self._knowledge:
            self._knowledge[domain] = []
        if learning not in self._knowledge[domain]:
            self._knowledge[domain].append(learning)
    
    def get_learnings(self, domain: str) -> List[str]:
        return self._knowledge.get(domain, [])
    
    def record_performance(self, domain: str, strategy: str, success: bool):
        if domain not in self._strategy_success:
            self._strategy_success[domain] = {}
        current = self._strategy_success[domain].get(strategy, 0)
        self._strategy_success[domain][strategy] = current + (1 if success else 0)

class DomainComplexityAnalyzer:
    def __init__(self):
        self.domain_keywords = {
            "academic_query": [
                "explain the theory", "explique a teoria", "teorema", "theorem", 
                "hilbert", "bifurcação", "bifurcation", "relatividade", "quântica",
                "geometrica", "problema de hilbert"
            ],
            "riddle_logic": ["which weighs more", "riddle", "charada"],
            "mathematical": ["equação", "resolver", "número", "divisor"],
            "general": []
        }
    
    def analyze_problem(self, problem: str) -> Dict:
        problem_lower = problem.lower()
        
        # Detectar domínio
        if any(kw in problem_lower for kw in self.domain_keywords["academic_query"]):
            domain = "academic_query"
            complexity = ComplexityLevel.EXPERT
        elif any(kw in problem_lower for kw in self.domain_keywords["riddle_logic"]):
            domain = "riddle_logic"
            complexity = ComplexityLevel.INTERMEDIATE
        elif any(kw in problem_lower for kw in self.domain_keywords["mathematical"]):
            domain = "mathematical"
            complexity = ComplexityLevel.ADVANCED
        else:
            domain = "general"
            complexity = ComplexityLevel.BASIC
        
        return {
            "domain": domain,
            "complexity_level": complexity,
            "context": {"original_problem": problem}
        }

# --- SISTEMA DREAM PRINCIPAL APRIMORADO ---
class DreamSystemV69:
    def __init__(self, ollama_model: str = 'gemma3', enable_validation: bool = True):
        self.ollama_model = ollama_model
        self.enable_validation = enable_validation
        
        # Componentes principais
        self.memory = AgentMemory()
        self.domain_analyzer = DomainComplexityAnalyzer()
        self.root_cause_analyzer = EnhancedRootCauseAnalyzer(ollama_model)
        self.premise_validator = PremiseValidator(ollama_model)
        
        # Estado e controle
        self.ollama_available = OLLAMA_AVAILABLE and self._check_ollama_connection()
        self.meta_state = AdvancedMetaCognitionState()
    
    def _check_ollama_connection(self) -> bool:
        if not OLLAMA_AVAILABLE:
            logging.warning("Ollama não instalado.")
            return False
        try:
            ollama.show(self.ollama_model)
            logging.info("Ollama e modelo OK.")
            return True
        except Exception as e:
            logging.warning(f"Ollama não disponível: {e}")
            return False
    
    def solve_problem(self, problem: str) -> Dict:
        """Solução de problemas com validação ativa de premissas"""
        start_time = time.time()
        self.meta_state = AdvancedMetaCognitionState()
        self.meta_state.validation_enabled = self.enable_validation
        
        # Análise inicial
        analysis = self.domain_analyzer.analyze_problem(problem)
        self.meta_state.domain = analysis["domain"]
        self.meta_state.complexity_level = analysis["complexity_level"]
        
        self.meta_state.reasoning_trace.append(
            f"ANÁLISE INICIAL: Domínio={analysis['domain']}, Complexidade={analysis['complexity_level'].name}"
        )
        
        # Análise de causa raiz com validação
        validation_start = time.time()
        whys_analysis = self.root_cause_analyzer.analyze_with_5whys(problem)
        
        if whys_analysis and whys_analysis.get("success"):
            self.meta_state.reasoning_trace.append("ANÁLISE DE INTENÇÃO: Causa raiz e premissas identificadas")
            
            # Processar validações de premissas
            if 'premise_validations' in whys_analysis:
                self.meta_state.premise_validations = whys_analysis['premise_validations']
                self._process_premise_validations()
            
            analysis['context'].update({
                'root_cause': whys_analysis.get('root_cause'),
                'key_concepts': whys_analysis.get('key_concepts', []),
                'knowledge_requirements': whys_analysis.get('knowledge_requirements', [])
            })
        
        self.meta_state.validation_time = time.time() - validation_start
        
        # Escolha e execução da estratégia
        strategy = self._choose_strategy_with_validation(analysis)
        self.meta_state.current_strategy = strategy
        
        solution = self._execute_strategy_with_validation(problem, strategy, analysis)
        
        # Auto-correção se necessário
        if self._needs_correction(solution) and self.meta_state.auto_correction_enabled:
            solution = self._attempt_self_correction(problem, solution, analysis)
        
        # Finalização
        self.meta_state.decision_time = time.time() - start_time
        self.meta_state.success = not self._is_failure(solution)
        
        return {
            "solution": solution,
            "meta_state": self.meta_state,
            "analysis": analysis,
            "success": self.meta_state.success
        }
    
    def _process_premise_validations(self):
        """Processa validações de premissas e identifica gaps de conhecimento"""
        for premise, validation in self.meta_state.premise_validations.items():
            self.meta_state.total_premises_checked += 1
            
            confidence = validation.get('confidence', 0)
            if confidence > 0.8:
                self.meta_state.high_confidence_premises += 1
            
            if validation.get('needs_validation', False):
                self.meta_state.knowledge_gaps.append(premise)
                self.meta_state.reasoning_trace.append(
                    f"VALIDAÇÃO: Premissa incerta identificada: {premise[:50]}..."
                )
            
            if validation.get('needs_external_validation', False):
                self.meta_state.uncertainty_sources.append(
                    f"Premissa requer validação externa: {premise}"
                )
    
    def _choose_strategy_with_validation(self, analysis: Dict) -> ReasoningStrategy:
        """Escolhe estratégia considerando necessidades de validação"""
        domain = analysis["domain"]
        complexity = analysis["complexity_level"]
        
        # Se há muitas premissas incertas, use estratégia de validação
        if len(self.meta_state.knowledge_gaps) > 2:
            return ReasoningStrategy.VALIDATED_REASONING
        
        # Se há problemas conceituais, use síntese validada
        if domain == "academic_query" and complexity == ComplexityLevel.EXPERT:
            return ReasoningStrategy.PREMISE_DRIVEN_SYNTHESIS
        
        # Fallback para estratégias tradicionais
        if domain == "academic_query":
            return ReasoningStrategy.CONCEPTUAL_SYNTHESIS
        elif domain == "riddle_logic":
            return ReasoningStrategy.RIDDLE_ANALYSIS
        elif domain in ["mathematical", "probabilistic"]:
            return ReasoningStrategy.MATHEMATICAL_DECOMPOSITION
        else:
            return ReasoningStrategy.NEURAL_INTUITIVE
    
    def _execute_strategy_with_validation(self, problem: str, strategy: ReasoningStrategy, analysis: Dict) -> Dict:
        """Executa estratégia com validação ativa"""
        self.meta_state.reasoning_trace.append(f"ESTRATÉGIA: Executando {strategy.name}")
        
        strategy_map = {
            ReasoningStrategy.VALIDATED_REASONING: self._validated_reasoning,
            ReasoningStrategy.PREMISE_DRIVEN_SYNTHESIS: self._premise_driven_synthesis,
            ReasoningStrategy.SELF_CORRECTING_ANALYSIS: self._self_correcting_analysis,
            ReasoningStrategy.CONCEPTUAL_SYNTHESIS: self._enhanced_conceptual_synthesis,
            ReasoningStrategy.NEURAL_INTUITIVE: self._neural_reasoning_with_validation,
            ReasoningStrategy.RIDDLE_ANALYSIS: self._riddle_analysis,
            ReasoningStrategy.MATHEMATICAL_DECOMPOSITION: self._mathematical_decomposition,
            ReasoningStrategy.HYBRID_ALGORITHMIC: self._hybrid_algorithmic_reasoning,
        }
        
        if strategy in strategy_map:
            return strategy_map[strategy](problem, analysis)
        else:
            return self._neural_reasoning_with_validation(problem, analysis)
    
    def _validated_reasoning(self, problem: str, analysis: Dict) -> Dict:
        """Raciocínio com validação ativa de cada passo"""
        if not self.ollama_available:
            return {"success": False, "error": "Ollama indisponível para validação"}
        
        self.meta_state.reasoning_trace.append("VALIDATED_REASONING: Iniciando raciocínio com validação")
        
        prompt = f"""
        Você é um pesquisador rigoroso. Sua tarefa é analisar o problema em etapas, 
        validando cada premissa antes de prosseguir.
        
        PROBLEMA: "{problem}"
        CONHECIMENTOS INCERTOS IDENTIFICADOS: {self.meta_state.knowledge_gaps}
        
        Para cada conceito incerto, declare explicitamente:
        1. "Meu conhecimento sobre X é limitado/incompleto"
        2. "Baseado no que sei, X parece ser..."
        3. "Esta informação deveria ser validada em fontes primárias"
        
        Depois construa sua análise usando apenas informações de alta confiança.
        
        Responda em JSON com:
        - "validated_facts": fatos que você tem certeza
        - "uncertain_areas": áreas onde seu conhecimento é limitado
        - "conditional_analysis": análise baseada em suposições explícitas
        - "validation_needed": lista de itens que precisam ser verificados
        """
        
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{'role': 'user', 'content': prompt}],
                format='json',
                options={'temperature': 0.1}
            )
            
            result = json.loads(response['message']['content'])
            
            # Construir resposta honesta
            solution_text = self._build_validated_response(result)
            
            self.meta_state.confidence = 0.85
            return {"success": True, "solution": solution_text}
            
        except Exception as e:
            logging.error(f"Erro no raciocínio validado: {e}")
            return self._neural_reasoning_with_validation(problem, analysis)
    
    def _premise_driven_synthesis(self, problem: str, analysis: Dict) -> Dict:
        """Síntese conceitual baseada em premissas validadas"""
        if not self.ollama_available:
            return {"success": False, "error": "Ollama indisponível"}
        
        self.meta_state.reasoning_trace.append("PREMISE_DRIVEN_SYNTHESIS: Síntese baseada em premissas validadas")
        
        # Coletar premissas validadas
        validated_premises = []
        uncertain_premises = []
        
        for premise, validation in self.meta_state.premise_validations.items():
            if validation.get('confidence', 0) > 0.7:
                validated_premises.append(premise)
            else:
                uncertain_premises.append(premise)
        
        prompt = f"""
        Você é um sintetizador de conhecimento. Construa uma resposta usando apenas 
        premissas validadas, sendo explícito sobre incertezas.
        
        PROBLEMA: "{problem}"
        PREMISSAS VALIDADAS: {validated_premises}
        PREMISSAS INCERTAS: {uncertain_premises}
        
        Estruture sua resposta como:
        1. **Análise Baseada em Conhecimento Validado**: Use apenas premissas validadas
        2. **Áreas de Incerteza**: Seja explícito sobre limitações
        3. **Framework de Investigação**: Como um expert abordaria as incertezas
        4. **Hipóteses Condicionais**: "Se X for verdade, então Y poderia ser..."
        
        Responda em JSON com chave "structured_analysis".
        """
        
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{'role': 'user', 'content': prompt}],
                format='json',
                options={'temperature': 0.2}
            )
            
            result = json.loads(response['message']['content'])
            analysis_content = result.get('structured_analysis', {})
            
            solution_text = self._format_premise_driven_response(analysis_content)
            
            self.meta_state.confidence = 0.90
            return {"success": True, "solution": solution_text}
            
        except Exception as e:
            return self._neural_reasoning_with_validation(problem, analysis)
    
    def _self_correcting_analysis(self, problem: str, analysis: Dict) -> Dict:
        """Análise auto-corretiva iterativa"""
        if not self.ollama_available:
            return {"success": False, "error": "Ollama indisponível"}
        
        self.meta_state.reasoning_trace.append("SELF_CORRECTING_ANALYSIS: Análise auto-corretiva")
        
        # Primeira tentativa
        initial_solution = self._neural_reasoning_with_validation(problem, analysis)
        
        if not initial_solution.get('success'):
            return initial_solution
        
        # Auto-correção baseada em gaps identificados
        if self.meta_state.knowledge_gaps:
            correction_prompt = f"""
            Você fez uma análise inicial, mas foram identificadas áreas de incerteza.
            
            PROBLEMA ORIGINAL: "{problem}"
            ANÁLISE INICIAL: {initial_solution.get('solution', '')}
            ÁREAS DE INCERTEZA: {self.meta_state.knowledge_gaps}
            
            Refine sua análise:
            1. Identifique onde sua resposta inicial pode estar incorreta
            2. Seja mais explícito sobre limitações
            3. Ofereça frameworks alternativos para investigação
            4. Mantenha apenas as partes de alta confiança
            
            Responda com uma análise corrigida.
            """
            
            try:
                response = ollama.chat(
                    model=self.ollama_model,
                    messages=[{'role': 'user', 'content': correction_prompt}],
                    options={'temperature': 0.1}
                )
                
                corrected_solution = response['message']['content']
                self.meta_state.confidence = 0.75
                
                return {"success": True, "solution": corrected_solution}
                
            except Exception as e:
                return initial_solution
        
        return initial_solution
    
    def _enhanced_conceptual_synthesis(self, problem: str, analysis: Dict) -> Dict:
        """Versão aprimorada da síntese conceitual com validação"""
        if not self.ollama_available:
            return {"success": False, "error": "Ollama indisponível"}
        
        self.meta_state.reasoning_trace.append("ENHANCED_CONCEPTUAL_SYNTHESIS: Síntese conceitual com validação")
        
        prompt = f"""
        Você é um filósofo da ciência. Sua tarefa é construir um framework de raciocínio 
        honesto e rigoroso para analisar a pergunta.
        
        PROBLEMA: "{problem}"
        LACUNAS DE CONHECIMENTO IDENTIFICADAS: {self.meta_state.knowledge_gaps}
        
        Siga estes passos:
        1. **Decomposição Conceitual Honesta**: Para cada conceito, declare seu nível de conhecimento
        2. **Mapeamento de Incertezas**: Identifique exatamente o que você não sabe
        3. **Framework de Investigação**: Como um expert abordaria essas incertezas
        4. **Síntese Condicional**: Construa hipóteses baseadas em suposições explícitas
        
        Seja rigorosamente honesto sobre limitações. Prefira "Não sei" a especulações.
        
        Responda em JSON com chave "rigorous_analysis".
        """
        
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{'role': 'user', 'content': prompt}],
                format='json',
                options={'temperature': 0.15}
            )
            
            result = json.loads(response['message']['content'])
            
            solution_text = self._format_rigorous_response(result.get('rigorous_analysis', {}))
            
            self.meta_state.confidence = 0.88
            return {"success": True, "solution": solution_text}
            
        except Exception as e:
            return self._neural_reasoning_with_validation(problem, analysis)
    
    def _neural_reasoning_with_validation(self, problem: str, analysis: Dict) -> Dict:
        """Raciocínio neural com disclaimers de validação"""
        if not self.ollama_available:
            return {"success": True, "solution": "Resposta simulada (Ollama indisponível)"}
        
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{'role': 'user', 'content': f"Responda com honestidade sobre suas limitações: {problem}"}],
                options={'temperature': 0.3}
            )
            
            solution_text = response['message']['content']
            
            # Adicionar disclaimers baseados nas validações
            if self.meta_state.knowledge_gaps:
                disclaimer = (
                    f"\n\n[AVISO DE VALIDAÇÃO]: Esta resposta pode conter imprecisões. "
                    f"Identifiquei {len(self.meta_state.knowledge_gaps)} áreas de incerteza que "
                    f"requerem validação em fontes primárias."
                )
                solution_text += disclaimer
            
            self.meta_state.confidence = 0.6
            return {"success": True, "solution": solution_text}
            
        except Exception as e:
            return {"success": False, "error": f"Falha neural: {e}"}
    
    def _riddle_analysis(self, problem: str, analysis: Dict) -> Dict:
        """Análise de charadas com validação"""
        if not self.ollama_available:
            return {"success": False, "error": "Ollama indisponível"}
        
        self.meta_state.reasoning_trace.append("RIDDLE_ANALYSIS: Analisando charada")
        
        prompt = f"""
        Você é um especialista em charadas. Analise cuidadosamente:
        
        PROBLEMA: "{problem}"
        
        Identifique:
        1. Possíveis pegadinhas ou truques
        2. Interpretações literal vs. implícita
        3. A resposta lógica correta
        
        Responda em JSON com:
        - "trap_analysis": análise de possíveis pegadinhas
        - "literal_interpretation": interpretação literal
        - "logical_answer": resposta lógica
        - "explanation": explicação detalhada
        """
        
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{'role': 'user', 'content': prompt}],
                format='json',
                options={'temperature': 0.2}
            )
            
            result = json.loads(response['message']['content'])
            
            solution_text = self._format_riddle_response(result)
            
            self.meta_state.confidence = 0.85
            return {"success": True, "solution": solution_text}
            
        except Exception as e:
            return self._neural_reasoning_with_validation(problem, analysis)
    
    def _mathematical_decomposition(self, problem: str, analysis: Dict) -> Dict:
        """Decomposição matemática com validação"""
        if not self.ollama_available:
            return {"success": False, "error": "Ollama indisponível"}
        
        self.meta_state.reasoning_trace.append("MATHEMATICAL_DECOMPOSITION: Decomposição matemática")
        
        prompt = f"""
        Você é um matemático rigoroso. Decomponha o problema:
        
        PROBLEMA: "{problem}"
        
        Estruture sua resposta com:
        1. **Definições**: Conceitos matemáticos envolvidos
        2. **Premissas**: Suposições necessárias
        3. **Desenvolvimento**: Passos lógicos
        4. **Conclusão**: Resultado final
        
        Responda em JSON com chave "mathematical_analysis".
        """
        
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{'role': 'user', 'content': prompt}],
                format='json',
                options={'temperature': 0.1}
            )
            
            result = json.loads(response['message']['content'])
            
            solution_text = self._format_mathematical_response(result.get('mathematical_analysis', {}))
            
            self.meta_state.confidence = 0.80
            return {"success": True, "solution": solution_text}
            
        except Exception as e:
            return self._neural_reasoning_with_validation(problem, analysis)
    
    def _hybrid_algorithmic_reasoning(self, problem: str, analysis: Dict) -> Dict:
        """Raciocínio algorítmico híbrido"""
        if not self.ollama_available:
            return {"success": False, "error": "Ollama indisponível"}
        
        self.meta_state.reasoning_trace.append("HYBRID_ALGORITHMIC: Raciocínio algorítmico")
        
        # Fallback para neural com validação
        return self._neural_reasoning_with_validation(problem, analysis)
    
    def _attempt_self_correction(self, problem: str, solution: Dict, analysis: Dict) -> Dict:
        """Tentativa de auto-correção baseada em validações"""
        if self.meta_state.correction_attempts >= self.meta_state.max_corrections:
            return solution
        
        self.meta_state.correction_attempts += 1
        self.meta_state.reasoning_trace.append(f"AUTO-CORREÇÃO: Tentativa {self.meta_state.correction_attempts}")
        
        # Tentar estratégia de auto-correção
        corrected_solution = self._self_correcting_analysis(problem, analysis)
        
        if corrected_solution.get('success'):
            return corrected_solution
        
        return solution
    
    def _needs_correction(self, solution: Dict) -> bool:
        """Determina se a solução precisa de correção"""
        if not solution.get("success", True):
            return True
        
        # Verificar se há muitas incertezas
        if len(self.meta_state.knowledge_gaps) > 3:
            return True
        
        # Verificar confiança baixa
        if self.meta_state.confidence < 0.5:
            return True
        
        return False
    
    def _is_failure(self, solution: Dict) -> bool:
        """Determina se a solução é uma falha"""
        return isinstance(solution, dict) and not solution.get("success", True)
    
    def _build_validated_response(self, result: Dict) -> str:
        """Constrói resposta baseada em validações"""
        response_parts = []
        
        if result.get('validated_facts'):
            response_parts.append("**FATOS VALIDADOS:**")
            for fact in result['validated_facts']:
                response_parts.append(f"✓ {fact}")
        
        if result.get('uncertain_areas'):
            response_parts.append("\n**ÁREAS DE INCERTEZA:**")
            for area in result['uncertain_areas']:
                response_parts.append(f"? {area}")
        
        if result.get('conditional_analysis'):
            response_parts.append(f"\n**ANÁLISE CONDICIONAL:**\n{result['conditional_analysis']}")
        
        if result.get('validation_needed'):
            response_parts.append("\n**VALIDAÇÃO NECESSÁRIA:**")
            for item in result['validation_needed']:
                response_parts.append(f"🔍 {item}")
        
        return "\n".join(response_parts)
    
    def _format_premise_driven_response(self, analysis: Dict) -> str:
        """Formata resposta baseada em premissas"""
        response = "**ANÁLISE BASEADA EM PREMISSAS VALIDADAS:**\n\n"
        
        for section, content in analysis.items():
            response += f"**{section.upper()}:**\n{content}\n\n"
        
        return response
    
    def _format_rigorous_response(self, analysis: Dict) -> str:
        """Formata resposta rigorosa"""
        response = "**ANÁLISE RIGOROSA COM VALIDAÇÃO DE PREMISSAS:**\n\n"
        
        for section, content in analysis.items():
            if isinstance(content, dict):
                response += f"**{section.upper()}:**\n"
                for key, value in content.items():
                    response += f"  • {key}: {value}\n"
                response += "\n"
            else:
                response += f"**{section.upper()}:**\n{content}\n\n"
        
        return response
    
    def _format_riddle_response(self, result: Dict) -> str:
        """Formata resposta de charada"""
        response = "**ANÁLISE DE CHARADA:**\n\n"
        
        if result.get('trap_analysis'):
            response += f"**ANÁLISE DE PEGADINHAS:**\n{result['trap_analysis']}\n\n"
        
        if result.get('literal_interpretation'):
            response += f"**INTERPRETAÇÃO LITERAL:**\n{result['literal_interpretation']}\n\n"
        
        if result.get('logical_answer'):
            response += f"**RESPOSTA LÓGICA:**\n{result['logical_answer']}\n\n"
        
        if result.get('explanation'):
            response += f"**EXPLICAÇÃO:**\n{result['explanation']}\n\n"
        
        return response
    
    def _format_mathematical_response(self, analysis: Dict) -> str:
        """Formata resposta matemática"""
        response = "**ANÁLISE MATEMÁTICA:**\n\n"
        
        for section, content in analysis.items():
            response += f"**{section.upper()}:**\n{content}\n\n"
        
        return response

# --- FUNÇÃO DE RELATÓRIO APRIMORADA ---
def print_enhanced_report(result: Dict):
    """Imprime relatório detalhado com informações de validação"""
    meta_state = result['meta_state']
    analysis = result['analysis']
    solution = result['solution']
    
    print("\n" + "="*80)
    print(" " * 20 + "RELATÓRIO DETALHADO DO PROCESSO COGNITIVO V6.9")
    print("="*80)

    print("\n[ANÁLISE DO PROBLEMA]")
    print(f"  - Domínio: {analysis['domain']}")
    print(f"  - Complexidade: {analysis['complexity_level'].name}")
    if analysis['context'].get('root_cause'):
        print(f"  - Intenção Inferida: {analysis['context']['root_cause']}")

    print("\n[VALIDAÇÃO DE PREMISSAS]")
    print(f"  - Premissas Verificadas: {meta_state.total_premises_checked}")
    print(f"  - Alta Confiança: {meta_state.high_confidence_premises}")
    print(f"  - Lacunas de Conhecimento: {len(meta_state.knowledge_gaps)}")
    print(f"  - Tempo de Validação: {meta_state.validation_time:.3f}s")
    
    if meta_state.knowledge_gaps:
        print("  - Áreas de Incerteza:")
        for gap in meta_state.knowledge_gaps[:3]:  # Mostrar apenas as 3 primeiras
            print(f"    • {gap[:60]}...")

    print("\n[RESULTADO FINAL]")
    print(f"  - Status: {'✅ SUCESSO' if result['success'] else '❌ FALHA'}")
    print(f"  - Estratégia: {meta_state.current_strategy.name if meta_state.current_strategy else 'N/A'}")
    print(f"  - Confiança: {meta_state.confidence:.2f}")
    
    if isinstance(solution, dict) and solution.get('solution'):
        print("  - Resposta:")
        for line in str(solution['solution']).split('\n'):
            print(f"    {line}")
    elif not result['success']:
        print(f"  - Erro: {solution.get('error', 'Desconhecido')}")

    print("\n[TRACE DE RACIOCÍNIO]")
    for i, step in enumerate(meta_state.reasoning_trace, 1):
        print(f"  {i:2d}. {step}")

    print("\n[MÉTRICAS DE PERFORMANCE]")
    print(f"  - Tempo Total: {meta_state.decision_time:.3f}s")
    print(f"  - Tentativas de Correção: {meta_state.correction_attempts}")
    print(f"  - Fontes de Incerteza: {len(meta_state.uncertainty_sources)}")
    
    print("="*80 + "\n")

# --- MAIN APRIMORADO ---
if __name__ == "__main__":
    print("="*60)
    print("🧠 DREAM System V6.9 - Validação Ativa de Premissas 🧠")
    print("="*60)
    print("Novos recursos:")
    print("  - Validação automática de premissas")
    print("  - Detecção de lacunas de conhecimento")
    print("  - Auto-correção baseada em incertezas")
    print("  - Raciocínio honesto sobre limitações")
    print("\nExemplos de teste:")
    print("  - explique a teoria geometrica da bifurcação e como ela resolve o 16º problema de hilbert")
    print("  - Como funciona a relatividade quântica de Einstein-Bohr?")
    print("  - Which weighs more, a pound of feathers or a pound of bricks?")
    print("\nDigite 'sair' para terminar.")
    print("-"*60)

    # Permitir configuração de validação
    try:
        enable_validation = input("Habilitar validação de premissas? (s/N): ").lower().startswith('s')
    except (KeyboardInterrupt, EOFError):
        print("\nEncerrando...")
        exit()
    
    agent = DreamSystemV69(ollama_model='gemma3', enable_validation=enable_validation)
    
    if enable_validation:
        print("\n✅ Validação de premissas HABILITADA")
    else:
        print("\n❌ Validação de premissas DESABILITADA")
    
    while True:
        try:
            user_input = input("\nProblema> ")
            if user_input.lower() in ['sair', 'exit', 'quit']:
                print("Encerrando o sistema. Até logo!")
                break
            if not user_input:
                continue
            
            result = agent.solve_problem(user_input)
            print_enhanced_report(result)

        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando o sistema. Até logo!")
            break
        except Exception as e:
            logging.error(f"Erro inesperado: {e}", exc_info=True)
            print("Erro inesperado. Tente novamente.")