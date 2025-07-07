# ==============================================================================
#           SISTEMA DREAM V6.4: FRAMEWORK ADAPTATIVO GENERALISTA
#               COM ANÁLISE APRIMORADA DE DOMÍNIO E COMPLEXIDADE
# ==============================================================================

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import math

# --- CONFIGURAÇÃO ---
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- NÍVEIS DE COMPLEXIDADE ---
class ComplexityLevel(Enum):
    TRIVIAL = 1      # Problemas simples, resposta direta
    BASIC = 2        # Problemas básicos com 1-2 etapas
    INTERMEDIATE = 3 # Problemas com múltiplas etapas
    ADVANCED = 4     # Problemas complexos com subestruturas
    EXPERT = 5       # Problemas altamente complexos

# --- DOMÍNIOS SUPORTADOS (GENERALISTA) ---
SUPPORTED_DOMAINS = {
    "tower_of_hanoi": {
        "description": "Torre de Hanói - quebra-cabeça clássico",
        "step_schema": {"from_peg": "int (0, 1, ou 2)", "to_peg": "int (0, 1, ou 2)"},
        "required_keys": ["from_peg", "to_peg"],
        "complexity_calculator": lambda ctx: ComplexityLevel(min(5, max(3, math.ceil(ctx.get("num_disks", 3) / 2))))
    },
    "mathematical": {
        "description": "Problemas matemáticos, incluindo teoria dos números.",
        "step_schema": {"step_description": "string", "calculation": "string", "result": "any"},
        "required_keys": ["step_description", "result"],
        "complexity_calculator": lambda ctx: ComplexityLevel.INTERMEDIATE if ctx.get('is_number_theory') else (ComplexityLevel.BASIC if "equation" in str(ctx.get("equation", "")) else ComplexityLevel.TRIVIAL)
    },
    "logical": {
        "description": "Problemas de lógica e raciocínio",
        "step_schema": {"premise": "string", "conclusion": "string", "reasoning": "string"},
        "required_keys": ["premise", "conclusion"],
        "complexity_calculator": lambda ctx: ComplexityLevel.INTERMEDIATE
    },
    "algorithmic": {
        "description": "Problemas algorítmicos gerais",
        "step_schema": {"step": "string", "action": "string", "state": "dict"},
        "required_keys": ["step", "action"],
        "complexity_calculator": lambda ctx: ComplexityLevel.ADVANCED
    },
    "general": {
        "description": "Perguntas gerais ou conceituais",
        "step_schema": {}, "required_keys": [],
        "complexity_calculator": lambda ctx: ComplexityLevel.TRIVIAL
    }
}

# --- ESTRATÉGIAS DE RACIOCÍNIO ---
class ReasoningStrategy(Enum):
    NEURAL_INTUITIVE = "neural_intuitive"
    SYMBOLIC_MONOLITHIC = "symbolic_monolithic"
    DECOMPOSITIONAL_EXECUTOR = "decompositional_executor"
    HYBRID_ALGORITHMIC = "hybrid_algorithmic"
    ENSEMBLE_REASONING = "ensemble_reasoning"
    MATHEMATICAL_DECOMPOSITION = "mathematical_decomposition"

# --- ESTADO METACOGNITIVO ---
@dataclass
class MetaCognitionState:
    current_strategy: ReasoningStrategy = ReasoningStrategy.NEURAL_INTUITIVE
    confidence: float = 0.0
    reasoning_trace: List[str] = field(default_factory=list)
    uncertainty_sources: List[str] = field(default_factory=list)
    context_complexity: float = 0.0
    complexity_level: ComplexityLevel = ComplexityLevel.TRIVIAL
    attempted_strategies: List[ReasoningStrategy] = field(default_factory=list)
    decision_time: float = 0.0
    adaptation_count: int = 0
    domain: str = "unknown"
    success: bool = False
    detailed_analysis: Dict = field(default_factory=dict)

# --- ANALISADOR DE CAUSA RAIZ (5 PORQUÊS) ---
class RootCauseAnalyzer:
    def __init__(self, ollama_model: str):
        self.ollama_model = ollama_model

    def analyze_with_5whys(self, problem: str) -> Optional[Dict]:
        if not OLLAMA_AVAILABLE:
            logging.warning("Análise de 5 Porquês desativada: Ollama não está disponível.")
            return None
        prompt = self._generate_5whys_prompt(problem)
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{'role': 'user', 'content': prompt}],
                format='json',
                options={'temperature': 0.3}
            )
            parsed_response = json.loads(response['message']['content'])
            if 'why_chain' in parsed_response and 'root_cause' in parsed_response:
                return {"success": True, **parsed_response}
            return {"success": False, "error": "Resposta do LLM não contém as chaves esperadas."}
        except Exception as e:
            logging.error(f"Erro na análise dos 5 Porquês: {e}")
            return {"success": False, "error": str(e)}

    def _generate_5whys_prompt(self, problem: str) -> str:
        return f"""
            Você é um especialista em análise de causa raiz usando a técnica dos "5 Porquês".
            Seu objetivo é analisar a pergunta de um usuário para descobrir a intenção ou necessidade fundamental por trás dela.

            PERGUNTA DO USUÁRIO: "{problem}"

            Aplique o método dos 5 Porquês para esta pergunta. Comece com a pergunta original e pergunte "Por quê?" recursivamente de 3 a 5 vezes para inferir o objetivo final do usuário.

            Responda com um único objeto JSON contendo duas chaves:
            1. "why_chain": Uma lista de objetos, onde cada objeto tem uma chave "why" (a pergunta) e "because" (a resposta inferida).
            2. "root_cause": Uma string concisa descrevendo a necessidade fundamental do usuário.

            Se a pergunta for muito simples ou direta (ex: "Qual a capital da França?"), a causa raiz pode ser simplesmente "obter uma informação factual".
            Se a pergunta for mais complexa (ex: "Qual o menor número com 12 divisores?"), a causa raiz pode ser "entender a relação entre fatoração prima e o número de divisores".

            Forneça APENAS o objeto JSON.
        """

# --- MÓDULO DE MEMÓRIA ---
class AgentMemory:
    def __init__(self):
        self._knowledge: Dict[str, List[str]] = {}
        self._strategy_success: Dict[str, Dict[str, int]] = {}
    
    def add_learning(self, domain: str, learning: str):
        if domain not in self._knowledge: self._knowledge[domain] = []
        if learning not in self._knowledge[domain]:
            self._knowledge[domain].append(learning); logging.info(f"MEMÓRIA: Aprendizado para '{domain}': {learning}")
    
    def get_learnings(self, domain: str) -> List[str]: return self._knowledge.get(domain, [])
    
    def record_performance(self, domain: str, strategy: str, success: bool):
        if domain not in self._strategy_success: self._strategy_success[domain] = {}
        self._strategy_success[domain][strategy] = self._strategy_success[domain].get(strategy, 0) + (1 if success else 0)
    
    def get_best_strategy(self, domain: str) -> Optional[str]:
        if domain in self._strategy_success and self._strategy_success[domain]:
            return max(self._strategy_success[domain].items(), key=lambda x: x[1])[0]
        return None

# --- DETECTOR DE DOMÍNIO E COMPLEXIDADE ---
class DomainComplexityAnalyzer:
    def __init__(self):
        self.domain_keywords = {
            "mathematical": [
                "equação", "resolver", "calcular", "x =", "y =", "z =", "+", "-", "*", "/",
                "número", "inteiro", "divisor", "primo", "fator", "menor", "maior",
                "múltiplo", "mmc", "mdc"
            ],
            "tower_of_hanoi": ["torre", "hanoi", "disco", "pino", "vara", "tower"],
            "logical": ["se", "então", "logo", "portanto", "premissa", "conclusão"],
            "algorithmic": ["algoritmo", "ordenar", "buscar", "percorrer", "implementar"]
        }
    
    def analyze_problem(self, problem: str) -> Dict:
        problem_lower = problem.lower()
        domain_scores = {d: sum(1 for kw in kws if kw in problem_lower) for d, kws in self.domain_keywords.items()}
        
        if domain_scores.get("mathematical", 0) > 0:
            detected_domain = "mathematical"
        else:
            detected_domain = max(domain_scores, key=domain_scores.get) if any(domain_scores.values()) else "general"

        context = self._extract_context(problem, detected_domain)
        complexity_level = SUPPORTED_DOMAINS[detected_domain]["complexity_calculator"](context)
        
        return {"domain": detected_domain, "context": context, "complexity_level": complexity_level}
    
    def _extract_context(self, problem: str, domain: str) -> Dict:
        context = {"original_problem": problem}
        if domain == "mathematical":
            if any(kw in problem.lower() for kw in ["divisor", "primo", "fator", "múltiplo"]):
                context['is_number_theory'] = True
            
            match = re.search(r'([a-zA-Z])\s*([\+\-])\s*(\d+)\s*=\s*(\d+)', problem)
            if match:
                context["equation"] = match.group(0); context["variable"] = match.group(1)
        elif domain == "tower_of_hanoi":
            match = re.search(r'(\d+)\s*discos?', problem.lower()); context["num_disks"] = int(match.group(1)) if match else 3
        
        return context

# --- EXECUTOR SIMBÓLICO ---
class SymbolicExecutor:
    def __init__(self, domain: str):
        self.domain, self.state, self.trace, self.error_context = domain, None, [], {}
    
    def execute_plan(self, initial_state: Dict, plan: List[Dict]) -> Dict:
        if self.domain == "tower_of_hanoi": return self._execute_hanoi_plan(initial_state, plan)
        return {"success": False, "error": f"Domínio '{self.domain}' não suportado pelo executor simbólico"}

    def _execute_hanoi_plan(self, initial_state: Dict, plan: List[Dict]) -> Dict:
        num_disks = initial_state.get("num_disks", 3)
        self.state = [list(range(num_disks, 0, -1)), [], []]
        self.trace.append(f"EXECUTOR: Estado inicial: {self.state}")
        for i, step in enumerate(plan):
            if not isinstance(step, dict) or not self._validate_step_schema(step):
                return self._format_error(i, step, "schema_violation", "Esquema inválido.")
            move_from, move_to = step.get("from_peg"), step.get("to_peg")
            validation_result = self._validate_hanoi_move(move_from, move_to)
            if not validation_result["valid"]:
                return self._format_error(i, step, "invalid_move", validation_result['reason'], validation_result)
            disk = self.state[move_from].pop()
            self.state[move_to].append(disk)
            self.trace.append(f"EXECUTOR: Passo {i+1}: Mover disco {disk} de {move_from} para {move_to}. Novo estado: {self.state}")
        target_peg = self.state[2]
        if len(target_peg) == num_disks and sorted(target_peg, reverse=True) == target_peg:
            self.trace.append("EXECUTOR: SUCESSO!"); return {"success": True, "final_state": self.state, "trace": self.trace}
        return {"success": False, "error": f"Estado final incorreto: {self.state}", "trace": self.trace}

    def _format_error(self, i, step, err_type, reason, details=None):
        error_msg = f"Erro no passo {i+1}: {reason}"
        self.error_context = {"step_index": i, "step": step, "error_type": err_type, "validation_details": details or {}}
        return {"success": False, "error": error_msg, "trace": self.trace, "error_context": self.error_context}

    def _validate_step_schema(self, step: Dict) -> bool:
        return all(key in step for key in SUPPORTED_DOMAINS[self.domain]["required_keys"])

    def _validate_hanoi_move(self, from_peg_idx: Any, to_peg_idx: Any) -> Dict:
        if not isinstance(from_peg_idx, int) or not (0 <= from_peg_idx <= 2): return {"valid": False, "reason": f"Pino origem inválido: {from_peg_idx}"}
        if not isinstance(to_peg_idx, int) or not (0 <= to_peg_idx <= 2): return {"valid": False, "reason": f"Pino destino inválido: {to_peg_idx}"}
        if not self.state[from_peg_idx]: return {"valid": False, "reason": f"Pino origem vazio ({from_peg_idx})"}
        disk_to_move = self.state[from_peg_idx][-1]
        if self.state[to_peg_idx] and self.state[to_peg_idx][-1] < disk_to_move: return {"valid": False, "reason": f"Disco maior ({disk_to_move}) sobre menor ({self.state[to_peg_idx][-1]})"}
        return {"valid": True}

# --- GERADOR DE SOLUÇÕES ALGORÍTMICAS ---
class AlgorithmicSolutionGenerator:
    def generate_solution(self, domain: str, context: Dict) -> List[Dict]:
        if domain == "tower_of_hanoi": return self.generate_hanoi_solution(context.get("num_disks", 3))
        return []
    
    def generate_hanoi_solution(self, num_disks: int, source: int = 0, target: int = 2, auxiliary: int = 1) -> List[Dict]:
        moves: List[Dict] = []
        self._solve_hanoi_recursive(moves, num_disks, source, target, auxiliary)
        return moves
    
    def _solve_hanoi_recursive(self, moves: List[Dict], n: int, source: int, target: int, auxiliary: int):
        if n > 0:
            self._solve_hanoi_recursive(moves, n - 1, source, auxiliary, target)
            moves.append({"from_peg": source, "to_peg": target})
            self._solve_hanoi_recursive(moves, n - 1, auxiliary, target, source)

# --- SISTEMA DREAM PRINCIPAL ---
class DreamSystem:
    def __init__(self, ollama_model: str = 'gemma3', max_correction_attempts: int = 2):
        self.ollama_model, self.max_correction_attempts = ollama_model, max_correction_attempts
        self.memory = AgentMemory()
        self.domain_analyzer = DomainComplexityAnalyzer()
        self.algorithmic_generator = AlgorithmicSolutionGenerator()
        self.root_cause_analyzer = RootCauseAnalyzer(ollama_model)
        self.symbolic_rules = {"mathematical": {"extract_linear_equation": r'(?i)([a-zA-Z])\s*([\+\-])\s*(\d+)\s*=\s*(\d+)'}}
        self.ollama_available = OLLAMA_AVAILABLE and self._check_ollama_connection()
        self.meta_state = MetaCognitionState()
    
    def _check_ollama_connection(self):
        if not OLLAMA_AVAILABLE: logging.warning("Ollama não instalado."); return False
        try:
            logging.info(f"Verificando Ollama e modelo '{self.ollama_model}'..."); ollama.show(self.ollama_model); logging.info("Ollama e modelo OK."); return True
        except Exception as e:
            logging.warning(f"Ollama não disponível: {e}."); return False
    
    def solve_problem(self, problem: str) -> Dict:
        start_time = time.time()
        self.meta_state = MetaCognitionState()
        
        analysis = self.domain_analyzer.analyze_problem(problem)
        self.meta_state.domain = analysis["domain"]
        self.meta_state.reasoning_trace.extend([f"ANÁLISE INICIAL: Domínio: {analysis['domain']}", f"ANÁLISE INICIAL: Complexidade: {analysis['complexity_level'].name}"])
        
        five_whys_analysis = self.root_cause_analyzer.analyze_with_5whys(problem)
        if five_whys_analysis and five_whys_analysis.get("success"):
            self.meta_state.reasoning_trace.append("ANÁLISE DE INTENÇÃO: Causa raiz inferida.")
            self.meta_state.detailed_analysis['5whys'] = five_whys_analysis
            analysis['context']['root_cause'] = five_whys_analysis.get('root_cause')
            
            original_complexity = analysis['complexity_level']
            analysis['complexity_level'] = self._reevaluate_complexity(original_complexity, five_whys_analysis, analysis['context'])
            if analysis['complexity_level'] != original_complexity:
                self.meta_state.reasoning_trace.append(f"REAVALIAÇÃO: Complexidade ajustada de {original_complexity.name} para {analysis['complexity_level'].name}.")

        self.meta_state.complexity_level = analysis['complexity_level']

        initial_strategy = self._choose_strategy(analysis["complexity_level"], analysis["domain"])
        self.meta_state.current_strategy = initial_strategy
        self.meta_state.attempted_strategies.append(initial_strategy)
        
        context = {"domain": analysis["domain"], "initial_state": analysis["context"]}
        solution = self._execute_strategy(problem, initial_strategy, context)
        
        if self._is_failure(solution) and len(self.meta_state.attempted_strategies) < 3:
            self.meta_state.reasoning_trace.append("META: Falha detectada. Adaptando estratégia...")
            self.meta_state.adaptation_count += 1
            alt_strategy = self._choose_alternative_strategy()
            if alt_strategy:
                self.meta_state.current_strategy = alt_strategy
                self.meta_state.attempted_strategies.append(alt_strategy)
                solution = self._execute_strategy(problem, alt_strategy, context)
        
        success = not self._is_failure(solution)
        self.memory.record_performance(analysis["domain"], self.meta_state.current_strategy.name, success)
        self.meta_state.decision_time, self.meta_state.success = time.time() - start_time, success
        
        if success and five_whys_analysis and five_whys_analysis.get("success"):
             solution = self._enrich_solution(solution, five_whys_analysis.get('root_cause'))

        return {"solution": solution, "meta_state": self.meta_state, "analysis": analysis, "success": success}

    def _reevaluate_complexity(self, current_complexity: ComplexityLevel, whys_analysis: Dict, context: Dict) -> ComplexityLevel:
        root_cause = whys_analysis.get("root_cause", "").lower()
        complex_keywords = [
            "implementar", "código", "comparar", "otimizar", "complexidade", "estrutura de dados",
            "relação entre", "propriedade de", "princípio de", "fórmula para"
        ]
        
        if context.get('is_number_theory') or any(keyword in root_cause for keyword in complex_keywords):
            return ComplexityLevel(max(ComplexityLevel.INTERMEDIATE.value, current_complexity.value))

        return current_complexity

    def _enrich_solution(self, solution: Any, root_cause: Optional[str]) -> Any:
        if not root_cause: return solution
        note = f"\n\n[Nota de Intenção] Percebi que sua real necessidade pode ser '{root_cause}'. Espero que esta resposta ajude nesse objetivo."
        if isinstance(solution, str): return solution + note
        if isinstance(solution, dict) and 'solution' in solution and isinstance(solution['solution'], str):
            solution['solution'] += note
        return solution
        
    def _choose_strategy(self, cl: ComplexityLevel, domain: str) -> ReasoningStrategy:
        if domain == "mathematical" and cl.value >= ComplexityLevel.INTERMEDIATE.value:
            return ReasoningStrategy.MATHEMATICAL_DECOMPOSITION
        
        strategy_map = {
            ComplexityLevel.TRIVIAL: ReasoningStrategy.NEURAL_INTUITIVE,
            ComplexityLevel.BASIC: ReasoningStrategy.SYMBOLIC_MONOLITHIC,
            ComplexityLevel.INTERMEDIATE: ReasoningStrategy.DECOMPOSITIONAL_EXECUTOR,
            ComplexityLevel.ADVANCED: ReasoningStrategy.HYBRID_ALGORITHMIC,
            ComplexityLevel.EXPERT: ReasoningStrategy.ENSEMBLE_REASONING,
        }
        return strategy_map[cl]

    def _choose_alternative_strategy(self) -> Optional[ReasoningStrategy]:
        unused = [s for s in list(ReasoningStrategy) if s not in self.meta_state.attempted_strategies]
        if not unused: return None
        priority = [
            ReasoningStrategy.MATHEMATICAL_DECOMPOSITION,
            ReasoningStrategy.HYBRID_ALGORITHMIC, 
            ReasoningStrategy.ENSEMBLE_REASONING, 
            ReasoningStrategy.DECOMPOSITIONAL_EXECUTOR
        ]
        for p in priority:
            if p in unused: return p
        return unused[0]

    def _is_failure(self, solution) -> bool: return isinstance(solution, dict) and not solution.get("success", True)
    
    def _execute_strategy(self, problem: str, strategy: ReasoningStrategy, context: Dict):
        self.meta_state.reasoning_trace.append(f"ESTRATÉGIA: Executando {strategy.name}")
        strategy_map = {
            ReasoningStrategy.NEURAL_INTUITIVE: self._neural_reasoning,
            ReasoningStrategy.SYMBOLIC_MONOLITHIC: self._symbolic_reasoning,
            ReasoningStrategy.DECOMPOSITIONAL_EXECUTOR: self._decompositional_reasoning,
            ReasoningStrategy.HYBRID_ALGORITHMIC: self._hybrid_algorithmic_reasoning,
            ReasoningStrategy.ENSEMBLE_REASONING: self._ensemble_reasoning,
            ReasoningStrategy.MATHEMATICAL_DECOMPOSITION: self._mathematical_decomposition
        }
        if strategy in strategy_map:
            if strategy == ReasoningStrategy.NEURAL_INTUITIVE:
                 return strategy_map[strategy](problem)
            return strategy_map[strategy](problem, context)
        return {"success": False, "error": "Estratégia não reconhecida"}

    def _neural_reasoning(self, problem: str) -> Union[str, Dict]:
        self.meta_state.reasoning_trace.append("NEURAL: Usando intuição rápida")
        if not self.ollama_available: return {"success": True, "solution": "Resposta simulada (Ollama indisponível)."}
        try:
            response = ollama.chat(model=self.ollama_model, messages=[{'role': 'user', 'content': f"Responda concisamente: {problem}"}])
            self.meta_state.confidence = 0.75; return {"success": True, "solution": response['message']['content']}
        except Exception as e: return {"success": False, "error": f"Falha neural: {e}"}

    def _symbolic_reasoning(self, problem: str, context: Dict) -> Dict:
        self.meta_state.reasoning_trace.append("SYMBOLIC: Aplicando regras")
        domain = context.get("domain")
        if domain == "mathematical":
            match = re.search(self.symbolic_rules["mathematical"]["extract_linear_equation"], problem)
            if match:
                var, op, v1, v2 = match.groups(); result = int(v2) + int(v1) if op == '-' else int(v2) - int(v1)
                self.meta_state.reasoning_trace.append(f"SYMBOLIC: Regra aplicada. {var}={result}"); self.meta_state.confidence = 1.0
                return {"success": True, "solution": f"O valor de {var} é {result}."}
        return {"success": False, "error": "Nenhuma regra simbólica aplicável encontrada"}

    def _decompositional_reasoning(self, problem: str, context: Dict) -> Dict:
        self.meta_state.reasoning_trace.append("DREAM: Iniciando ciclo de Decomposição-Execução")
        domain = context.get("domain", "unknown")
        for attempt in range(self.max_correction_attempts):
            self.meta_state.reasoning_trace.append(f"DREAM: Tentativa #{attempt + 1}")
            learnings = self.memory.get_learnings(domain)
            plan = self._decompose_problem_with_llm(problem, domain, learnings)
            if self._is_failure(plan):
                new_learning = self._analyze_failure_detailed(plan.get('error', 'desconhecida'), {}); self.memory.add_learning(domain, new_learning["learning"]); continue
            executor = SymbolicExecutor(domain=domain)
            execution_result = executor.execute_plan(context.get("initial_state", {}), plan)
            if execution_result.get("success"):
                self.meta_state.confidence = 1.0; return execution_result
            else:
                new_learning = self._analyze_failure_detailed(execution_result.get("error", ""), execution_result.get("error_context", {})); self.memory.add_learning(domain, new_learning["learning"])
        return {"success": False, "error": f"Falha após {self.max_correction_attempts} tentativas"}

    def _hybrid_algorithmic_reasoning(self, problem: str, context: Dict) -> Dict:
        self.meta_state.reasoning_trace.append("HYBRID: Gerando e executando solução algorítmica")
        domain = context.get("domain", "unknown")
        algorithmic_plan = self.algorithmic_generator.generate_solution(domain, context.get("initial_state", {}))
        if not algorithmic_plan: return {"success": False, "error": "Não foi possível gerar um plano algorítmico."}
        executor = SymbolicExecutor(domain=domain)
        execution_result = executor.execute_plan(context.get("initial_state", {}), algorithmic_plan)
        if execution_result.get("success"):
            self.meta_state.confidence = 1.0
            execution_result['solution'] = f"Solução algorítmica executada com sucesso em {len(algorithmic_plan)} passos."
        return execution_result

    def _ensemble_reasoning(self, problem: str, context: Dict) -> Dict:
        self.meta_state.reasoning_trace.append("ENSEMBLE: Executando múltiplas estratégias")
        strategies_to_try = [self._decompositional_reasoning, self._hybrid_algorithmic_reasoning, self._neural_reasoning]
        for strategy_func in strategies_to_try:
            result = strategy_func(problem, context) if strategy_func != self._neural_reasoning else strategy_func(problem)
            if not self._is_failure(result): return result
        return {"success": False, "error": "Todas as estratégias avançadas falharam."}

    def _mathematical_decomposition(self, problem: str, context: Dict) -> Dict:
        self.meta_state.reasoning_trace.append("MATHEMATICAL_DECOMPOSITION: Decompondo problema matemático")
        if not self.ollama_available: return {"success": False, "error": "Ollama indisponível."}

        prompt = f"""
            Você é um professor de matemática especialista em teoria dos números.
            Resolva o seguinte problema passo a passo, explicando cada etapa do seu raciocínio de forma clara e lógica.
            O objetivo não é apenas dar a resposta, mas demonstrar o processo para encontrá-la.

            PROBLEMA: "{problem}"

            Forneça sua resposta em um formato JSON com uma chave "solution_steps", que é uma lista de strings. Cada string é uma etapa do raciocínio.
            A última etapa deve conter a resposta final de forma explícita.

            Exemplo de formato de saída para "menor número com 12 divisores":
            {{
                "solution_steps": [
                    "Passo 1: Entender a fórmula para o número de divisores (τ(n)). Se n = p₁^a₁ * p₂^a₂ * ..., então τ(n) = (a₁+1)(a₂+1)...",
                    "Passo 2: Encontrar as fatorações de 12. As fatorações são 12, 6*2, 4*3, e 3*2*2.",
                    "Passo 3: Para cada fatoração, construir o menor número possível usando os menores primos (2, 3, 5, ...).",
                    "Passo 4: Analisar o caso '12' -> expoente (11). O número é 2¹¹ = 2048.",
                    "Passo 5: Analisar o caso '6*2' -> expoentes (5, 1). O número é 2⁵ * 3¹ = 96.",
                    "Passo 6: Analisar o caso '4*3' -> expoentes (3, 2). O número é 2³ * 3² = 72.",
                    "Passo 7: Analisar o caso '3*2*2' -> expoentes (2, 1, 1). O número é 2² * 3¹ * 5¹ = 60.",
                    "Passo 8: Comparar os resultados (2048, 96, 72, 60) e encontrar o menor.",
                    "Conclusão: O menor número inteiro positivo com exatamente 12 divisores é 60."
                ]
            }}

            Agora, resolva o problema fornecido. Forneça APENAS o objeto JSON.
        """
        try:
            response = ollama.chat(
                model=self.ollama_model, messages=[{'role': 'user', 'content': prompt}],
                format='json', options={'temperature': 0.1}
            )
            parsed_response = json.loads(response['message']['content'])
            
            if 'solution_steps' in parsed_response and isinstance(parsed_response['solution_steps'], list):
                self.meta_state.confidence = 0.95
                solution_text = "\n".join(f"  - {step}" for step in parsed_response['solution_steps'])
                return {"success": True, "solution": f"Aqui está o raciocínio passo a passo para encontrar a solução:\n{solution_text}"}
            else:
                self.meta_state.reasoning_trace.append("MATH_DECOMP: Falha no formato, recorrendo à abordagem neural.")
                return self._neural_reasoning(problem)
        except Exception as e:
            self.meta_state.reasoning_trace.append(f"MATH_DECOMP: Erro - {e}. Recorrendo à abordagem neural.")
            return self._neural_reasoning(problem)

    def _decompose_problem_with_llm(self, problem: str, domain: str, learnings: List[str] = None) -> Union[List[Dict], Dict]:
        if not self.ollama_available: return {"success": False, "error": "Ollama indisponível"}
        schema_def = SUPPORTED_DOMAINS.get(domain, {})
        if not schema_def.get("step_schema"): return {"success": False, "error": f"Nenhum esquema para o domínio '{domain}'."}
        schema_text = json.dumps(schema_def["step_schema"], indent=2)
        learnings_text = f"**REGRAS APRENDIDAS:**\n{'\n'.join(f'- {L}' for L in learnings)}\n" if learnings else ""
        prompt = f"Gere um plano JSON estritamente de acordo com o esquema.\n\n**ESQUEMA OBRIGATÓRIO:**\n{schema_text}\n\n{learnings_text}**TAREFA:**\nProblema: \"{problem}\"\n\nSua resposta deve conter APENAS o array JSON."
        try:
            response = ollama.chat(model=self.ollama_model, messages=[{'role': 'user', 'content': prompt}])
            raw_content = response['message']['content'].strip()
            match = re.search(r'\[.*\]', raw_content, re.DOTALL)
            if match: return json.loads(match.group(0))
            return {"success": False, "error": "Nenhum array JSON encontrado."}
        except Exception as e: return {"success": False, "error": f"Erro ao processar JSON: {e}"}

    def _analyze_failure_detailed(self, error_info: str, error_context: Dict) -> Dict:
        error_type = error_context.get("error_type", "unknown"); learning = "Erro de execução."
        if error_type == "schema_violation": learning = f"Erro de Esquema: As chaves {SUPPORTED_DOMAINS[self.meta_state.domain]['required_keys']} são obrigatórias."
        elif error_type == "invalid_move": learning = f"Regra Violada: {error_context.get('validation_details', {}).get('reason', 'desconhecida')}"
        return {"learning": learning}

def print_report(result: Dict):
    meta_state = result['meta_state']
    analysis = result['analysis']
    solution = result['solution']
    
    print("\n" + "="*80)
    print(" " * 25 + "RELATÓRIO FINAL DO PROCESSO COGNITIVO")
    print("="*80)

    print("\n[ANÁLISE DO PROBLEMA]")
    print(f"  - Domínio Detectado: {analysis['domain']}")
    print(f"  - Complexidade Inicial: {analysis['complexity_level'].name}")
    if '5whys' in meta_state.detailed_analysis:
        print(f"  - Complexidade Final (pós-análise): {meta_state.complexity_level.name}")
        print(f"  - Intenção Inferida (Causa Raiz): {analysis['context'].get('root_cause', 'N/A')}")
    else:
        print(f"  - Contexto Extraído: {analysis['context']}")

    print("\n[RESULTADO FINAL]")
    print(f"  - Status: {'✅ SUCESSO' if result['success'] else '❌ FALHA'}")
    if isinstance(solution, dict):
        final_solution_text = solution.get('solution') or solution.get('final_state')
        if final_solution_text:
            print(f"  - Resposta:\n")
            for line in str(final_solution_text).split('\n'):
                print(f"    {line}")
        elif not result['success']:
             print(f"  - Erro: {solution.get('error', 'Desconhecido')}")
    else:
        print(f"  - Resposta: {solution}")

    print("\n[ANÁLISE META-COGNITIVA]")
    print(f"  - Estratégia Final Utilizada: {meta_state.current_strategy.name}")
    print(f"  - Confiança na Solução: {meta_state.confidence:.2f}")
    print(f"  - Tempo Total de Decisão: {meta_state.decision_time:.3f} segundos")

    print("\n[TRACE DE RACIOCÍNIO (PROCESSO DE PENSAMENTO)]")
    for i, step in enumerate(meta_state.reasoning_trace, 1):
        print(f"  {i:2d}. {step}")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    print("="*50)
    print("🤖 Agente com Arquitetura Cognitiva Híbrida V6.4 🤖")
    print("="*50)
    print("Eu analiso sua pergunta para entender sua real necessidade.")
    print("Exemplos:")
    print("  - Conceitual: o que é a teoria da relatividade?")
    print("  - Matemática Simples: Resolva a equação x + 10 = 25")
    print("  - Matemática Complexa: Qual é o menor número inteiro positivo que tem exatamente 12 divisores?")
    print("  - Planejamento: Resolva a torre de hanoi com 3 discos")
    print("  - Digite 'sair' para terminar.")
    print("-"*50)

    agent = DreamSystem(ollama_model='gemma3')

    while True:
        try:
            user_input = input("Problema> ")
            if user_input.lower() in ['sair', 'exit', 'quit']:
                print("Encerrando o sistema. Até logo!")
                break
            if not user_input:
                continue
            
            result = agent.solve_problem(user_input)
            print_report(result)

        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando o sistema. Até logo!")
            break
        except Exception as e:
            logging.error(f"Ocorreu um erro inesperado no loop principal: {e}", exc_info=True)
            print("Ocorreu um erro. Por favor, tente novamente.")