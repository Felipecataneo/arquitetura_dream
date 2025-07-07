# ==============================================================================
#           SISTEMA DREAM V6.0: FRAMEWORK ADAPTATIVO MULTI-NÍVEL
#               Sistema completo para resolução de problemas com entrada dinâmica
#               do usuário e múltiplos níveis de complexidade
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

# --- DOMÍNIOS SUPORTADOS ---
SUPPORTED_DOMAINS = {
    "tower_of_hanoi": {
        "description": "Torre de Hanói - quebra-cabeça clássico",
        "step_schema": {"from_peg": "int (0, 1, ou 2)", "to_peg": "int (0, 1, ou 2)"},
        "required_keys": ["from_peg", "to_peg"],
        "complexity_calculator": lambda ctx: ComplexityLevel(min(5, max(3, math.ceil(ctx.get("num_disks", 3) / 2))))
    },
    "mathematical": {
        "description": "Problemas matemáticos diversos",
        "step_schema": {"operation": "string", "operands": "list", "result": "number"},
        "required_keys": ["operation", "result"],
        "complexity_calculator": lambda ctx: ComplexityLevel.BASIC if "equation" in str(ctx).lower() else ComplexityLevel.TRIVIAL
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

# --- MÓDULO DE MEMÓRIA EXPANDIDO ---
class AgentMemory:
    def __init__(self):
        self._knowledge: Dict[str, List[str]] = {}
        self._performance_history: Dict[str, List[float]] = {}
        self._strategy_success: Dict[str, Dict[str, int]] = {}
    
    def add_learning(self, domain: str, learning: str):
        if domain not in self._knowledge: self._knowledge[domain] = []
        if learning not in self._knowledge[domain]:
            self._knowledge[domain].append(learning); logging.info(f"MEMÓRIA: Aprendizado para '{domain}': {learning}")
    
    def get_learnings(self, domain: str) -> List[str]: return self._knowledge.get(domain, [])
    
    def record_performance(self, domain: str, strategy: str, success: bool):
        if domain not in self._performance_history: self._performance_history[domain] = []
        if domain not in self._strategy_success: self._strategy_success[domain] = {}
        if strategy not in self._strategy_success[domain]: self._strategy_success[domain][strategy] = 0
        if success: self._strategy_success[domain][strategy] += 1
        self._performance_history[domain].append(1.0 if success else 0.0)
    
    def get_best_strategy(self, domain: str) -> Optional[str]:
        if domain in self._strategy_success:
            strategies = self._strategy_success[domain]
            if strategies: return max(strategies.items(), key=lambda x: x[1])[0]
        return None

# --- DETECTOR DE DOMÍNIO E COMPLEXIDADE ---
class DomainComplexityAnalyzer:
    def __init__(self):
        self.domain_keywords = {
            "tower_of_hanoi": ["torre", "hanoi", "disco", "pino", "vara", "tower"],
            "mathematical": ["equação", "resolver", "calcular", "x =", "y =", "z =", "+", "-", "*", "/"],
            "logical": ["se", "então", "logo", "portanto", "premissa", "conclusão"],
            "algorithmic": ["algoritmo", "ordenar", "buscar", "percorrer", "implementar"]
        }
    
    def analyze_problem(self, problem: str) -> Dict:
        problem_lower = problem.lower()
        domain_scores = {domain: sum(1 for kw in kws if kw in problem_lower) for domain, kws in self.domain_keywords.items()}
        detected_domain = max(domain_scores.items(), key=lambda x: x[1])[0] if any(domain_scores.values()) else "general"
        context = self._extract_context(problem, detected_domain)
        complexity_level = SUPPORTED_DOMAINS[detected_domain]["complexity_calculator"](context)
        return {"domain": detected_domain, "context": context, "complexity_level": complexity_level}
    
    def _extract_context(self, problem: str, domain: str) -> Dict:
        context = {}
        if domain == "tower_of_hanoi":
            match = re.search(r'(\d+)\s*discos?', problem.lower()); context["num_disks"] = int(match.group(1)) if match else 3
        elif domain == "mathematical":
            match = re.search(r'([a-zA-Z])\s*([\+\-])\s*(\d+)\s*=\s*(\d+)', problem)
            if match: context["equation"] = match.group(0); context["variable"] = match.group(1)
        return context

# --- EXECUTOR SIMBÓLICO EXPANDIDO ---
class SymbolicExecutor:
    def __init__(self, domain: str):
        self.domain, self.state, self.trace, self.error_context = domain, None, [], {}
    
    def execute_plan(self, initial_state: Dict, plan: List[Dict]) -> Dict:
        # ... (Implementation from user's code is good, no changes needed here) ...
        self.trace = []
        if self.domain == "tower_of_hanoi": return self._execute_hanoi_plan(initial_state, plan)
        return {"success": False, "error": f"Domínio '{self.domain}' não suportado pelo executor"}

    def _execute_hanoi_plan(self, initial_state: Dict, plan: List[Dict]) -> Dict:
        num_disks = initial_state.get("num_disks", 3)
        self.state = [list(range(num_disks, 0, -1)), [], []]
        self.trace.append(f"EXECUTOR: Estado inicial: {self.state}")
        for i, step in enumerate(plan):
            if not isinstance(step, dict):
                return self._format_error(i, step, "invalid_format", "Passo não é um dicionário.")
            if not self._validate_step_schema(step):
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

# --- GERADOR DE SOLUÇÕES ALGORÍTMICAS EXPANDIDO ---
class AlgorithmicSolutionGenerator:
    def generate_solution(self, domain: str, context: Dict) -> List[Dict]:
        if domain == "tower_of_hanoi": return self.generate_hanoi_solution(context.get("num_disks", 3))
        return []
    
    def generate_hanoi_solution(self, num_disks: int, source: int = 0, target: int = 2, auxiliary: int = 1) -> List[Dict]:
        if num_disks <= 0: return []
        moves = []
        if num_disks > 1: moves.extend(self.generate_hanoi_solution(num_disks - 1, source, auxiliary, target))
        moves.append({"from_peg": source, "to_peg": target})
        if num_disks > 1: moves.extend(self.generate_hanoi_solution(num_disks - 1, auxiliary, target, source))
        return moves

# --- SISTEMA DREAM PRINCIPAL ---
class DreamSystem:
    def __init__(self, ollama_model: str = 'gemma3', max_correction_attempts: int = 2):
        self.ollama_model, self.max_correction_attempts = ollama_model, max_correction_attempts
        self.memory, self.domain_analyzer, self.algorithmic_generator = AgentMemory(), DomainComplexityAnalyzer(), AlgorithmicSolutionGenerator()
        self.symbolic_rules = {"mathematical": {"extract_linear_equation": r'(?i)([a-zA-Z])\s*([\+\-])\s*(\d+)\s*=\s*(\d+)'}}
        self.ollama_available = OLLAMA_AVAILABLE and self._check_ollama_connection()
        self.meta_state = MetaCognitionState()
    
    def _check_ollama_connection(self):
        if not OLLAMA_AVAILABLE: logging.warning("Ollama não instalado. Modo simulação."); return False
        try: logging.info(f"Verificando Ollama e modelo '{self.ollama_model}'..."); ollama.show(self.ollama_model); logging.info("Ollama e modelo OK."); return True
        except Exception as e: logging.warning(f"Ollama não disponível: {e}. Modo simulação."); return False
    
    def solve_problem(self, problem: str) -> Dict:
        start_time = time.time()
        self.meta_state = MetaCognitionState()
        analysis = self.domain_analyzer.analyze_problem(problem)
        self.meta_state.domain, self.meta_state.complexity_level = analysis["domain"], analysis["complexity_level"]
        self.meta_state.reasoning_trace.extend([f"ANÁLISE: Domínio: {analysis['domain']}", f"ANÁLISE: Complexidade: {analysis['complexity_level'].name}", f"ANÁLISE: Contexto: {analysis['context']}"])
        
        initial_strategy = self._choose_strategy_by_complexity(analysis["complexity_level"])
        self.meta_state.current_strategy = initial_strategy
        self.meta_state.attempted_strategies.append(initial_strategy)
        
        context = {"domain": analysis["domain"], "initial_state": analysis["context"]}
        solution = self._execute_strategy(problem, initial_strategy, context)
        
        if self._is_failure(solution) and len(self.meta_state.attempted_strategies) < 3:
            self.meta_state.reasoning_trace.append("META: Falha detectada. Adaptando estratégia...")
            self.meta_state.adaptation_count += 1
            alt_strategy = self._choose_alternative_strategy()
            if alt_strategy:
                self.meta_state.current_strategy, self.meta_state.attempted_strategies = alt_strategy, self.meta_state.attempted_strategies + [alt_strategy]
                solution = self._execute_strategy(problem, alt_strategy, context)
        
        success = not self._is_failure(solution)
        self.memory.record_performance(analysis["domain"], self.meta_state.current_strategy.name, success)
        self.meta_state.decision_time, self.meta_state.success = time.time() - start_time, success
        return {"solution": solution, "meta_state": self.meta_state, "analysis": analysis, "success": success}
    
    def _choose_strategy_by_complexity(self, cl: ComplexityLevel) -> ReasoningStrategy:
        if cl == ComplexityLevel.TRIVIAL: return ReasoningStrategy.NEURAL_INTUITIVE
        if cl == ComplexityLevel.BASIC: return ReasoningStrategy.SYMBOLIC_MONOLITHIC
        if cl == ComplexityLevel.INTERMEDIATE: return ReasoningStrategy.DECOMPOSITIONAL_EXECUTOR
        if cl == ComplexityLevel.ADVANCED: return ReasoningStrategy.HYBRID_ALGORITHMIC
        return ReasoningStrategy.ENSEMBLE_REASONING # EXPERT

    def _choose_alternative_strategy(self) -> Optional[ReasoningStrategy]:
        unused = [s for s in list(ReasoningStrategy) if s not in self.meta_state.attempted_strategies]
        if not unused: return None
        if ReasoningStrategy.HYBRID_ALGORITHMIC in unused: return ReasoningStrategy.HYBRID_ALGORITHMIC
        if ReasoningStrategy.ENSEMBLE_REASONING in unused: return ReasoningStrategy.ENSEMBLE_REASONING
        return unused[0]

    def _is_failure(self, solution) -> bool: return isinstance(solution, dict) and not solution.get("success", True)
    
    def _execute_strategy(self, problem: str, strategy: ReasoningStrategy, context: Dict):
        self.meta_state.reasoning_trace.append(f"ESTRATÉGIA: Executando {strategy.name}")
        if strategy == ReasoningStrategy.NEURAL_INTUITIVE: return self._neural_reasoning(problem)
        if strategy == ReasoningStrategy.SYMBOLIC_MONOLITHIC: return self._symbolic_reasoning(problem, context)
        if strategy == ReasoningStrategy.DECOMPOSITIONAL_EXECUTOR: return self._decompositional_reasoning(problem, context)
        if strategy == ReasoningStrategy.HYBRID_ALGORITHMIC: return self._hybrid_algorithmic_reasoning(problem, context)
        if strategy == ReasoningStrategy.ENSEMBLE_REASONING: return self._ensemble_reasoning(problem, context)
        return {"success": False, "error": "Estratégia não reconhecida"}

    def _neural_reasoning(self, problem: str):
        self.meta_state.reasoning_trace.append("NEURAL: Usando intuição rápida")
        if not self.ollama_available: return "Resposta simulada (Ollama indisponível)."
        try:
            response = ollama.chat(model=self.ollama_model, messages=[{'role': 'user', 'content': f"Responda: {problem}"}])
            self.meta_state.confidence = 0.75; return response['message']['content']
        except Exception as e: return {"success": False, "error": f"Falha neural: {e}"}

    def _symbolic_reasoning(self, problem: str, context: Dict):
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
        self.meta_state.reasoning_trace.append("DREAM: Iniciando ciclo de Decomposição-Execução-Análise-Memória")
        domain = context.get("domain", "unknown")
        for attempt in range(self.max_correction_attempts):
            self.meta_state.reasoning_trace.append(f"DREAM: Tentativa #{attempt + 1}")
            learnings = self.memory.get_learnings(domain)
            self.meta_state.reasoning_trace.append(f"DREAM: Consultando memória. Aprendizados: {learnings}")
            plan = self._decompose_problem_with_llm(problem, domain, learnings)
            if self._is_failure(plan):
                error_msg = plan.get('error', 'desconhecida'); self.meta_state.reasoning_trace.append(f"DREAM: Falha na decomposição. Erro: {error_msg}")
                new_learning = self._analyze_failure_detailed(f"Falha ao gerar plano: {error_msg}", {}); self.memory.add_learning(domain, new_learning["learning"]); continue
            self.meta_state.reasoning_trace.append(f"DREAM: Plano gerado com {len(plan)} passos.")
            executor = SymbolicExecutor(domain=domain)
            execution_result = executor.execute_plan(context.get("initial_state", {}), plan)
            self.meta_state.reasoning_trace.extend(execution_result.get("trace", []))
            if execution_result.get("success"):
                self.meta_state.reasoning_trace.append(f"DREAM: Sucesso na tentativa #{attempt + 1}"); self.meta_state.confidence = 1.0; return execution_result
            else:
                error_info = execution_result.get("error", "Erro desconhecido"); error_context = execution_result.get("error_context", {})
                self.meta_state.reasoning_trace.append(f"DREAM: Falha na execução. Erro: {error_info}")
                new_learning = self._analyze_failure_detailed(error_info, error_context); self.memory.add_learning(domain, new_learning["learning"])
        return {"success": False, "error": f"Falha após {self.max_correction_attempts} tentativas"}

    def _hybrid_algorithmic_reasoning(self, problem: str, context: Dict) -> Dict:
        self.meta_state.reasoning_trace.append("HYBRID: Gerando e executando solução algorítmica")
        domain = context.get("domain", "unknown")
        if domain not in SUPPORTED_DOMAINS: return {"success": False, "error": f"Domínio '{domain}' não suportado por esta estratégia"}
        algorithmic_plan = self.algorithmic_generator.generate_solution(domain, context.get("initial_state", {}))
        if not algorithmic_plan: return {"success": False, "error": "Não foi possível gerar um plano algorítmico."}
        self.meta_state.reasoning_trace.append(f"HYBRID: Plano algorítmico com {len(algorithmic_plan)} passos gerado.")
        executor = SymbolicExecutor(domain=domain)
        execution_result = executor.execute_plan(context.get("initial_state", {}), algorithmic_plan)
        self.meta_state.reasoning_trace.extend(execution_result.get("trace", []))
        if execution_result.get("success"): self.meta_state.confidence = 1.0
        return execution_result

    def _ensemble_reasoning(self, problem: str, context: Dict) -> Dict:
        self.meta_state.reasoning_trace.append("ENSEMBLE: Executando múltiplas estratégias para validação cruzada")
        results = {}
        # Tenta a abordagem DREAM
        results["dream"] = self._decompositional_reasoning(problem, context)
        if results["dream"].get("success"): return results["dream"] # Se DREAM funciona, é bom o suficiente
        # Tenta a abordagem algorítmica
        results["algorithmic"] = self._hybrid_algorithmic_reasoning(problem, context)
        if results["algorithmic"].get("success"): return results["algorithmic"]
        # Como último recurso, usa a abordagem neural
        results["neural"] = self._neural_reasoning(problem)
        return results["neural"] if not self._is_failure(results["neural"]) else {"success": False, "error": "Todas as estratégias avançadas falharam."}

    def _decompose_problem_with_llm(self, problem: str, domain: str, learnings: List[str] = None) -> Union[List[Dict], Dict]:
        if not self.ollama_available: return {"success": False, "error": "Ollama indisponível para decomposição."}
        schema_def = SUPPORTED_DOMAINS.get(domain, {})
        if not schema_def.get("step_schema"): return {"success": False, "error": f"Nenhum esquema de passo para o domínio '{domain}'."}
        schema_text = json.dumps(schema_def["step_schema"], indent=2)
        learnings_text = ""
        if learnings: formatted_learnings = "\n".join(f"- {L}" for L in learnings); learnings_text = f"**REGRAS APRENDIDAS (OBRIGATÓRIO SEGUIR):**\n{formatted_learnings}\n"
        prompt = f"Você é um planejador lógico. Gere um plano JSON estritamente de acordo com o esquema.\n\n**ESQUEMA DE AÇÃO OBRIGATÓRIO:**\nCada passo deve ser um dicionário com a estrutura:\n```json\n{schema_text}\n```\n\n{learnings_text}**TAREFA:**\nProblema: \"{problem}\"\n\nSua resposta deve conter APENAS o array JSON, nada mais."
        try:
            response = ollama.chat(model=self.ollama_model, messages=[{'role': 'user', 'content': prompt}])
            raw_content = response['message']['content'].strip()
            self.meta_state.reasoning_trace.append(f"DECOMPOSIÇÃO: Resposta crua do LLM: {raw_content}")
            match = re.search(r'\[.*\]', raw_content, re.DOTALL)
            if match: plan_str = match.group(0); return json.loads(re.sub(r',\s*([\}\]])', r'\1', plan_str))
            return {"success": False, "error": "Nenhum array JSON ([...]) encontrado na resposta."}
        except Exception as e: return {"success": False, "error": f"Erro ao processar JSON: {e}"}

    def _analyze_failure_detailed(self, error_info: str, error_context: Dict) -> Dict:
        self.meta_state.reasoning_trace.append(f"ANÁLISE DETALHADA: Erro: '{error_info}' Contexto: {error_context}")
        error_type = error_context.get("error_type", "unknown"); learning = "Um erro de execução ocorreu."
        if error_type == "schema_violation": learning = f"Erro de Esquema: As chaves {SUPPORTED_DOMAINS[self.meta_state.domain]['required_keys']} são obrigatórias e devem ter o tipo correto."
        elif error_type == "invalid_move": learning = f"Regra Violada: {error_context.get('validation_details', {}).get('reason', 'desconhecida')}"
        elif "Falha ao gerar plano" in error_info: learning = "O plano JSON deve ser uma lista de dicionários, seguindo o esquema."
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
    print(f"  - Nível de Complexidade: {analysis['complexity_level'].name} (Nível {analysis['complexity_level'].value})")
    print(f"  - Contexto Extraído: {analysis['context']}")

    print("\n[RESULTADO FINAL]")
    print(f"  - Status: {'✅ SUCESSO' if result['success'] else '❌ FALHA'}")
    if isinstance(solution, dict):
        if result['success']:
            if 'final_state' in solution: print(f"  - Estado Final: {solution['final_state']}")
            if 'solution' in solution: print(f"  - Solução: {solution['solution']}")
        else:
            print(f"  - Erro: {solution.get('error', 'Desconhecido')}")
    else:
        print(f"  - Resposta: {solution}")

    print("\n[ANÁLISE META-COGNITIVA]")
    print(f"  - Estratégia Final Utilizada: {meta_state.current_strategy.name}")
    print(f"  - Estratégias Tentadas: {[s.name for s in meta_state.attempted_strategies]}")
    print(f"  - Adaptações de Estratégia: {meta_state.adaptation_count}")
    print(f"  - Confiança na Solução: {meta_state.confidence:.2f}")
    print(f"  - Tempo Total de Decisão: {meta_state.decision_time:.3f} segundos")

    print("\n[TRACE DE RACIOCÍNIO (PROCESSO DE PENSAMENTO)]")
    for i, step in enumerate(meta_state.reasoning_trace, 1):
        print(f"  {i:2d}. {step}")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    print("="*50)
    print("🤖 Agente com Arquitetura Cognitiva Híbrida 🤖")
    print("="*50)
    print("Você pode me pedir para resolver diferentes tipos de problemas.")
    print("Exemplos:")
    print("  - Conceitual: O que é a teoria da relatividade?")
    print("  - Matemática: Resolva a equação x + 10 = 25")
    print("  - Planejamento: Resolva a torre de hanoi com 4 discos")
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