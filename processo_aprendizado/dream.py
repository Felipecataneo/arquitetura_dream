# main.py
# ==============================================================================
#           PROTÓTIPO V5.0: AGENTE ADAPTATIVO COM MÚLTIPLAS ESTRATÉGIAS
#               (Versão Final da Tese com Análise, Memória e Criatividade)
# ==============================================================================

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# --- CONFIGURAÇÃO ---
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- DEFINIÇÕES GLOBAIS DE ESQUEMAS ---
ACTION_SCHEMAS = {
    "tower_of_hanoi": {
        "description": "Um passo no quebra-cabeça da Torre de Hanói.",
        "step_schema": {"from_peg": "int (0, 1, ou 2)", "to_peg": "int (0, 1, ou 2)"},
        "required_keys": ["from_peg", "to_peg"]
    }
}

# --- ESTRUTURAS DE DADOS ---
class ReasoningStrategy(Enum):
    NEURAL_INTUITIVE = "neural_intuitive"
    SYMBOLIC_MONOLITHIC = "symbolic_monolithic"
    DECOMPOSITIONAL_EXECUTOR = "decompositional_executor"
    HYBRID_ALGORITHMIC = "hybrid_algorithmic"

@dataclass
class MetaCognitionState:
    current_strategy: ReasoningStrategy = ReasoningStrategy.NEURAL_INTUITIVE
    confidence: float = 0.0
    reasoning_trace: List[str] = field(default_factory=list)
    uncertainty_sources: List[str] = field(default_factory=list)
    context_complexity: float = 0.0
    attempted_strategies: List[ReasoningStrategy] = field(default_factory=list)
    decision_time: float = 0.0
    adaptation_count: int = 0

# --- MÓDULO DE MEMÓRIA APRIMORADO ---
class AgentMemory:
    def __init__(self):
        self._knowledge: Dict[str, List[str]] = {}
    
    def add_learning(self, domain: str, learning: str):
        if domain not in self._knowledge: 
            self._knowledge[domain] = []
        if learning not in self._knowledge[domain]:
            self._knowledge[domain].append(learning)
            logging.info(f"MEMÓRIA: Aprendizado para '{domain}': {learning}")
    
    def get_learnings(self, domain: str) -> List[str]: 
        return self._knowledge.get(domain, [])

# --- MÓDULO DE EXECUÇÃO SIMBÓLICA APRIMORADO ---
class SymbolicExecutor:
    def __init__(self, domain: str):
        self.domain = domain
        self.state = None
        self.trace = []
        self.error_context = {}
    
    def execute_plan(self, initial_state: Dict, plan: List[Dict]) -> Dict:
        if self.domain == "tower_of_hanoi": 
            return self._execute_hanoi_plan(initial_state, plan)
        return {"success": False, "error": "Dominio de execução desconhecido."}
    
    def _execute_hanoi_plan(self, initial_state: Dict, plan: List[Dict]) -> Dict:
        num_disks = initial_state.get("num_disks", 0)
        self.state = [list(range(num_disks, 0, -1)), [], []]
        self.trace.append(f"EXECUTOR: Estado inicial: {self.state}")
        
        for i, step in enumerate(plan):
            if not isinstance(step, dict):
                error_msg = f"Erro no passo {i+1}: Passo não é um dicionário."
                self.error_context = {"step_index": i, "step": step, "error_type": "invalid_format"}
                return {"success": False, "error": error_msg, "trace": self.trace, "error_context": self.error_context}
            
            if not self._validate_step_schema(step):
                error_msg = f"Erro no passo {i+1}: Esquema inválido."
                self.error_context = {"step_index": i, "step": step, "error_type": "schema_violation"}
                return {"success": False, "error": error_msg, "trace": self.trace, "error_context": self.error_context}
            
            move_from = step.get("from_peg")
            move_to = step.get("to_peg")
            
            validation_result = self._validate_hanoi_move(move_from, move_to)
            if not validation_result["valid"]:
                error_msg = f"Erro no passo {i+1}: {validation_result['reason']}."
                self.error_context = {"step_index": i, "step": step, "error_type": "invalid_move", "validation_details": validation_result}
                return {"success": False, "error": error_msg, "trace": self.trace, "error_context": self.error_context}
            
            # Fix: Use move_from instead of undefined from_peg_idx
            disk = self.state[move_from].pop()
            self.state[move_to].append(disk)
            self.trace.append(f"EXECUTOR: Passo {i+1}: Mover disco {disk} de {move_from} para {move_to}. Novo estado: {self.state}")
        
        target_peg = self.state[2]
        if len(target_peg) == num_disks and sorted(target_peg, reverse=True) == target_peg:
            self.trace.append("EXECUTOR: SUCESSO!")
            return {"success": True, "final_state": self.state, "trace": self.trace}
        else:
            error_msg = f"Estado final incorreto: {self.state}"
            return {"success": False, "error": error_msg, "trace": self.trace}
    
    def _validate_step_schema(self, step: Dict) -> bool: 
        return all(key in step for key in ACTION_SCHEMAS[self.domain]["required_keys"])
    
    def _validate_hanoi_move(self, from_peg_idx: Any, to_peg_idx: Any) -> Dict:
        if not isinstance(from_peg_idx, int) or not (0 <= from_peg_idx <= 2): 
            return {"valid": False, "reason": f"Pino origem inválido: {from_peg_idx}"}
        if not isinstance(to_peg_idx, int) or not (0 <= to_peg_idx <= 2): 
            return {"valid": False, "reason": f"Pino destino inválido: {to_peg_idx}"}
        if not self.state[from_peg_idx]: 
            return {"valid": False, "reason": f"Pino origem vazio ({from_peg_idx})"}
        
        disk_to_move = self.state[from_peg_idx][-1]
        if self.state[to_peg_idx] and self.state[to_peg_idx][-1] < disk_to_move: 
            return {"valid": False, "reason": f"Disco maior ({disk_to_move}) sobre menor ({self.state[to_peg_idx][-1]})"}
        
        return {"valid": True}

# --- GERADOR DE SOLUÇÕES ALGORÍTMICAS ---
class AlgorithmicSolutionGenerator:
    def generate_hanoi_solution(self, num_disks: int, source: int = 0, target: int = 2, auxiliary: int = 1) -> List[Dict]:
        if num_disks <= 0: 
            return []
        
        moves = []
        if num_disks > 1: 
            moves.extend(self.generate_hanoi_solution(num_disks - 1, source, auxiliary, target))
        
        moves.append({"from_peg": source, "to_peg": target})
        
        if num_disks > 1: 
            moves.extend(self.generate_hanoi_solution(num_disks - 1, auxiliary, target, source))
        
        return moves

# --- CLASSE PRINCIPAL DO AGENTE META-COGNITIVO ---
class NeuroSymbolicMetaCognition:
    def __init__(self, ollama_model: str = 'gemma3', max_correction_attempts: int = 2):
        self.ollama_model = ollama_model
        self.max_correction_attempts = max_correction_attempts
        self.memory = AgentMemory()
        self.algorithmic_generator = AlgorithmicSolutionGenerator()
        self.symbolic_rules = self._initialize_symbolic_rules()
        self.ollama_available = OLLAMA_AVAILABLE and self._check_ollama_connection()
        self.meta_state = MetaCognitionState()

    def _check_ollama_connection(self):
        if not OLLAMA_AVAILABLE: 
            logging.warning("Ollama não instalado. Modo simulação.")
            return False
        try: 
            logging.info(f"Verificando Ollama e modelo '{self.ollama_model}'...")
            ollama.show(self.ollama_model)
            logging.info("Ollama e modelo OK.")
            return True
        except Exception as e: 
            logging.warning(f"Ollama não disponível: {e}. Modo simulação.")
            return False

    def reset_state(self): 
        self.meta_state = MetaCognitionState()
        logging.info("Estado meta-cognitivo resetado.")
    
    def _initialize_symbolic_rules(self): 
        return {"mathematical": {"extract_linear_equation": r'(?i)([a-zA-Z])\s*([\+\-])\s*(\d+)\s*=\s*(\d+)'}}
    
    def _is_problem_highly_compositional(self, problem: str):
        if "torre de hanoi" in problem.lower(): 
            return 1.0
        return 0.0

    def _meta_analyze_problem(self, problem: str):
        self.meta_state.reasoning_trace.append("META: Iniciando análise.")
        is_technical = any(kw in problem.lower() for kw in ['resolver', 'equação'])
        is_compositional = self._is_problem_highly_compositional(problem)
        
        if is_compositional > 0.8: 
            self.meta_state.context_complexity = 1.0
        elif is_technical: 
            self.meta_state.context_complexity = 0.7
        else: 
            self.meta_state.context_complexity = 0.2
        
        self.meta_state.reasoning_trace.append(f"META: Complexidade percebida: {self.meta_state.context_complexity:.2f}")

    def _choose_initial_strategy(self) -> ReasoningStrategy:
        if self.meta_state.context_complexity > 0.8: 
            return ReasoningStrategy.DECOMPOSITIONAL_EXECUTOR
        if self.meta_state.context_complexity > 0.5: 
            return ReasoningStrategy.SYMBOLIC_MONOLITHIC
        return ReasoningStrategy.NEURAL_INTUITIVE

    def solve_problem(self, problem: str, context: Dict):
        start_time = time.time()
        self.reset_state()
        self._meta_analyze_problem(problem)
        initial_strategy = self._choose_initial_strategy()
        
        self.meta_state.current_strategy = initial_strategy
        self.meta_state.attempted_strategies.append(initial_strategy)
        solution = self._execute_strategy(problem, initial_strategy, context)
        
        is_failure = isinstance(solution, dict) and not solution.get("success", True)
        if is_failure and initial_strategy == ReasoningStrategy.DECOMPOSITIONAL_EXECUTOR:
            self.meta_state.reasoning_trace.append("META: Estratégia DREAM-S falhou. Adaptando para a estratégia HYBRID_ALGORITHMIC.")
            self.meta_state.adaptation_count += 1
            alternative_strategy = ReasoningStrategy.HYBRID_ALGORITHMIC
            self.meta_state.current_strategy = alternative_strategy
            self.meta_state.attempted_strategies.append(alternative_strategy)
            solution = self._execute_strategy(problem, alternative_strategy, context)

        self.meta_state.decision_time = time.time() - start_time
        return {"solution": solution, "meta_state": self.meta_state}

    def _execute_strategy(self, problem: str, strategy: ReasoningStrategy, context: Dict):
        if strategy == ReasoningStrategy.NEURAL_INTUITIVE: 
            return self._neural_reasoning(problem)
        elif strategy == ReasoningStrategy.SYMBOLIC_MONOLITHIC: 
            return self._symbolic_reasoning(problem)
        elif strategy == ReasoningStrategy.DECOMPOSITIONAL_EXECUTOR: 
            return self._decompositional_reasoning(problem, context)
        elif strategy == ReasoningStrategy.HYBRID_ALGORITHMIC: 
            return self._hybrid_algorithmic_reasoning(problem, context)
        return "Estratégia não reconhecida."
        
    def _neural_reasoning(self, problem: str):
        self.meta_state.reasoning_trace.append("NEURAL: Usando intuição rápida.")
        if not self.ollama_available: 
            self.meta_state.reasoning_trace.append("NEURAL: (Simulação)")
            return "Resposta simulada."
        try: 
            response = ollama.chat(model=self.ollama_model, messages=[{'role': 'user', 'content': f"Responda: {problem}"}])
            self.meta_state.confidence = 0.75
            return response['message']['content']
        except Exception as e: 
            return f"Falha neural: {e}"

    def _symbolic_reasoning(self, problem: str):
        self.meta_state.reasoning_trace.append("SYMBOLIC: Buscando regra.")
        match = re.search(self.symbolic_rules["mathematical"]["extract_linear_equation"], problem)
        if match: 
            variable, op, v1, v2 = match.groups()
            result = int(v2) + int(v1) if op == '-' else int(v2) - int(v1)
            self.meta_state.reasoning_trace.append(f"SYMBOLIC: Regra aplicada. {variable}={result}")
            self.meta_state.confidence = 1.0
            return f"O valor de {variable} é {result}."
        return "Nenhuma regra simbólica encontrada."
    
    def _hybrid_algorithmic_reasoning(self, problem: str, context: Dict) -> Dict:
        self.meta_state.reasoning_trace.append("HYBRID: Iniciando raciocínio algorítmico.")
        domain = context.get("domain", "unknown")
        
        if domain == "tower_of_hanoi":
            num_disks = context.get("initial_state", {}).get("num_disks", 3)
            self.meta_state.reasoning_trace.append(f"HYBRID: Gerando solução algorítmica para {num_disks} discos.")
            algorithmic_plan = self.algorithmic_generator.generate_hanoi_solution(num_disks)
            
            executor = SymbolicExecutor(domain=domain)
            execution_result = executor.execute_plan(context.get("initial_state", {}), algorithmic_plan)
            
            self.meta_state.reasoning_trace.extend(execution_result.get("trace", []))
            if execution_result.get("success"): 
                self.meta_state.confidence = 1.0
            return execution_result
        
        return {"success": False, "error": "Domínio não suportado pela estratégia híbrida."}

    def _decompositional_reasoning(self, problem: str, context: Dict) -> Dict:
        self.meta_state.reasoning_trace.append("DREAM-S: Iniciando ciclo com Ancoragem de Esquema.")
        domain = context.get("domain", "unknown")
        
        for attempt in range(self.max_correction_attempts):
            self.meta_state.reasoning_trace.append(f"DREAM-S: Ciclo de Raciocínio #{attempt + 1}")
            
            learnings = self.memory.get_learnings(domain)
            self.meta_state.reasoning_trace.append(f"DREAM-S (Decomposição): Consultando memória. Aprendizados: {learnings}")
            
            plan = self._decompose_problem_with_llm(problem, domain, learnings)
            if not plan or (isinstance(plan, dict) and "error" in plan):
                error_msg = plan.get('error', 'desconhecida') if isinstance(plan, dict) else "Plano vazio."
                self.meta_state.reasoning_trace.append(f"DREAM-S (Análise): FALHA DECOMPOSIÇÃO. Erro: {error_msg}")
                new_learning = self._analyze_failure(f"Falha ao gerar plano: {error_msg}")
                self.memory.add_learning(domain, new_learning)
                continue
            
            self.meta_state.reasoning_trace.append(f"DREAM-S (Decomposição): Plano gerado com {len(plan)} passos.")
            
            executor = SymbolicExecutor(domain=domain)
            execution_result = executor.execute_plan(context.get("initial_state", {}), plan)
            self.meta_state.reasoning_trace.extend(execution_result.get("trace", []))
            
            if execution_result.get("success"):
                self.meta_state.reasoning_trace.append(f"DREAM-S: SUCESSO no ciclo #{attempt + 1}.")
                self.meta_state.confidence = 1.0
                return execution_result
            else:
                error_info = execution_result.get("error", "Erro desconhecido.")
                error_context = execution_result.get("error_context", {})
                self.meta_state.reasoning_trace.append(f"DREAM-S (Análise): FALHA EXECUÇÃO. Erro: {error_info}")
                new_learning = self._analyze_failure_detailed(error_info, error_context)
                self.memory.add_learning(domain, new_learning["learning"])
        
        final_error_message = f"Falha após {self.max_correction_attempts} tentativas."
        self.meta_state.reasoning_trace.append(f"DREAM-S: {final_error_message}")
        self.meta_state.confidence = 0.0
        return {"success": False, "error": final_error_message}

    def _analyze_failure(self, error_msg: str) -> str:
        """Simple failure analysis for basic errors"""
        return f"Erro encontrado: {error_msg}"

    def _analyze_failure_detailed(self, error_info: str, error_context: Dict) -> Dict:
        self.meta_state.reasoning_trace.append(f"ANÁLISE DETALHADA: Erro: '{error_info}' Contexto: {error_context}")
        error_type = error_context.get("error_type", "unknown")
        learning = "Um erro de execução ocorreu."
        
        if error_type == "schema_violation": 
            learning = f"Erro de Esquema: As chaves {ACTION_SCHEMAS['tower_of_hanoi']['required_keys']} são obrigatórias."
        elif error_type == "invalid_move": 
            learning = f"Regra Violada: {error_context.get('validation_details', {}).get('reason', 'desconhecida')}"
        
        return {"learning": learning}
        
    def _decompose_problem_with_llm(self, problem: str, domain: str, learnings: List[str] = None) -> List[Dict]:
        if not OLLAMA_AVAILABLE:
            self.meta_state.reasoning_trace.append("DECOMPOSIÇÃO (Simulação)")
            if domain == "tower_of_hanoi":
                if learnings and any("Regra Violada" in l for l in learnings): 
                    return self.algorithmic_generator.generate_hanoi_solution(3)
                else: 
                    return [{"from_peg": "a", "to_peg": "c"}]  # Simula erro de esquema
            return {"error": "Simulação não implementada."}
            
        schema_definition = ACTION_SCHEMAS.get(domain)
        if not schema_definition: 
            return {"error": f"Nenhum esquema definido para o domínio '{domain}'."}
        
        schema_text = json.dumps(schema_definition["step_schema"], indent=2)
        learnings_text = ""
        if learnings: 
            formatted_learnings = "\n".join(f"- {L}" for L in learnings)
            learnings_text = f"**REGRAS APRENDIDAS (OBRIGATÓRIO SEGUIR):**\n{formatted_learnings}\n"

        prompt = f"""Você é um planejador lógico. Gere um plano JSON estritamente de acordo com o esquema.

**ESQUEMA DE AÇÃO OBRIGATÓRIO:**
Cada passo deve ser um dicionário com a estrutura:
```json
{schema_text}
```

{learnings_text}**TAREFA:**
Problema: "{problem}"

Sua resposta deve conter APENAS o array JSON, nada mais."""
        
        try:
            response = ollama.chat(model=self.ollama_model, messages=[{'role': 'user', 'content': prompt}])
            raw_content = response['message']['content'].strip()
            self.meta_state.reasoning_trace.append(f"DECOMPOSIÇÃO: Resposta crua do LLM: {raw_content}")
            
            match = re.search(r'\[.*\]', raw_content, re.DOTALL)
            if match: 
                plan_str = match.group(0)
                plan_str = re.sub(r',\s*([\}\]])', r'\1', plan_str)
                return json.loads(plan_str)
            else: 
                return {"error": "Nenhum array JSON ([...]) encontrado na resposta."}
        except Exception as e: 
            return {"error": f"Erro ao processar JSON: {e}"}

# --- FUNÇÕES DE TESTE ---
def run_simulation(system: NeuroSymbolicMetaCognition, problem_statement: str, context_info: Dict):
    print("="*80)
    print(f"PROBLEMA: '{problem_statement}'")
    print(f"CONTEXTO: {context_info}")
    print("-"*80)
    
    result = system.solve_problem(problem_statement, context_info)
    solution, meta_state = result['solution'], result['meta_state']
    
    print("\n=== RESULTADO FINAL ===")
    if isinstance(solution, dict): 
        print(f"Sucesso: {solution.get('success')}")
        if not solution.get('success'): 
            print(f"Erro: {solution.get('error')}")
        else: 
            print(f"Estado Final: {solution.get('final_state')}")
    else: 
        print(f"Solução: {solution}")
    
    print(f"Estratégia Final: {meta_state.current_strategy.name}")
    print(f"Confiança: {meta_state.confidence:.2f}")
    print(f"Adaptações: {meta_state.adaptation_count}")
    
    print("\n=== TRACE DE RACIOCÍNIO META-COGNITIVO ===")
    for i, step in enumerate(meta_state.reasoning_trace, 1): 
        print(f"{i:2d}. {step}")
    print("="*80 + "\n")

def run_comprehensive_test():
    agent = NeuroSymbolicMetaCognition(ollama_model='gemma3', max_correction_attempts=2)
    
    test_scenarios = [
        {
            "name": "Cenário 1: Conceitual", 
            "problem": "O que é entropia?", 
            "context": {}
        },
        {
            "name": "Cenário 2: Simbólico", 
            "problem": "Resolver: y - 15 = 30", 
            "context": {}
        },
        {
            "name": "Cenário 3: Complexo (DREAM-S com fallback para Híbrido)", 
            "problem": "Por favor, resolva o quebra-cabeça da torre de hanoi com 3 discos.", 
            "context": {"domain": "tower_of_hanoi", "initial_state": {"num_disks": 3}}
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n{'='*20} {scenario['name']} {'='*20}")
        run_simulation(agent, scenario['problem'], scenario['context'])

if __name__ == "__main__":
    run_comprehensive_test()