# main.py
# ==============================================================================
#           PROTÓTIPO V3.7: AGENTE META-COGNITIVO COM AUTO-CORREÇÃO
#               (Versão Completa e Autocontida)
# ==============================================================================

import numpy as np
import re
import json
from typing import Dict, List, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

# --- CONFIGURAÇÃO ---
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- ESTRUTURAS DE DADOS ---
class ReasoningStrategy(Enum):
    NEURAL_INTUITIVE = "neural_intuitive"
    SYMBOLIC_MONOLITHIC = "symbolic_monolithic"
    DECOMPOSITIONAL_EXECUTOR = "decompositonal_executor"

@dataclass
class MetaCognitionState:
    current_strategy: ReasoningStrategy = ReasoningStrategy.NEURAL_INTUITIVE
    confidence_in_decision: float = 0.0
    reasoning_trace: List[str] = field(default_factory=list)
    uncertainty_sources: List[str] = field(default_factory=list)
    context_complexity: float = 0.0
    textual_complexity: float = 0.0
    attempted_strategies: List[ReasoningStrategy] = field(default_factory=list)
    decision_time: float = 0.0

# --- EXECUTOR SIMBÓLICO ---
class SymbolicExecutor:
    def __init__(self, domain: str):
        self.domain = domain
        self.state: Any = None
        self.trace: List[str] = []

    def execute_plan(self, initial_state: Dict, plan: List[Dict]) -> Dict:
        if self.domain == "tower_of_hanoi":
            return self._execute_hanoi_plan(initial_state, plan)
        else:
            return {"success": False, "error": "Dominio de execução desconhecido."}

    def _execute_hanoi_plan(self, initial_state: Dict, plan: List[Dict]) -> Dict:
        num_disks = initial_state.get("num_disks", 0)
        self.state = [list(range(num_disks, 0, -1)), [], []]
        self.trace.append(f"EXECUTOR: Estado inicial: {self.state}")
        for i, step in enumerate(plan):
            if not isinstance(step, dict):
                error_msg = f"Erro no passo {i+1}: O passo não é um dicionário. Passo: {step}"
                self.trace.append(f"EXECUTOR: FALHA! {error_msg}")
                return {"success": False, "error": error_msg, "trace": self.trace}
            move_from, move_to = step.get("from_peg"), step.get("to_peg")
            if not self._is_valid_hanoi_move(move_from, move_to):
                error_msg = f"Erro no passo {i+1}: Movimento inválido de {move_from} para {move_to}. Estado atual: {self.state}"
                self.trace.append(f"EXECUTOR: FALHA! {error_msg}")
                return {"success": False, "error": error_msg, "trace": self.trace}
            disk = self.state[move_from].pop()
            self.state[move_to].append(disk)
            self.trace.append(f"EXECUTOR: Passo {i+1}: Mover disco {disk} de {move_from} para {move_to}. Novo estado: {self.state}")
        target_peg = self.state[2]
        if len(target_peg) == num_disks and sorted(target_peg, reverse=True) == target_peg:
            self.trace.append("EXECUTOR: SUCESSO! Estado final alcançado.")
            return {"success": True, "final_state": self.state, "trace": self.trace}
        else:
            error_msg = f"Estado final incorreto: {self.state}"
            self.trace.append(f"EXECUTOR: FALHA! {error_msg}")
            return {"success": False, "error": error_msg, "trace": self.trace}

    def _is_valid_hanoi_move(self, from_peg_idx: Any, to_peg_idx: Any) -> bool:
        if not (isinstance(from_peg_idx, int) and 0 <= from_peg_idx <= 2): return False
        if not (isinstance(to_peg_idx, int) and 0 <= to_peg_idx <= 2): return False
        if not self.state[from_peg_idx]: return False
        disk_to_move = self.state[from_peg_idx][-1]
        if not self.state[to_peg_idx] or self.state[to_peg_idx][-1] > disk_to_move: return True
        return False

# --- CLASSE PRINCIPAL DO AGENTE ---
class NeuroSymbolicMetaCognition:
    def __init__(self, ollama_model: str = 'gemma3', max_correction_attempts: int = 3):
        self.ollama_model = ollama_model
        self.max_correction_attempts = max_correction_attempts
        self.symbolic_rules = self._initialize_symbolic_rules()
        self.ollama_available = OLLAMA_AVAILABLE and self._check_ollama_connection()

    def _check_ollama_connection(self) -> bool:
        if not OLLAMA_AVAILABLE:
            logging.warning("Biblioteca 'ollama' não instalada. Executando em modo simulação.")
            return False
        try:
            logging.info(f"Verificando conexão com Ollama e modelo '{self.ollama_model}'...")
            ollama.show(self.ollama_model)
            logging.info("Conexão com Ollama e modelo verificada com sucesso.")
            return True
        except Exception as e:
            logging.warning(f"Ollama não disponível: {e}. Executando em modo simulação.")
            return False

    def reset_state(self):
        self.meta_state = MetaCognitionState()
        logging.info("Estado meta-cognitivo resetado.")

    def _initialize_symbolic_rules(self) -> Dict:
        return {"mathematical": {"extract_linear_equation": r'(?i)([a-zA-Z])\s*([\+\-])\s*(\d+)\s*=\s*(\d+)'}}
    
    def _is_problem_highly_compositional(self, problem: str) -> float:
        keywords = ['torre de hanoi', 'passos múltiplos', 'planejamento', 'sequência de movimentos']
        keyword_score = sum(1 for kw in keywords if kw in problem.lower())
        match = re.search(r'com (\d+) discos', problem.lower())
        if match and int(match.group(1)) >= 3: return 1.0
        return min(1.0, keyword_score * 0.5)

    def _meta_analyze_problem(self, problem: str):
        self.meta_state.reasoning_trace.append("META: Iniciando análise do problema.")
        is_technical = any(kw in problem.lower() for kw in ['calcular', 'resolver', 'equação'])
        is_compositional = self._is_problem_highly_compositional(problem)
        if is_compositional > 0.8: self.meta_state.context_complexity = 1.0
        elif is_technical: self.meta_state.context_complexity = 0.7
        else: self.meta_state.context_complexity = 0.2
        self.meta_state.reasoning_trace.append(f"META: Complexidade percebida: {self.meta_state.context_complexity:.2f}")

    def _choose_initial_strategy(self) -> ReasoningStrategy:
        self.meta_state.reasoning_trace.append("META: Escolhendo estratégia inicial.")
        if self.meta_state.context_complexity > 0.8: return ReasoningStrategy.DECOMPOSITIONAL_EXECUTOR
        if self.meta_state.context_complexity > 0.5: return ReasoningStrategy.SYMBOLIC_MONOLITHIC
        return ReasoningStrategy.NEURAL_INTUITIVE

    def solve_problem(self, problem: str, context: Dict) -> Dict:
        start_time = time.time()
        self.reset_state()
        self._meta_analyze_problem(problem)
        strategy = self._choose_initial_strategy()
        self.meta_state.current_strategy = strategy
        self.meta_state.attempted_strategies.append(strategy)
        solution = self._execute_strategy(problem, strategy, context)
        self.meta_state.decision_time = time.time() - start_time
        return {"solution": solution, "meta_state": self.meta_state}

    def _execute_strategy(self, problem: str, strategy: ReasoningStrategy, context: Dict) -> Any:
        if strategy == ReasoningStrategy.NEURAL_INTUITIVE: return self._neural_reasoning(problem)
        elif strategy == ReasoningStrategy.SYMBOLIC_MONOLITHIC: return self._symbolic_reasoning(problem)
        elif strategy == ReasoningStrategy.DECOMPOSITIONAL_EXECUTOR: return self._decompositonal_reasoning(problem, context)
        return "Estratégia não reconhecida."
        
    def _neural_reasoning(self, problem: str) -> str:
        self.meta_state.reasoning_trace.append("NEURAL: Usando intuição rápida.")
        if not self.ollama_available:
            self.meta_state.reasoning_trace.append("NEURAL: (Modo Simulação) LLM offline.")
            self.meta_state.confidence_in_decision = 0.6
            return "Resposta simulada."
        prompt = f"Responda à seguinte pergunta de forma concisa: {problem}"
        try:
            response = ollama.chat(model=self.ollama_model, messages=[{'role': 'user', 'content': prompt}])
            solution = response['message']['content']
            self.meta_state.confidence_in_decision = 0.75
            return solution
        except Exception as e:
            self.meta_state.uncertainty_sources.append(f"neural_module_failed: {e}")
            return "Falha ao consultar o módulo neural."

    def _symbolic_reasoning(self, problem: str) -> str:
        self.meta_state.reasoning_trace.append("SYMBOLIC: Buscando regra monolítica.")
        match = re.search(self.symbolic_rules["mathematical"]["extract_linear_equation"], problem)
        if match:
            variable, operator, val1_str, val2_str = match.groups()
            val1, val2 = int(val1_str), int(val2_str)
            if operator == '-': result = val2 + val1
            else: result = val2 - val1
            self.meta_state.reasoning_trace.append(f"SYMBOLIC: Regra aplicada. {variable} {operator} {val1} = {val2} -> {variable} = {result}")
            self.meta_state.confidence_in_decision = 1.0
            return f"O valor de {variable} é {result}."
        self.meta_state.uncertainty_sources.append("no_symbolic_rule_matched")
        self.meta_state.confidence_in_decision = 0.2
        return "Nenhuma regra simbólica encontrada."

    def _decompositonal_reasoning(self, problem: str, context: Dict) -> Dict:
        self.meta_state.reasoning_trace.append("DECOMPOSITIONAL: Iniciando ciclo de planejamento e execução.")
        last_error = None
        for attempt in range(self.max_correction_attempts):
            self.meta_state.reasoning_trace.append(f"DECOMPOSITIONAL: Tentativa de planejamento #{attempt + 1}")
            plan = self._decompose_problem_with_llm(problem, last_error)
            if not plan or (isinstance(plan, dict) and "error" in plan):
                error_msg = plan.get('error', 'desconhecida') if isinstance(plan, dict) else "Plano vazio retornado."
                self.meta_state.reasoning_trace.append(f"DECOMPOSITIONAL: Falha ao gerar plano. Causa: {error_msg}")
                self.meta_state.uncertainty_sources.append("plan_generation_failed")
                last_error = f"Na tentativa anterior, seu planejador falhou ao gerar um JSON válido. Causa: {error_msg}"
                continue
            self.meta_state.reasoning_trace.append(f"DECOMPOSITIONAL: Plano gerado: {json.dumps(plan)}")
            domain, executor = context.get("domain", "unknown"), SymbolicExecutor(domain=context.get("domain", "unknown"))
            initial_state = context.get("initial_state", {})
            execution_result = executor.execute_plan(initial_state, plan)
            if execution_result.get("success"):
                self.meta_state.reasoning_trace.extend(execution_result.get("trace", []))
                self.meta_state.confidence_in_decision = 1.0
                return execution_result
            else:
                last_error = execution_result.get("error", "Erro desconhecido na execução.")
                self.meta_state.reasoning_trace.append(f"DECOMPOSITIONAL: Execução falhou. Erro: '{last_error}'. Preparando para corrigir.")
                self.meta_state.uncertainty_sources.append("plan_execution_failed")
        self.meta_state.reasoning_trace.append("DECOMPOSITIONAL: Máximo de tentativas de correção atingido. Falha final.")
        self.meta_state.confidence_in_decision = 0.0
        return {"success": False, "error": f"Falha após {self.max_correction_attempts} tentativas. Último erro: {last_error}"}

    def _decompose_problem_with_llm(self, problem: str, last_error: str = None) -> List[Dict]:
        if not self.ollama_available:
            self.meta_state.reasoning_trace.append("DECOMPOSITIONAL: (Modo Simulação) LLM offline.")
            if "torre de hanoi com 3 discos" in problem.lower() and last_error is None:
                return [{"from_peg": 0, "to_peg": 2}, {"from_peg": 0, "to_peg": 1}, {"from_peg": 2, "to_peg": 1}, {"from_peg": 0, "to_peg": 2}, {"from_peg": 1, "to_peg": 0}, {"from_peg": 1, "to_peg": 2}, {"from_peg": 0, "to_peg": 2}]
            return {"error": "Simulação de decomposição falhou ou está em ciclo de correção."}
        prompt_header = "Você é um planejador lógico que converte problemas em planos JSON executáveis."
        feedback_section = ""
        if last_error:
            feedback_section = f"""
AVISO: Seu plano anterior falhou.
Erro reportado: "{last_error}"
Analise o erro e gere um plano novo e CORRETO.
"""
        main_task = f"""
Exemplo de tarefa bem-sucedida:
Problema: "Resolva a Torre de Hanói com 2 discos."
Sua saída JSON:
[
    {{"from_peg": 0, "to_peg": 1}},
    {{"from_peg": 0, "to_peg": 2}},
    {{"from_peg": 1, "to_peg": 2}}
]

Agora, execute a seguinte tarefa:
Problema: "{problem}"
Sua saída JSON:
"""
        full_prompt = f"{prompt_header}\n\n{feedback_section}\n\n{main_task}"
        try:
            response = ollama.chat(model=self.ollama_model, messages=[{'role': 'user', 'content': full_prompt}])
            raw_content = response['message']['content']
            self.meta_state.reasoning_trace.append(f"DECOMPOSITIONAL: Resposta crua do LLM: {raw_content}")
            match = re.search(r'\[.*\]', raw_content, re.DOTALL)
            if match:
                json_str = match.group(0)
                plan = json.loads(json_str)
                if isinstance(plan, list):
                    if not plan: return {"error": "Plano extraído é uma lista vazia."}
                    if all(isinstance(s, dict) and "from_peg" in s and "to_peg" in s for s in plan): return plan
                    else: return {"error": "JSON extraído, mas chaves necessárias estão faltando."}
                else: return {"error": "JSON extraído, mas não é uma lista."}
            else: return {"error": "Nenhum bloco JSON encontrado na resposta."}
        except Exception as e:
            self.meta_state.reasoning_trace.append(f"DECOMPOSITIONAL: ERRO crítico na decomposição - {e}")
            return {"error": str(e)}

# --- FUNÇÕES DE TESTE ---
def run_simulation(system: NeuroSymbolicMetaCognition, problem_statement: str, context_info: Dict):
    print("=" * 80)
    print(f"PROBLEMA: '{problem_statement}'")
    print(f"CONTEXTO: {context_info}")
    print("-" * 80)
    result = system.solve_problem(problem_statement, context_info)
    solution = result['solution']
    meta_state = result['meta_state']
    print("\n=== RESULTADO FINAL ===")
    if isinstance(solution, dict):
        print(f"Sucesso: {solution.get('success')}")
        if solution.get('success'): print(f"Estado Final: {solution.get('final_state')}")
        else: print(f"Erro: {solution.get('error')}")
    else: print(f"Solução: {solution}")
    print(f"Estratégia Final: {meta_state.current_strategy.name}")
    print(f"Confiança na Decisão: {meta_state.confidence_in_decision:.2f}")
    print("\n=== TRACE DE RACIOCÍNIO META-COGNITIVO ===")
    for i, step in enumerate(meta_state.reasoning_trace, 1): print(f"{i:2d}. {step}")
    print("=" * 80 + "\n")

def run_comprehensive_test():
    """Executa bateria completa de testes."""
    agent = NeuroSymbolicMetaCognition(ollama_model='gemma3', max_correction_attempts=3)
    test_scenarios = [
        {"name": "Cenário 1: Baixa Complexidade (Conceitual)", "problem": "O que é entropia?", "context": {}},
        {"name": "Cenário 2: Média Complexidade (Simbólico Monolítico)", "problem": "Preciso resolver esta equação para um projeto: y - 15 = 30", "context": {}},
        {"name": "Cenário 3: Alta Complexidade (Decomposição e Execução)", "problem": "Por favor, resolva o quebra-cabeça da torre de hanoi com 3 discos.", "context": {"domain": "tower_of_hanoi", "initial_state": {"num_disks": 3}}}
    ]
    for scenario in test_scenarios:
        print(f"\n{'='*20} {scenario['name']} {'='*20}")
        run_simulation(agent, scenario['problem'], scenario['context'])

if __name__ == "__main__":
    run_comprehensive_test()