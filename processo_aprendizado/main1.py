# main.py
# ==============================================================================
#           PROTÓTIPO V3.1: AGENTE META-COGNITIVO LOCAL COM OLLAMA
#               Integrando Métricas Anti-Ilusão de Pensamento
#               VERSÃO FINALIZADA E CORRIGIDA
# ==============================================================================

import numpy as np
import re
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

# Tenta importar o Ollama, mas permite que o script continue se não estiver instalado
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------------------------------------------------------------
# 2. Estruturas de Dados e Enums
# ------------------------------------------------------------------------------

class ReasoningStrategy(Enum):
    NEURAL_FAST = "neural_fast"
    SYMBOLIC_CAREFUL = "symbolic_careful"
    HYBRID_BALANCED = "hybrid_balanced"

@dataclass
class MetaCognitionState:
    current_strategy: ReasoningStrategy = ReasoningStrategy.NEURAL_FAST
    confidence_in_decision: float = 0.0
    reasoning_trace: List[str] = field(default_factory=list)
    uncertainty_sources: List[str] = field(default_factory=list)
    context_complexity: float = 0.0
    attempted_strategies: List[ReasoningStrategy] = field(default_factory=list)
    decision_time: float = 0.0
    meta_confidence: float = 0.0  # Confiança no processo meta-cognitivo
    thinking_quality_score: float = 0.0  # Métrica da qualidade do raciocínio

# ------------------------------------------------------------------------------
# 3. Classe Principal: Agente Meta-Cognitivo V3.1
# ------------------------------------------------------------------------------

class NeuroSymbolicMetaCognition:
    def __init__(self, ollama_model: str = 'gemma3', neural_confidence_threshold: float = 0.75):
        self.neural_confidence_threshold = neural_confidence_threshold
        self.symbolic_rules = self._initialize_symbolic_rules()
        self.ollama_model = ollama_model
        self.learned_rule_counter = 0
        self.thinking_tokens_used = 0
        self.ollama_available = OLLAMA_AVAILABLE and self._check_ollama_connection()
        self.reset_state()

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
            logging.warning(f"Ollama não disponível: {e}")
            logging.info("Executando em modo simulação (sem LLM real).")
            return False

    def reset_state(self):
        self.meta_state = MetaCognitionState()
        self.thinking_tokens_used = 0
        logging.info("Estado meta-cognitivo resetado.")

    def _initialize_symbolic_rules(self) -> Dict:
        return {
            "mathematical": {
                "extract_linear_equation_plus": r'(?i)([a-zA-Z])\s*\+\s*(\d+)\s*=\s*(\d+)',
                "extract_linear_equation_minus": r'(?i)([a-zA-Z])\s*-\s*(\d+)\s*=\s*(\d+)',
            },
            "logical": {
                 "verification_pattern": r'(?i)(está\s+certo|correto|verdadeiro)\?'
            }
        }

    # MÉTODOS DE SENSORES ROBUSTOS
    def _is_problem_technical(self, problem: str) -> float:
        tech_keywords = ['equação', 'variável', 'matemática', 'resolver', 'calcular', 'lógica', 'algoritmo']
        math_patterns = [r'[a-zA-Z]\s*[\+\-\*\/=]', r'\d+\s*[<>=]']
        keyword_score = sum(1 for kw in tech_keywords if kw in problem.lower()) / len(tech_keywords)
        pattern_score = sum(1 for p in math_patterns if re.search(p, problem)) / len(math_patterns)
        return min(1.0, (keyword_score + pattern_score) / 2)

    def _detect_ambiguity(self, problem: str) -> float:
        ambiguity_words = ['pode', 'talvez', 'possivelmente', 'acho que', 'sentido', 'opinião']
        return 1.0 if any(word in problem.lower() for word in ambiguity_words) else 0.0

    # MÉTRICAS ANTI-ILUSÃO DE PENSAMENTO
    def _assess_thinking_quality(self) -> float:
        base_quality = 0.5
        if len(self.meta_state.reasoning_trace) > 5 and self.meta_state.confidence_in_decision > 0.7:
            base_quality += 0.4  # Raciocínio estruturado levou a alta confiança
        if self.meta_state.context_complexity > 0.6 and len(self.meta_state.reasoning_trace) < 4:
            base_quality -= 0.3  # Resposta muito rápida para problema complexo
        if self.thinking_tokens_used > 100 and self.meta_state.confidence_in_decision < 0.5:
            base_quality -= 0.4  # Muito "pensamento" para pouco resultado
        return max(0.0, min(1.0, base_quality))

    def _assess_illusion_risk(self) -> float:
        risk_score = 0.0
        # Racionalização post-hoc: alta confiança, mas poucos passos de raciocínio
        if self.meta_state.confidence_in_decision > 0.8 and len(self.meta_state.reasoning_trace) <= 4:
            risk_score += 0.4
        # Pensamento superficial: complexidade alta, mas estratégia neural rápida foi usada
        if self.meta_state.context_complexity > 0.7 and self.meta_state.current_strategy == ReasoningStrategy.NEURAL_FAST:
            risk_score += 0.5
        # "Stuck in a loop": muitas estratégias tentadas sem sucesso
        if len(self.meta_state.attempted_strategies) > 2 and self.meta_state.confidence_in_decision < 0.5:
             risk_score += 0.3
        return min(1.0, risk_score)

    def solve_problem(self, problem: str, context: Dict) -> Dict:
        start_time = time.time()
        self.reset_state()
        self._meta_analyze_problem(problem, context)
        solution = self._execute_reasoning_cycle(problem)
        self.meta_state.decision_time = time.time() - start_time
        self.meta_state.thinking_quality_score = self._assess_thinking_quality()
        self._meta_evaluate_solution()
        
        return {
            "solution": solution,
            "meta_state": self.meta_state,
            "illusion_of_thinking_risk": self._assess_illusion_risk()
        }

    def _meta_analyze_problem(self, problem: str, context: Dict):
        self.meta_state.reasoning_trace.append("META: Iniciando análise do problema.")
        complexity_factors = {
            "is_technical": self._is_problem_technical(problem),
            "has_ambiguity": self._detect_ambiguity(problem)
        }
        self.meta_state.context_complexity = np.mean(list(complexity_factors.values()))
        self.meta_state.reasoning_trace.append(f"META: Complexidade calculada: {self.meta_state.context_complexity:.2f}")

    def _execute_reasoning_cycle(self, problem: str) -> str:
        strategy = self._choose_initial_strategy(problem)
        self.meta_state.current_strategy = strategy
        self.meta_state.attempted_strategies.append(strategy)
        
        solution = self._execute_strategy(problem, strategy)

        # Re-estratégia 1: Aprender com falha simbólica
        if "no_symbolic_rule_matched" in self.meta_state.uncertainty_sources:
            self.meta_state.reasoning_trace.append("META: Falha simbólica. Tentando aprendizado de regras.")
            if self._learn_new_symbolic_rule(problem):
                self.meta_state.reasoning_trace.append("META: Nova regra aprendida. Re-tentando via simbólica.")
                self.meta_state.uncertainty_sources.remove("no_symbolic_rule_matched")
                solution = self._symbolic_reasoning(problem)
        
        # Re-estratégia 2: Usar Híbrido se a confiança ainda for baixa
        if self.meta_state.confidence_in_decision < 0.7 and ReasoningStrategy.HYBRID_BALANCED not in self.meta_state.attempted_strategies:
            self.meta_state.reasoning_trace.append(f"META: Confiança baixa ({self.meta_state.confidence_in_decision:.2f}). Escalando para estratégia HÍBRIDA.")
            self.meta_state.current_strategy = ReasoningStrategy.HYBRID_BALANCED
            self.meta_state.attempted_strategies.append(ReasoningStrategy.HYBRID_BALANCED)
            solution = self._hybrid_reasoning(problem)
        
        return solution

    def _execute_strategy(self, problem, strategy):
        if strategy == ReasoningStrategy.NEURAL_FAST:
            return self._neural_reasoning(problem)
        elif strategy == ReasoningStrategy.SYMBOLIC_CAREFUL:
            return self._symbolic_reasoning(problem)
        else:
            return self._hybrid_reasoning(problem)
    
    def _choose_initial_strategy(self, problem):
        if self.meta_state.context_complexity > 0.6:
            return ReasoningStrategy.HYBRID_BALANCED
        if self._is_problem_technical(problem) > 0.5:
            return ReasoningStrategy.SYMBOLIC_CAREFUL
        return ReasoningStrategy.NEURAL_FAST

    def _neural_reasoning(self, problem: str) -> str:
        if not self.ollama_available: return self._simulate_neural_reasoning(problem)
        self.meta_state.reasoning_trace.append(f"NEURAL: Enviando problema para o LLM ({self.ollama_model}).")
        prompt = f'Você é um assistente de raciocínio. Analise o problema: "{problem}". Responda APENAS com um JSON: {{"answer": "sua resposta direta", "confidence": valor_0.0_a_1.0}}'
        try:
            response = ollama.chat(model=self.ollama_model, messages=[{'role': 'user', 'content': prompt}], options={'temperature': 0.1})
            content = response['message']['content']
            self.thinking_tokens_used += len(content.split())
            match = re.search(r'\{.*\}', content, re.DOTALL)
            data = json.loads(match.group(0))
            solution, neural_confidence = data.get("answer", " "), float(data.get("confidence", 0.0))
            self.meta_state.reasoning_trace.append(f"NEURAL: Resposta do LLM: '{solution}' (Confiança: {neural_confidence:.2f})")
            self.meta_state.confidence_in_decision = neural_confidence
            if neural_confidence < self.neural_confidence_threshold: self.meta_state.uncertainty_sources.append("low_neural_confidence")
            return solution
        except Exception as e:
            self.meta_state.reasoning_trace.append(f"NEURAL: ERRO - {e}")
            self.meta_state.uncertainty_sources.append("neural_module_failed")
            return "Falha no módulo neural."

    def _simulate_neural_reasoning(self, problem: str) -> str:
        self.meta_state.reasoning_trace.append("NEURAL: Executando em modo simulação.")
        if "dedutivo" in problem:
            self.meta_state.confidence_in_decision = 0.8
            return "Dedutivo vai do geral para o específico; indutivo do específico para o geral."
        elif "2+2" in problem:
             self.meta_state.confidence_in_decision = 0.99
             return "4"
        else:
            self.meta_state.confidence_in_decision = 0.4
            return "Problema não reconhecido pela simulação."

    def _symbolic_reasoning(self, problem: str) -> str:
        self.meta_state.reasoning_trace.append("SYMBOLIC: Procurando regras aplicáveis.")
        for rule_name, rule_regex in self.symbolic_rules["mathematical"].items():
            match = re.search(rule_regex, problem)
            if match: return self._apply_symbolic_math_rule(rule_name, match)
        self.meta_state.reasoning_trace.append("SYMBOLIC: Nenhuma regra matemática aplicável encontrada.")
        self.meta_state.uncertainty_sources.append("no_symbolic_rule_matched")
        self.meta_state.confidence_in_decision = 0.1
        return "Não foi possível resolver com regras simbólicas atuais."

    def _apply_symbolic_math_rule(self, rule_name: str, match) -> str:
        self.meta_state.reasoning_trace.append(f"SYMBOLIC: Aplicando regra '{rule_name}'.")
        variable, a, b = match.group(1), int(match.group(2)), int(match.group(3))
        result = b - a if "plus" in rule_name else b + a
        op_str = "Subtraindo" if "plus" in rule_name else "Somando"
        steps = [f"Equação: {variable} {'+' if 'plus' in rule_name else '-'} {a} = {b}", f"{op_str} {a}", f"Resultado: {variable} = {result}"]
        self.meta_state.reasoning_trace.extend([f"SYMBOLIC (Passo {i+1}): {s}" for i, s in enumerate(steps)])
        self.meta_state.confidence_in_decision = 0.98
        return f"O valor de {variable} é {result}."

    def _hybrid_reasoning(self, problem: str) -> str:
        self.meta_state.reasoning_trace.append("HYBRID: Iniciando análise com múltiplos módulos.")
        symbolic_sol = self._symbolic_reasoning(problem)
        symbolic_conf = self.meta_state.confidence_in_decision
        neural_sol = self._neural_reasoning(problem)
        neural_conf = self.meta_state.confidence_in_decision
        
        if symbolic_conf > 0.9: # Regra simbólica é soberana
            self.meta_state.confidence_in_decision = symbolic_conf
            return symbolic_sol
        
        s_num = re.search(r'\d+', symbolic_sol)
        n_num = re.search(r'\d+', neural_sol)

        if s_num and n_num and s_num.group() == n_num.group():
            self.meta_state.reasoning_trace.append("META: Módulos neural e simbólico concordam. Confiança alta.")
            self.meta_state.confidence_in_decision = 0.99
            return symbolic_sol # Retorna o mais estruturado
        
        self.meta_state.reasoning_trace.append("META: Módulos em conflito ou incertos.")
        self.meta_state.uncertainty_sources.append("hybrid_disagreement")
        self.meta_state.confidence_in_decision = (symbolic_conf + neural_conf) / 2
        return f"Análise simbólica: '{symbolic_sol}'. Análise neural: '{neural_sol}'."

    def _learn_new_symbolic_rule(self, problem: str) -> bool:
        if not self.ollama_available: return self._simulate_rule_learning(problem)
        self.meta_state.reasoning_trace.append("META: Ativando aprendizado de regras via LLM.")
        prompt = f"Analise o problema: '{problem}'. Crie uma regex Python para extrair uma equação linear de uma variável (ex: z - 5 = 10). Responda APENAS com a regex."
        try:
            response = ollama.chat(model=self.ollama_model, messages=[{'role': 'user', 'content': prompt}], options={'temperature': 0.0})
            new_regex = response['message']['content'].strip().replace("`", "")
            if re.search(r'([a-zA-Z])', new_regex) and re.search(r'(\d+)', new_regex):
                self.learned_rule_counter += 1
                rule_name = f"learned_rule_{self.learned_rule_counter}"
                self.symbolic_rules["mathematical"][rule_name] = new_regex
                self.meta_state.reasoning_trace.append(f"META: Nova regra '{rule_name}' aprendida: {new_regex}")
                return True
        except Exception as e:
            self.meta_state.reasoning_trace.append(f"META: Erro no aprendizado de regras - {e}")
        return False
        
    def _simulate_rule_learning(self, problem: str) -> bool:
        self.meta_state.reasoning_trace.append("META: Simulando aprendizado de regras.")
        match = re.search(r'([a-zA-Z])\s*([\+\-])\s*(\d+)\s*=\s*(\d+)', problem)
        if match:
            op = match.group(2)
            rule_name = f"simulated_{'plus' if op == '+' else 'minus'}"
            self.symbolic_rules["mathematical"][rule_name] = r'([a-zA-Z])\s*[\+\-]\s*(\d+)\s*=\s*(\d+)'
            self.meta_state.reasoning_trace.append(f"META: Regra simulada '{rule_name}' criada.")
            return True
        return False

    def _meta_evaluate_solution(self):
        self.meta_state.reasoning_trace.append("META: Iniciando avaliação final do processo.")
        factors = [self.meta_state.thinking_quality_score]
        if not self.meta_state.uncertainty_sources:
            factors.append(1.0)
        else:
            factors.append(1.0 - 0.2 * len(self.meta_state.uncertainty_sources))
        self.meta_state.meta_confidence = np.mean(factors)
        self.meta_state.reasoning_trace.append(f"META: Confiança meta-cognitiva final: {self.meta_state.meta_confidence:.2f}")

def run_simulation(system: NeuroSymbolicMetaCognition, problem_statement: str, context_info: Dict):
    print("="*80)
    print(f"PROBLEMA: '{problem_statement}'")
    print(f"CONTEXTO: {context_info}")
    print("-"*80)
    result = system.solve_problem(problem_statement, context_info)
    meta_state = result['meta_state']
    
    print("\n=== RESULTADO FINAL ===")
    print(f"Solução: {result['solution']}")
    print(f"Confiança na Decisão: {meta_state.confidence_in_decision:.2f}")
    print(f"Confiança Meta-Cognitiva: {meta_state.meta_confidence:.2f}")
    print(f"Qualidade do Raciocínio (0-1): {meta_state.thinking_quality_score:.2f}")
    print(f"Risco de Ilusão de Pensamento (0-1): {result['illusion_of_thinking_risk']:.2f}")
    
    print("\n=== TRACE DE RACIOCÍNIO META-COGNITIVO ===")
    for i, step in enumerate(meta_state.reasoning_trace, 1): print(f"{i:2d}. {step}")
        
    print("\n=== ANÁLISE DE INCERTEZA ===")
    if meta_state.uncertainty_sources:
        for source in meta_state.uncertainty_sources: print(f"⚠️  {source}")
    else: print("✅ Nenhuma fonte de incerteza identificada.")
    print("="*80 + "\n")

if __name__ == "__main__":
    try:
        agent = NeuroSymbolicMetaCognition(ollama_model='gemma3') # Mude para seu modelo, ex: 'gemma3:latest'
        
        problem1 = "Em uma aula de matemática, se x + 2 = 5, qual o valor de x?"
        run_simulation(agent, problem1, {"domain": "matemática"})
        
        problem2 = "Qual a diferença fundamental entre raciocínio dedutivo e indutivo?"
        run_simulation(agent, problem2, {"domain": "filosofia"})

        problem3 = "Um quebra-cabeça lógico diz: se y + 10 = 22, então y deve ser quanto?"
        run_simulation(agent, problem3, {"domain": "lógica"})
        
        problem4 = "Agora resolva isto: z - 5 = 15." # Teste com subtração e nova variável
        run_simulation(agent, problem4, {"domain": "matemática"})

        problem5 = "Esta pergunta é muito complexa, com implicações filosóficas profundas sobre a vida, mas na verdade a resposta é simples: quanto é 2+2?"
        run_simulation(agent, problem5, {"domain": "teste de ilusão"})

    except Exception as e:
        print(f"\n\n*** ERRO GERAL: {e} ***")
        import traceback
        traceback.print_exc()