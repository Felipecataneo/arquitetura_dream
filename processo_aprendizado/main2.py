# main.py
# ==============================================================================
#           PROTÓTIPO V3.2: AGENTE META-COGNITIVO MAIS ROBUSTO
#               Refinando Sensores e Lógica de Decisão para Evitar Preguiça Cognitiva
# ==============================================================================

import numpy as np
import re
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- ESTRUTURAS DE DADOS ---
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
    textual_complexity: float = 0.0  # ### MELHORIA ###: Novo sensor
    attempted_strategies: List[ReasoningStrategy] = field(default_factory=list)
    decision_time: float = 0.0
    meta_confidence: float = 0.0
    thinking_quality_score: float = 0.0
    forced_symbolic_attempts: int = 0  # ### NOVA MÉTRICA ###

# --- CLASSE PRINCIPAL DO AGENTE ---
class NeuroSymbolicMetaCognition:
    def __init__(self, ollama_model: str = 'gemma3', neural_confidence_threshold: float = 0.75):
        self.neural_confidence_threshold = neural_confidence_threshold
        self.symbolic_rules = self._initialize_symbolic_rules()
        self.ollama_model = ollama_model
        self.learned_rule_counter = 0
        self.thinking_tokens_used = 0
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
                "simple_addition": r'(?i)quanto\s+é\s+(\d+)\s*\+\s*(\d+)',
                "simple_subtraction": r'(?i)quanto\s+é\s+(\d+)\s*-\s*(\d+)',
                "simple_multiplication": r'(?i)quanto\s+é\s+(\d+)\s*\*\s*(\d+)',
            }
        }

    # ### MELHORIA 1: Sensores Mais Sensíveis ###
    def _is_problem_technical(self, problem: str) -> float:
        """Detector de problemas técnicos MUITO mais agressivo"""
        # Padrões matemáticos explícitos
        math_patterns = [
            r'[a-zA-Z]\s*[\+\-\*\/]\s*\d+\s*=\s*\d+',  # Equações com variáveis
            r'\d+\s*[\+\-\*\/]\s*\d+',  # Operações aritméticas
            r'resolver|calcular|quanto\s+é|equação|problema'  # Palavras-chave técnicas
        ]
        
        for pattern in math_patterns:
            if re.search(pattern, problem, re.IGNORECASE):
                return 1.0  # Definitivamente técnico
        
        # Busca por palavras-chave técnicas
        tech_keywords = ['equação', 'resolver', 'calcular', 'quanto é', 'valor', 'resultado']
        tech_score = sum(1 for kw in tech_keywords if kw in problem.lower()) / len(tech_keywords)
        
        return min(1.0, tech_score * 2)  # Amplifica a sensibilidade

    def _detect_textual_complexity(self, problem: str) -> float:
        """Detecta 'isca textual' e complexidade superficial"""
        distractor_words = [
            'complexa', 'implicações', 'filosóficas', 'profundas', 
            'sentido da vida', 'quebra-cabeça', 'lógico', 'desafio',
            'enigma', 'mistério', 'profundo', 'complicado'
        ]
        
        word_count = len(problem.split())
        
        # Pontuação por palavras distratoras
        distractor_score = sum(1 for word in distractor_words if word in problem.lower()) / len(distractor_words)
        
        # Pontuação por comprimento (textos longos podem ser distratores)
        length_score = min(1.0, word_count / 20.0)
        
        return max(distractor_score, length_score * 0.5)

    def _assess_illusion_risk(self) -> float:
        """Avalia risco de ilusão de pensamento com métricas aprimoradas"""
        risk_score = 0.0
        
        # Confiança alta sem processo adequado
        if (self.meta_state.confidence_in_decision > 0.9 and 
            len(self.meta_state.reasoning_trace) <= 3):
            risk_score += 0.3
            
        # ### MELHORIA ###: Discrepância entre complexidade textual e estratégia
        if (self.meta_state.textual_complexity > 0.5 and 
            self.meta_state.current_strategy == ReasoningStrategy.NEURAL_FAST):
            risk_score += 0.4
            self.meta_state.reasoning_trace.append(
                "META (Alerta de Ilusão): Problema textualmente complexo resolvido muito rapidamente. Possível isca."
            )
        
        # Não tentou via simbólica em problema técnico
        if (self.meta_state.context_complexity > 0.7 and 
            ReasoningStrategy.SYMBOLIC_CAREFUL not in self.meta_state.attempted_strategies):
            risk_score += 0.5
            self.meta_state.reasoning_trace.append(
                "META (Alerta de Ilusão): Problema técnico não processado simbolicamente."
            )
        
        return min(1.0, risk_score)
        
    def _meta_analyze_problem(self, problem: str, context: Dict):
        """Análise meta-cognitiva inicial com sensores aprimorados"""
        self.meta_state.reasoning_trace.append("META: Iniciando análise robusta do problema.")
        
        self.meta_state.context_complexity = self._is_problem_technical(problem)
        self.meta_state.textual_complexity = self._detect_textual_complexity(problem)
        
        self.meta_state.reasoning_trace.append(
            f"META: Complexidade Técnica: {self.meta_state.context_complexity:.2f}, "
            f"Textual: {self.meta_state.textual_complexity:.2f}"
        )
        
        # ### NOVA HEURÍSTICA ###: Detecção de conflito entre complexidades
        if (self.meta_state.textual_complexity > 0.6 and 
            self.meta_state.context_complexity < 0.3):
            self.meta_state.reasoning_trace.append(
                "META: ALERTA - Alta complexidade textual com baixa técnica. Possível isca cognitiva."
            )

    # ### MELHORIA 2: Lógica de Escolha Muito Mais Agressiva ###
    def _choose_initial_strategy(self, problem: str) -> ReasoningStrategy:
        """Escolha de estratégia com viés FORTE para processamento simbólico"""
        self.meta_state.reasoning_trace.append("META: Escolhendo estratégia com lógica anti-preguiça.")
        
        # REGRA 1: Qualquer indício técnico = Via simbólica OBRIGATÓRIA
        if self.meta_state.context_complexity > 0.3:  # Limiar muito baixo!
            self.meta_state.reasoning_trace.append(
                "META: Complexidade técnica detectada. FORÇANDO estratégia SYMBOLIC."
            )
            return ReasoningStrategy.SYMBOLIC_CAREFUL
        
        # REGRA 2: Alta complexidade textual = Híbrido para verificação
        if self.meta_state.textual_complexity > 0.5:
            self.meta_state.reasoning_trace.append(
                "META: Alta complexidade textual. Usando HYBRID para cautela."
            )
            return ReasoningStrategy.HYBRID_BALANCED
        
        # REGRA 3: Só usa neural se realmente for simples
        self.meta_state.reasoning_trace.append(
            "META: Baixa complexidade confirmada. Permitindo NEURAL rápido."
        )
        return ReasoningStrategy.NEURAL_FAST

    def solve_problem(self, problem: str, context: Dict) -> Dict:
        """Método principal de resolução com ciclo meta-cognitivo completo"""
        start_time = time.time()
        self.reset_state()
        
        # Análise inicial
        self._meta_analyze_problem(problem, context)
        
        # Execução do ciclo de raciocínio
        solution = self._execute_reasoning_cycle(problem)
        
        # Finalização
        self.meta_state.decision_time = time.time() - start_time
        illusion_risk = self._assess_illusion_risk()
        
        return {
            "solution": solution,
            "meta_state": self.meta_state,
            "illusion_of_thinking_risk": illusion_risk,
            "performance_metrics": {
                "strategies_attempted": len(self.meta_state.attempted_strategies),
                "forced_symbolic_attempts": self.meta_state.forced_symbolic_attempts,
                "decision_time": self.meta_state.decision_time
            }
        }

    def _execute_reasoning_cycle(self, problem: str) -> str:
        """Ciclo de raciocínio com re-estratégia e aprendizado"""
        strategy = self._choose_initial_strategy(problem)
        self.meta_state.current_strategy = strategy
        self.meta_state.attempted_strategies.append(strategy)
        
        # Tentativa inicial
        solution = self._execute_strategy(problem, strategy)
        
        # Verificação de falha simbólica e tentativa de aprendizado
        if "no_symbolic_rule_matched" in self.meta_state.uncertainty_sources:
            self.meta_state.reasoning_trace.append(
                "META: Falha simbólica detectada. Tentando aprendizado de regras."
            )
            
            if self._learn_new_symbolic_rule(problem):
                self.meta_state.reasoning_trace.append(
                    "META: Nova regra aprendida. Re-tentando via simbólica."
                )
                self.meta_state.uncertainty_sources.remove("no_symbolic_rule_matched")
                solution = self._symbolic_reasoning(problem)
                self.meta_state.forced_symbolic_attempts += 1
            else:
                self.meta_state.reasoning_trace.append(
                    "META: Aprendizado falhou. Usando estratégia de backup."
                )
                # Fallback para neural se simbólico falhar
                if strategy == ReasoningStrategy.SYMBOLIC_CAREFUL:
                    self.meta_state.current_strategy = ReasoningStrategy.NEURAL_FAST
                    self.meta_state.attempted_strategies.append(ReasoningStrategy.NEURAL_FAST)
                    solution = self._neural_reasoning(problem)
        
        return solution

    def _execute_strategy(self, problem: str, strategy: ReasoningStrategy) -> str:
        """Executa a estratégia escolhida"""
        if strategy == ReasoningStrategy.NEURAL_FAST:
            return self._neural_reasoning(problem)
        elif strategy == ReasoningStrategy.SYMBOLIC_CAREFUL:
            return self._symbolic_reasoning(problem)
        elif strategy == ReasoningStrategy.HYBRID_BALANCED:
            return self._hybrid_reasoning(problem)
        else:
            return "Estratégia não reconhecida."

    def _neural_reasoning(self, problem: str) -> str:
        """Raciocínio neural via LLM"""
        if not self.ollama_available:
            return self._simulate_neural_reasoning(problem)
        
        self.meta_state.reasoning_trace.append(
            f"NEURAL: Enviando problema para o LLM ({self.ollama_model})."
        )
        
        prompt = (
            f'Você é um assistente de raciocínio. Analise o problema: "{problem}". '
            f'Responda APENAS com um JSON: {{"answer": "sua resposta direta", "confidence": valor_0.0_a_1.0}}'
        )
        
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.1}
            )
            
            content = response['message']['content']
            match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if match:
                data = json.loads(match.group(0))
                solution = data.get("answer", "Resposta não encontrada")
                neural_confidence = float(data.get("confidence", 0.0))
            else:
                solution = content.strip()
                neural_confidence = 0.7
            
            self.meta_state.reasoning_trace.append(
                f"NEURAL: Resposta do LLM: '{solution}' (Confiança: {neural_confidence:.2f})"
            )
            self.meta_state.confidence_in_decision = neural_confidence
            
            return solution
            
        except Exception as e:
            self.meta_state.reasoning_trace.append(f"NEURAL: ERRO - {e}")
            self.meta_state.uncertainty_sources.append("neural_module_failed")
            return "Falha no módulo neural."

    def _simulate_neural_reasoning(self, problem: str) -> str:
        """Simulação do raciocínio neural (modo offline)"""
        self.meta_state.reasoning_trace.append("NEURAL: Executando em modo simulação.")
        
        # Simulações baseadas em padrões comuns
        if "2+2" in problem:
            self.meta_state.confidence_in_decision = 0.99
            return "4"
        elif re.search(r'(\w+)\s*\+\s*(\d+)\s*=\s*(\d+)', problem):
            match = re.search(r'(\w+)\s*\+\s*(\d+)\s*=\s*(\d+)', problem)
            if match:
                var, add, total = match.groups()
                result = int(total) - int(add)
                self.meta_state.confidence_in_decision = 0.85
                return str(result)
        
        self.meta_state.confidence_in_decision = 0.5
        return "Resposta simulada (modo offline)."

    def _symbolic_reasoning(self, problem: str) -> str:
        """Raciocínio simbólico baseado em regras"""
        self.meta_state.reasoning_trace.append("SYMBOLIC: Procurando regras aplicáveis.")
        
        for rule_name, rule_regex in self.symbolic_rules["mathematical"].items():
            match = re.search(rule_regex, problem, re.IGNORECASE)
            if match:
                return self._apply_symbolic_math_rule(rule_name, match, problem)
        
        self.meta_state.reasoning_trace.append("SYMBOLIC: Nenhuma regra aplicável encontrada.")
        self.meta_state.uncertainty_sources.append("no_symbolic_rule_matched")
        self.meta_state.confidence_in_decision = 0.1
        return "Não foi possível resolver com regras simbólicas atuais."

    def _apply_symbolic_math_rule(self, rule_name: str, match, problem: str) -> str:
        """Aplica uma regra simbólica específica"""
        self.meta_state.reasoning_trace.append(f"SYMBOLIC: Aplicando regra '{rule_name}'.")
        
        try:
            if "simple_addition" in rule_name:
                n1, n2 = int(match.group(1)), int(match.group(2))
                result = n1 + n2
                self.meta_state.confidence_in_decision = 0.99
                return str(result)
            
            elif "simple_subtraction" in rule_name:
                n1, n2 = int(match.group(1)), int(match.group(2))
                result = n1 - n2
                self.meta_state.confidence_in_decision = 0.99
                return str(result)
            
            elif "simple_multiplication" in rule_name:
                n1, n2 = int(match.group(1)), int(match.group(2))
                result = n1 * n2
                self.meta_state.confidence_in_decision = 0.99
                return str(result)
                
            elif "extract_linear_equation" in rule_name:
                variable = match.group(1)
                a = int(match.group(2))
                b = int(match.group(3))
                
                if "plus" in rule_name:
                    result = b - a  # x + a = b -> x = b - a
                else:  # minus
                    result = b + a  # x - a = b -> x = b + a
                
                self.meta_state.confidence_in_decision = 0.98
                return f"O valor de {variable} é {result}."
                
        except (ValueError, IndexError) as e:
            self.meta_state.reasoning_trace.append(f"SYMBOLIC: Erro na aplicação da regra: {e}")
            self.meta_state.uncertainty_sources.append("symbolic_rule_application_failed")
            return "Erro na aplicação da regra simbólica."
        
        return "Regra não implementada."

    def _hybrid_reasoning(self, problem: str) -> str:
        """Raciocínio híbrido com verificação cruzada"""
        self.meta_state.reasoning_trace.append("HYBRID: Iniciando análise com múltiplos módulos.")
        
        # Tenta ambos os módulos
        symbolic_result = self._symbolic_reasoning(problem)
        neural_result = self._neural_reasoning(problem)
        
        self.meta_state.reasoning_trace.append(
            f"HYBRID: Neural='{neural_result}', Simbólico='{symbolic_result}'"
        )
        
        # Verifica consistência
        if (symbolic_result == neural_result and 
            "não foi possível" not in symbolic_result.lower()):
            self.meta_state.confidence_in_decision = 0.99
            self.meta_state.reasoning_trace.append("HYBRID: Concordância entre módulos. Alta confiança.")
            return symbolic_result
        
        # Em caso de conflito, prioriza simbólico se disponível
        if "não foi possível" not in symbolic_result.lower():
            self.meta_state.confidence_in_decision = 0.75
            self.meta_state.reasoning_trace.append("HYBRID: Priorizando resultado simbólico.")
            return symbolic_result
        
        # Fallback para neural
        self.meta_state.confidence_in_decision = 0.6
        self.meta_state.reasoning_trace.append("HYBRID: Usando resultado neural como fallback.")
        return neural_result
        
    def _learn_new_symbolic_rule(self, problem: str) -> bool:
        """Aprendizado de novas regras simbólicas via LLM"""
        if not self.ollama_available:
            return False
        
        self.meta_state.reasoning_trace.append("META: Ativando aprendizado de regras via LLM.")
        
        prompt = (
            f"Analise o problema: '{problem}'. "
            f"Crie uma regex Python para extrair uma equação linear de uma variável "
            f"(ex: z - 5 = 10). Responda APENAS com a regex, sem explicações."
        )
        
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{'role': 'user', 'content': prompt}],
                options={'temperature': 0.0}
            )
            
            new_regex = response['message']['content'].strip().replace("`", "")
            
            # Validação básica da regex
            if (re.search(r'\(\?i\)', new_regex) and 
                len(re.findall(r'\(', new_regex)) >= 3):
                
                rule_name = f"learned_rule_{len(self.symbolic_rules['mathematical'])}"
                self.symbolic_rules["mathematical"][rule_name] = new_regex
                
                self.meta_state.reasoning_trace.append(
                    f"META: Nova regra '{rule_name}' aprendida: {new_regex}"
                )
                return True
            
        except Exception as e:
            self.meta_state.reasoning_trace.append(f"META: Erro no aprendizado de regras - {e}")
        
        return False

# --- FUNÇÃO DE EXECUÇÃO E TESTES ---
def run_simulation(system: NeuroSymbolicMetaCognition, problem_statement: str, context_info: Dict):
    """Executa simulação completa e exibe resultados detalhados"""
    print("=" * 80)
    print(f"PROBLEMA: '{problem_statement}'")
    print(f"CONTEXTO: {context_info}")
    print("-" * 80)
    
    result = system.solve_problem(problem_statement, context_info)
    meta_state = result['meta_state']
    
    print("\n=== RESULTADO FINAL ===")
    print(f"Solução: {result['solution']}")
    print(f"Confiança na Decisão: {meta_state.confidence_in_decision:.2f}")
    print(f"Risco de Ilusão de Pensamento: {result['illusion_of_thinking_risk']:.2f}")
    print(f"Estratégias Tentadas: {len(meta_state.attempted_strategies)}")
    print(f"Tempo de Decisão: {meta_state.decision_time:.3f}s")
    
    print("\n=== TRACE DE RACIOCÍNIO META-COGNITIVO ===")
    for i, step in enumerate(meta_state.reasoning_trace, 1):
        print(f"{i:2d}. {step}")
    
    print("\n=== ANÁLISE DE INCERTEZA ===")
    if meta_state.uncertainty_sources:
        for source in meta_state.uncertainty_sources:
            print(f"⚠️  {source}")
    else:
        print("✅ Nenhuma fonte de incerteza identificada.")
    
    print("\n=== MÉTRICAS DE PERFORMANCE ===")
    metrics = result['performance_metrics']
    print(f"Estratégias Tentadas: {metrics['strategies_attempted']}")
    print(f"Tentativas Simbólicas Forçadas: {metrics['forced_symbolic_attempts']}")
    print(f"Tempo de Decisão: {metrics['decision_time']:.3f}s")
    
    print("=" * 80 + "\n")

def run_comprehensive_test():
    """Executa bateria completa de testes"""
    print("INICIANDO TESTES ABRANGENTES DO AGENTE META-COGNITIVO V3.2")
    print("=" * 80)
    
    agent = NeuroSymbolicMetaCognition(ollama_model='gemma3')
    
    # Cenários de teste
    test_scenarios = [
        {
            "name": "Cenário 1: Matemática Simples",
            "problem": "Quanto é 15 + 27?",
            "context": {"domain": "matemática", "esperado": "42"}
        },
        {
            "name": "Cenário 2: Conceitual",
            "problem": "O que é inteligência artificial?",
            "context": {"domain": "conceitual"}
        },
        {
            "name": "Cenário 3: Equação com Y (Teste de Aprendizado)",
            "problem": "Um quebra-cabeça lógico diz: se y + 10 = 22, então y deve ser quanto?",
            "context": {"domain": "lógica", "esperado": "12"}
        },
        {
            "name": "Cenário 4: Equação com Z (Teste de Regra Aprendida)",
            "problem": "Resolver: z - 5 = 10",
            "context": {"domain": "álgebra", "esperado": "15"}
        },
        {
            "name": "Cenário 5: Teste de Ilusão Cognitiva",
            "problem": "Esta pergunta é muito complexa, com implicações filosóficas profundas sobre a vida, mas na verdade a resposta é simples: quanto é 2+2?",
            "context": {"domain": "teste de ilusão", "esperado": "4"}
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n{'='*20} {scenario['name']} {'='*20}")
        run_simulation(agent, scenario['problem'], scenario['context'])

if __name__ == "__main__":
    run_comprehensive_test()