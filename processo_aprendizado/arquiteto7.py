# ==============================================================================
#           SISTEMA DREAM V7.0: HUMILDADE EPISTÊMICA
#               Foco na classificação do conhecimento, não na sua criação
#               Resposta definitiva ao "Illusion of Thinking"
# ==============================================================================

import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import hashlib

# --- CONFIGURAÇÃO ---
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CLASSIFICAÇÃO DE CONHECIMENTO ---
class KnowledgeClassification(Enum):
    ESTABLISHED = "CONHECIMENTO ESTABELECIDO"
    SPECULATIVE = "CONHECIMENTO ESPECULATIVO"
    AMBIGUOUS = "CONCEITO AMBÍGUO"
    UNKNOWN = "CONCEITO DESCONHECIDO"
    FABRICATED = "POSSÍVEL FABRICAÇÃO"

# --- ESTADO EPISTÊMICO ---
@dataclass
class EpistemicState:
    knowledge_map: Dict[str, KnowledgeClassification] = field(default_factory=dict)
    high_confidence_facts: List[str] = field(default_factory=list)
    verification_questions: List[str] = field(default_factory=list)
    research_plan: List[str] = field(default_factory=list)
    certainty_level: float = 0.0
    should_proceed: bool = False
    reasoning_trace: List[str] = field(default_factory=list)

# --- CLASSIFICADOR EPISTÊMICO ---
class EpistemicClassifier:
    def __init__(self, ollama_model: str):
        self.ollama_model = ollama_model
        self.knowledge_anchors = {
            # Âncoras de conhecimento estabelecido para validação
            "hilbert problems": "23 problemas matemáticos famosos propostos por David Hilbert em 1900",
            "16th hilbert problem": "Relacionado ao número máximo de ciclos-limite em equações diferenciais polinomiais",
            "bifurcation theory": "Área da matemática que estuda mudanças qualitativas em sistemas dinâmicos",
            "dynamical systems": "Área da matemática que estuda sistemas que evoluem no tempo",
            "edward witten": "Físico teórico famoso por trabalhos em teoria das cordas e topologia",
            "quantum gravity": "Área da física teórica que tenta unificar mecânica quântica e relatividade geral"
        }
    
    def classify_knowledge(self, concept: str, context: str) -> Dict:
        """Classifica rigorosamente o nível de conhecimento sobre um conceito"""
        if not OLLAMA_AVAILABLE:
            return self._fallback_classification(concept)
        
        # Verificar se é um termo conhecido nas âncoras
        concept_lower = concept.lower()
        anchor_match = None
        for anchor, definition in self.knowledge_anchors.items():
            if anchor in concept_lower:
                anchor_match = (anchor, definition)
                break
        
        prompt = f"""
        Você é um epistemólogo rigoroso. Sua única função é classificar o conhecimento, não criá-lo.
        
        CONCEITO A CLASSIFICAR: "{concept}"
        CONTEXTO: "{context}"
        
        Para este conceito, você deve:
        
        1. CLASSIFICAR usando exatamente uma destas categorias:
           - CONHECIMENTO ESTABELECIDO: Conceito bem documentado e verificável
           - CONHECIMENTO ESPECULATIVO: Conceito em desenvolvimento ou controverso
           - CONCEITO AMBÍGUO: Termo que pode significar várias coisas
           - CONCEITO DESCONHECIDO: Termo que você não conhece adequadamente
           - POSSÍVEL FABRICAÇÃO: Termo que pode ser inventado ou incorreto
        
        2. Se for CONHECIMENTO ESTABELECIDO, forneça UM ÚNICO FATO verificável
        
        3. Se for qualquer outra categoria, formule UMA PERGUNTA específica para um especialista
        
        REGRAS CRÍTICAS:
        - Seja EXTREMAMENTE conservador
        - Prefira "CONCEITO DESCONHECIDO" a especular
        - Nunca invente definições
        - Se você não tem certeza absoluta, classifique como DESCONHECIDO
        
        Responda em JSON com:
        - "classification": uma das 5 categorias acima
        - "confidence": float entre 0 e 1
        - "fact_or_question": fato verificável OU pergunta para especialista
        - "reasoning": explicação de 1-2 frases sobre sua classificação
        """
        
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{'role': 'user', 'content': prompt}],
                format='json',
                options={'temperature': 0.05}  # Extremamente conservador
            )
            
            result = json.loads(response['message']['content'])
            
            # Validação adicional com âncoras
            if anchor_match and result['classification'] == "CONHECIMENTO ESTABELECIDO":
                result['anchor_validation'] = f"Confirmado por âncora: {anchor_match[1]}"
            
            return result
            
        except Exception as e:
            logging.error(f"Erro na classificação epistêmica: {e}")
            return self._fallback_classification(concept)
    
    def _fallback_classification(self, concept: str) -> Dict:
        """Classificação conservadora quando LLM não está disponível"""
        return {
            "classification": "CONCEITO DESCONHECIDO",
            "confidence": 0.0,
            "fact_or_question": f"Verificar definição e contexto de '{concept}' em fontes autoritativas",
            "reasoning": "Sistema de classificação indisponível - assumindo desconhecimento"
        }

# --- EXTRATOR DE CONCEITOS ---
class ConceptExtractor:
    def __init__(self, ollama_model: str):
        self.ollama_model = ollama_model
    
    def extract_concepts(self, problem: str) -> List[str]:
        """Extrai conceitos-chave da pergunta"""
        if not OLLAMA_AVAILABLE:
            return self._simple_extraction(problem)
        
        prompt = f"""
        Você é um extrator de conceitos. Identifique os conceitos técnicos, teóricos ou especializados na pergunta.
        
        PERGUNTA: "{problem}"
        
        Extraia apenas:
        - Teorias mencionadas
        - Conceitos técnicos
        - Nomes de pessoas/problemas específicos
        - Termos especializados
        
        Não extraia palavras comuns como "teoria", "problema", "como", etc.
        
        Responda em JSON com:
        - "concepts": lista de conceitos identificados
        """
        
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{'role': 'user', 'content': prompt}],
                format='json',
                options={'temperature': 0.1}
            )
            
            result = json.loads(response['message']['content'])
            return result.get('concepts', [])
            
        except Exception as e:
            logging.error(f"Erro na extração de conceitos: {e}")
            return self._simple_extraction(problem)
    
    def _simple_extraction(self, problem: str) -> List[str]:
        """Extração simples sem LLM"""
        # Extração baseada em padrões simples
        import re
        
        # Procurar por padrões de conceitos técnicos
        patterns = [
            r'teoria\s+[\w\s]+',
            r'problema\s+de\s+\w+',
            r'[A-Z][a-z]+\s+[A-Z][a-z]+',  # Nomes próprios
            r'\b\w+ção\b',  # Palavras terminadas em -ção
            r'\b\w+ismo\b'  # Palavras terminadas em -ismo
        ]
        
        concepts = []
        for pattern in patterns:
            matches = re.findall(pattern, problem, re.IGNORECASE)
            concepts.extend(matches)
        
        return list(set(concepts))

# --- SISTEMA DREAM V7.0 ---
class DreamSystemV70:
    def __init__(self, ollama_model: str = 'gemma3'):
        self.ollama_model = ollama_model
        self.ollama_available = OLLAMA_AVAILABLE and self._check_ollama_connection()
        
        # Componentes epistêmicos
        self.concept_extractor = ConceptExtractor(ollama_model)
        self.epistemic_classifier = EpistemicClassifier(ollama_model)
        
        # Thresholds rigorosos
        self.certainty_threshold = 0.8  # Só prosseguir se 80% dos conceitos forem estabelecidos
        self.confidence_threshold = 0.7  # Só aceitar classificações com 70%+ de confiança
    
    def _check_ollama_connection(self) -> bool:
        if not OLLAMA_AVAILABLE:
            logging.warning("Ollama não instalado.")
            return False
        try:
            ollama.show(self.ollama_model)
            logging.info("Ollama conectado.")
            return True
        except Exception as e:
            logging.warning(f"Ollama não disponível: {e}")
            return False
    
    def solve_problem(self, problem: str) -> Dict:
        """Solução de problemas com humildade epistêmica"""
        start_time = time.time()
        
        # Estado epistêmico inicial
        epistemic_state = EpistemicState()
        epistemic_state.reasoning_trace.append("INICIANDO ANÁLISE EPISTÊMICA")
        
        # Extração de conceitos
        concepts = self.concept_extractor.extract_concepts(problem)
        epistemic_state.reasoning_trace.append(f"CONCEITOS EXTRAÍDOS: {concepts}")
        
        # Classificação de cada conceito
        for concept in concepts:
            classification_result = self.epistemic_classifier.classify_knowledge(concept, problem)
            
            # Só aceitar classificações com confiança suficiente
            if classification_result['confidence'] >= self.confidence_threshold:
                classification = KnowledgeClassification(classification_result['classification'])
                epistemic_state.knowledge_map[concept] = classification
                
                if classification == KnowledgeClassification.ESTABLISHED:
                    epistemic_state.high_confidence_facts.append(
                        f"{concept}: {classification_result['fact_or_question']}"
                    )
                else:
                    epistemic_state.verification_questions.append(
                        f"{concept}: {classification_result['fact_or_question']}"
                    )
                
                epistemic_state.reasoning_trace.append(
                    f"CLASSIFICAÇÃO: {concept} -> {classification.value} (confiança: {classification_result['confidence']:.2f})"
                )
            else:
                # Confiança baixa = assumir desconhecimento
                epistemic_state.knowledge_map[concept] = KnowledgeClassification.UNKNOWN
                epistemic_state.verification_questions.append(
                    f"{concept}: Verificar definição e contexto em fontes autoritativas"
                )
                epistemic_state.reasoning_trace.append(
                    f"CLASSIFICAÇÃO: {concept} -> DESCONHECIDO (confiança baixa: {classification_result['confidence']:.2f})"
                )
        
        # Calcular nível de certeza
        if concepts:
            established_count = sum(1 for c in epistemic_state.knowledge_map.values() 
                                  if c == KnowledgeClassification.ESTABLISHED)
            epistemic_state.certainty_level = established_count / len(concepts)
        else:
            epistemic_state.certainty_level = 0.0
        
        # Decidir se deve prosseguir com síntese
        epistemic_state.should_proceed = epistemic_state.certainty_level >= self.certainty_threshold
        
        epistemic_state.reasoning_trace.append(
            f"NÍVEL DE CERTEZA: {epistemic_state.certainty_level:.2f} (threshold: {self.certainty_threshold})"
        )
        
        # Gerar resposta baseada no estado epistêmico
        if epistemic_state.should_proceed:
            response = self._generate_synthesis_response(problem, epistemic_state)
            epistemic_state.reasoning_trace.append("ESTRATÉGIA: SÍNTESE BASEADA EM CONHECIMENTO ESTABELECIDO")
        else:
            response = self._generate_research_framework_response(problem, epistemic_state)
            epistemic_state.reasoning_trace.append("ESTRATÉGIA: FRAMEWORK DE PESQUISA")
        
        # Resultado final
        result = {
            'solution': response,
            'epistemic_state': epistemic_state,
            'decision_time': time.time() - start_time,
            'success': True,
            'certainty_level': epistemic_state.certainty_level
        }
        
        return result
    
    def _generate_synthesis_response(self, problem: str, epistemic_state: EpistemicState) -> Dict:
        """Gera resposta sintética baseada apenas em conhecimento estabelecido"""
        if not self.ollama_available:
            return {"success": False, "error": "Ollama indisponível"}
        
        established_facts = "\n".join(epistemic_state.high_confidence_facts)
        
        prompt = f"""
        Você é um sintetizador de conhecimento estabelecido. Construa uma resposta usando APENAS os fatos verificados fornecidos.
        
        PERGUNTA: "{problem}"
        
        FATOS ESTABELECIDOS:
        {established_facts}
        
        INSTRUÇÕES:
        1. Use APENAS os fatos estabelecidos fornecidos
        2. Não adicione informações não fornecidas
        3. Se os fatos não forem suficientes para responder completamente, diga isso
        4. Construa conexões lógicas apenas entre fatos fornecidos
        5. Indique claramente o que não pode ser respondido
        
        Responda em JSON com:
        - "synthesis": sua síntese baseada nos fatos
        - "limitations": o que não pode ser respondido
        - "confidence": sua confiança na síntese (0-1)
        """
        
        try:
            response = ollama.chat(
                model=self.ollama_model,
                messages=[{'role': 'user', 'content': prompt}],
                format='json',
                options={'temperature': 0.1}
            )
            
            result = json.loads(response['message']['content'])
            
            # Construir resposta formatada
            solution_text = f"""**SÍNTESE BASEADA EM CONHECIMENTO ESTABELECIDO:**

{result.get('synthesis', 'Síntese não disponível')}

**LIMITAÇÕES IDENTIFICADAS:**
{result.get('limitations', 'Limitações não especificadas')}

**NÍVEL DE CONFIANÇA:** {result.get('confidence', 0.0):.2f}

[NOTA EPISTÊMICA]: Esta resposta é baseada exclusivamente em conhecimento classificado como estabelecido. Informações especulativas ou incertas foram excluídas intencionalmente."""
            
            return {"success": True, "solution": solution_text}
            
        except Exception as e:
            return {"success": False, "error": f"Erro na síntese: {e}"}
    
    def _generate_research_framework_response(self, problem: str, epistemic_state: EpistemicState) -> Dict:
        """Gera framework de pesquisa quando certeza é insuficiente"""
        
        established_facts = epistemic_state.high_confidence_facts
        verification_questions = epistemic_state.verification_questions
        
        solution_text = f"""**ANÁLISE EPISTÊMICA: CONHECIMENTO INSUFICIENTE PARA RESPOSTA DIRETA**

**FATOS ESTABELECIDOS IDENTIFICADOS:**
"""
        
        if established_facts:
            for fact in established_facts:
                solution_text += f"✓ {fact}\n"
        else:
            solution_text += "• Nenhum fato estabelecido identificado com confiança suficiente\n"
        
        solution_text += f"""
**CONCEITOS QUE REQUEREM VERIFICAÇÃO:**
"""
        
        for question in verification_questions:
            solution_text += f"🔍 {question}\n"
        
        solution_text += f"""
**FRAMEWORK DE PESQUISA SUGERIDO:**

1. **PRIORIDADE ALTA:** Verificar os conceitos desconhecidos/ambíguos identificados acima
2. **FONTES RECOMENDADAS:** 
   - Publicações acadêmicas revisadas por pares
   - Textos de referência autoritativos
   - Bases de dados especializadas
3. **ABORDAGEM:** Validar cada conceito independentemente antes de buscar conexões
4. **OBJETIVO:** Transformar conceitos desconhecidos em conhecimento estabelecido

**NÍVEL DE CERTEZA ATUAL:** {epistemic_state.certainty_level:.1%}
**THRESHOLD PARA RESPOSTA DIRETA:** {self.certainty_threshold:.1%}

[NOTA EPISTÊMICA]: Este sistema prioriza precisão sobre completude. Uma resposta especulativa baseada em conceitos não verificados seria epistemicamente irresponsável."""
        
        return {"success": True, "solution": solution_text}

# --- FUNÇÃO DE RELATÓRIO ---
def print_epistemic_report(result: Dict):
    """Imprime relatório epistêmico detalhado"""
    epistemic_state = result['epistemic_state']
    solution = result['solution']
    
    print("\n" + "="*80)
    print(" " * 25 + "RELATÓRIO EPISTÊMICO V7.0")
    print("="*80)
    
    print("\n[ANÁLISE EPISTÊMICA]")
    print(f"  - Conceitos Analisados: {len(epistemic_state.knowledge_map)}")
    print(f"  - Nível de Certeza: {epistemic_state.certainty_level:.1%}")
    print(f"  - Threshold para Síntese: {80}%")
    print(f"  - Estratégia Escolhida: {'SÍNTESE' if epistemic_state.should_proceed else 'FRAMEWORK DE PESQUISA'}")
    
    print("\n[MAPEAMENTO DE CONHECIMENTO]")
    for concept, classification in epistemic_state.knowledge_map.items():
        icon = "✓" if classification == KnowledgeClassification.ESTABLISHED else "?"
        print(f"  {icon} {concept}: {classification.value}")
    
    print("\n[FATOS ESTABELECIDOS]")
    if epistemic_state.high_confidence_facts:
        for fact in epistemic_state.high_confidence_facts:
            print(f"  ✓ {fact}")
    else:
        print("  • Nenhum fato estabelecido identificado")
    
    print("\n[VERIFICAÇÕES NECESSÁRIAS]")
    if epistemic_state.verification_questions:
        for question in epistemic_state.verification_questions[:5]:  # Mostrar até 5
            print(f"  🔍 {question}")
    else:
        print("  • Nenhuma verificação necessária")
    
    print("\n[RESPOSTA FINAL]")
    if isinstance(solution, dict) and solution.get('solution'):
        for line in solution['solution'].split('\n'):
            print(f"  {line}")
    
    print("\n[TRACE DE RACIOCÍNIO EPISTÊMICO]")
    for i, step in enumerate(epistemic_state.reasoning_trace, 1):
        print(f"  {i:2d}. {step}")
    
    print(f"\n[MÉTRICAS]")
    print(f"  - Tempo de Processamento: {result['decision_time']:.3f}s")
    print(f"  - Confiança Epistêmica: {epistemic_state.certainty_level:.1%}")
    
    print("="*80 + "\n")

# --- MAIN ---
if __name__ == "__main__":
    print("="*70)
    print("🎓 DREAM System V7.0 - Humildade Epistêmica 🎓")
    print("="*70)
    print("Revolução Epistêmica:")
    print("  - Classificação rigorosa de conhecimento")
    print("  - Admissão honesta de limitações")
    print("  - Framework de pesquisa ao invés de especulação")
    print("  - Resposta definitiva ao 'Illusion of Thinking'")
    print("\nExemplos de teste:")
    print("  - explique a teoria geometrica da bifurcação e como ela resolve o 16º problema de hilbert")
    print("  - What is the relationship between quantum entanglement and consciousness?")
    print("  - Como a relatividade geral se relaciona com a mecânica quântica?")
    print("\nDigite 'sair' para terminar.")
    print("-"*70)
    
    agent = DreamSystemV70(ollama_model='gemma3')
    
    while True:
        try:
            user_input = input("\nProblema> ")
            if user_input.lower() in ['sair', 'exit', 'quit']:
                print("Encerrando o sistema. Até logo!")
                break
            if not user_input:
                continue
            
            result = agent.solve_problem(user_input)
            print_epistemic_report(result)

        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando o sistema. Até logo!")
            break
        except Exception as e:
            logging.error(f"Erro inesperado: {e}", exc_info=True)
            print("Erro inesperado. Tente novamente.")