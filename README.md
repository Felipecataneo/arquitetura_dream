
# 🧠 DREAM V13.2 — Sistema AGI Multi-LLM Robusto com API Assíncrona

---

## 📘 Descrição

**DREAM (Dynamic Reasoning and Epistemic Metacognition) V13.2** é a mais recente evolução de uma arquitetura cognitiva que transforma LLMs em agentes de raciocínio robustos e autoconscientes. Esta versão representa um salto em capacidade e escalabilidade, introduzindo:

1.  **Estratégias de Raciocínio Avançadas:** O sistema agora vai além da simples resposta, empregando técnicas como **debate adversarial** para verificação de fatos, **autocrítica e refinamento** para questões complexas, e **decomposição hierárquica** para sínteses amplas.
2.  **Geração de Código de Nível Profissional:** O handler de código foi completamente redesenhado para analisar requisitos, planejar arquiteturas, gerar código de alta qualidade com templates de fallback e até mesmo tentar uma execução segura.
3.  **API Assíncrona com FastAPI:** O backend foi migrado para FastAPI, permitindo o processamento de tarefas complexas e demoradas em segundo plano. Os usuários recebem um `task_id` imediatamente e podem consultar o status e o resultado posteriormente, evitando timeouts e melhorando a experiência do usuário.

O núcleo do sistema continua a combater a "Ilusão de Pensamento" — respostas convincentes, mas falhas — utilizando uma camada metacognitiva de controle, validação e a filosofia de **humildade epistêmica**: saber quando não sabe e aplicar a melhor estratégia para cada desafio.

---

## 📁 Estrutura do Projeto

```bash
📦 dream-v13.2
├── arquiteto_final.py        # Núcleo cognitivo V13.2 com pipeline, estratégias avançadas e geração de código robusta
├── server.py                 # Backend FastAPI para servir o sistema via API assíncrona
├── frontend.html             # Interface web que interage com a API assíncrona (com polling)
├── requirements.txt          # Dependências Python atualizadas
├── .env                      # Arquivo para chaves de API (NÃO ENVIAR PARA O GIT)
└── README.md                 # Este arquivo
```

---

## 🚀 Como Rodar Localmente

### 1. Pré-requisitos

- **Git**: Para clonar o repositório.
- **Python 3.8+**: Para executar o backend.
- **Ollama**: (Opcional, para usar modelos locais) Certifique-se de que o [Ollama](https://ollama.com) está instalado e em execução.
- **Chave da API OpenAI**: (Opcional, para usar GPT) Obtenha uma chave em [platform.openai.com](https://platform.openai.com/).

### 2. Clone o repositório

```bash
git clone https://github.com/felipecataneo/arquitetura_dream.git
cd arquitetura_dream
```

### 3. Configure as Dependências e Chaves

> É altamente recomendado usar um ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

**a. Instale as dependências:**

```bash
pip install -r requirements.txt
```

**`requirements.txt` atualizado:**
```txt
fastapi
uvicorn[standard]
python-dotenv
ollama
openai
```

**b. Crie o arquivo de ambiente:**

Crie um arquivo chamado `.env` na raiz do projeto e adicione sua chave da API da OpenAI (se for usar):

```
OPENAI_API_KEY="sk-sua-chave-secreta-da-openai-aqui"
```
> **Importante:** Adicione `.env` ao seu arquivo `.gitignore` para nunca enviá-lo para o repositório.

### 4. Inicie os Modelos Locais (Opcional)

Se quiser usar modelos locais, inicie o Ollama e baixe o modelo desejado (o padrão é `gemma3`):

```bash
ollama run gemma3
```

> O sistema funcionará mesmo que o Ollama não esteja rodando, desde que a chave da OpenAI esteja configurada.

### 5. Execute o Servidor FastAPI

Na raiz do projeto, execute o servidor com **Uvicorn**:

```bash
uvicorn server:app --reload
```

A API estará disponível na porta `8000` com os seguintes endpoints principais:
```
- http://127.0.0.1:8000/solve_async (POST) -> Inicia uma tarefa e retorna um task_id.
- http://127.0.0.1:8000/task/{task_id} (GET) -> Verifica o status de uma tarefa.
- http://127.0.0.1:8000/available_models (GET) -> Lista os modelos disponíveis.
- http://127.0.0.1:8000/solve (POST) -> Endpoint síncrono para compatibilidade.
```

### 6. Abra a Interface Web

Abra o arquivo `frontend.html` no seu navegador. A interface irá se conectar à API rodando na porta `8000`, carregar dinamicamente os modelos disponíveis e interagir de forma assíncrona com o backend.

---

## 💡 Exemplos de Perguntas

Use os botões ou digite comandos na interface, selecionando o modelo desejado:

- 💻 **Código:** `Faça uma animação web com um pentágono rotativo e uma bola com física dentro dele`
- 🧩 **Lógica:** `Um grupo de 4 pessoas com tempos de 1, 2, 5 e 10 minutos precisa cruzar uma ponte com uma tocha. Qual o tempo mínimo?`
- 🔬 **Acadêmico:** `Explique a teoria da relatividade geral usando o método de debate adversarial para garantir a precisão.`
- 📜 **Síntese:** `Faça uma síntese completa sobre a história da inteligência artificial, decompondo o problema em sub-questões.`

---

## 🧠 Principais Recursos da V13.2

- **✨ Suporte Multi-Modelo:** Alterne facilmente entre LLMs locais (`gemma3`) e de nuvem (`gpt-4o`, `gpt-4o-mini`) através da interface.
- **🚀 API Assíncrona com FastAPI:** Arquitetura robusta e escalável que processa tarefas complexas em segundo plano, melhorando drasticamente a experiência do usuário.
- **🛠️ Geração de Código Robusta:** Pipeline completo que inclui análise de requisitos, planejamento de arquitetura, geração de código de alta qualidade, validação e templates de fallback.
- **⚔️ Estratégias de Raciocínio Avançadas:**
    - **Debate Adversarial:** Usa um modelo "proponente" e um "desafiador" para verificar fatos e aumentar a precisão.
    - **Autocrítica e Refinamento:** Gera um rascunho, critica-o rigorosamente e depois o refina.
    - **Decomposição Hierárquica:** Quebra problemas amplos em sub-questões gerenciáveis e depois sintetiza a resposta final.
- **📚 Humildade Epistêmica:** O sistema continua a reconhecer os limites do seu próprio conhecimento, aplicando a melhor estratégia para cada desafio.
- **📈 Sistema de Cache, Saúde e Métricas:** Otimiza o desempenho, evita trabalho redundante e monitora a estabilidade do sistema em tempo real.

---

## 🛠️ Tecnologias Usadas

- Python 3.8+
- **FastAPI** + **Uvicorn** (API REST assíncrona de alta performance)
- [Ollama](https://ollama.com) (para executar LLMs locais)
- [OpenAI API](https://platform.openai.com/) (para LLMs de nuvem)
- `python-dotenv` (para gerenciamento de chaves de API)
- HTML/CSS/JS puro (interface interativa com polling para resultados assíncronos)
- Programação Orientada a Objetos com `abc` para abstração de modelos.

---

## 📜 Licença

Este projeto é distribuído sob a Licença MIT.  
© 2024 — Felipe Cataneo