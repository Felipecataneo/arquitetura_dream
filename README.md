
# 🧠 DREAM V12.3 — Sistema Cognitivo Multi-Modelo com Humildade Epistêmica

---

## 📘 Descrição

**DREAM (Dynamic Reasoning and Epistemic Metacognition)** é uma arquitetura cognitiva evolutiva que transforma LLMs em agentes autoconscientes de suas limitações. Na versão 12.3, o sistema se torna **multi-modelo**, permitindo a alternância dinâmica entre LLMs locais (via Ollama) e de nuvem (via OpenAI), combinando o melhor dos dois mundos: privacidade e velocidade com poder e conhecimento de ponta.

O núcleo do sistema combate a "Ilusão de Pensamento" — respostas convincentes, mas logicamente falhas — utilizando uma camada metacognitiva de controle, validação simbólica e uma filosofia de **humildade epistêmica**: saber quando **não sabe**.

---

## 📁 Estrutura do Projeto

```bash
📦 dream-v12.3
├── arquiteto_final.py        # Núcleo cognitivo com pipeline multi-modelo
├── server.py                 # Backend Flask para servir o sistema via API
├── frontend.html             # Interface web com seletor de modelo
├── requirements.txt          # Dependências Python
├── .env                      # Arquivo para chaves de API (NÃO ENVIAR PARA O GIT)
└── README.md                 # Este arquivo
```

---

## 🚀 Como Rodar Localmente

### 1. Pré-requisitos

- **Git**: Para clonar o repositório.
- **Python 3.8+**: Para executar o backend.
- **Ollama**: (Opcional, para usar modelos locais) Certifique-se de que o [Ollama](https://ollama.com) está instalado.
- **Chave da API OpenAI**: (Opcional, para usar GPT) Obtenha uma chave em [platform.openai.com](https://platform.openai.com/).

### 2. Clone o repositório

```bash
git clone https://github.com/felipecataneo/arquitetura_dream.git
cd arquitetura_dream
```

### 3. Configure as Dependências e Chaves

> Recomendado usar um ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

**a. Instale as dependências:**

```bash
pip install -r requirements.txt
```

**`requirements.txt` atualizado:**
```txt
ollama
numpy
networkx
Flask 
flask-cors 
openai
dotenv
```

**b. Crie o arquivo de ambiente:**

Crie um arquivo chamado `.env` na raiz do projeto e adicione sua chave da API da OpenAI (se for usar):

```
OPENAI_API_KEY="sk-sua-chave-secreta-da-openai-aqui"
```
> **Importante:** Adicione `.env` ao seu arquivo `.gitignore` para nunca enviá-lo para o repositório.

### 4. Inicie os Modelos

**a. Modelo Local (Ollama):**

Se quiser usar modelos locais, inicie o Ollama e baixe o modelo desejado (o padrão é `gemma3`):

```bash
ollama run gemma3
```

> O sistema funcionará mesmo que o Ollama não esteja rodando, desde que a chave da OpenAI esteja configurada.

**b. Modelo de Nuvem (OpenAI):**

Nenhuma ação é necessária, desde que a chave no arquivo `.env` esteja correta.

### 5. Execute o Servidor

Na raiz do projeto, execute:

```bash
python server.py
```

A API estará disponível com dois endpoints:
```
- http://127.0.0.1:5000/solve (POST)
- http://127.0.0.1:5000/available_models (GET)
```

### 6. Abra a Interface Web

Abra o arquivo `frontend.html` no seu navegador. A interface irá carregar dinamicamente os modelos que estiverem disponíveis (Ollama e/ou OpenAI) em um menu suspenso.

---

## 💡 Exemplos de Comando Rápido

Use os botões ou digite comandos na interface, selecionando o modelo desejado:

- 💻 `Crie uma calculadora Python simples com GUI usando tkinter`
- 🎪 `O que pesa mais, um quilograma de aço ou um quilograma de penas?`
- 🔢 `Quantas vezes a letra e aparece nesta frase?`
- 🔬 `Explique a relação entre relatividade geral e mecânica quântica.`

---

## 🧠 Principais Recursos da V12.3

- **✨ Suporte Multi-Modelo:** Alterne facilmente entre LLMs locais (`gemma3`) e de nuvem (`gpt-4o-mini`) através da interface.
- ✅ **Classificação de Intenção:** Identifica se o pedido é para código, lógica, fatos, etc.
- 🧩 **Executor Simbólico:** Valida a lógica de problemas passo a passo.
- 🔄 **Feedback Loop com Auto-correção:** Utiliza fallback algorítmico quando a geração neural falha.
- 📚 **Humildade Epistêmica:** Reconhece e sinaliza os limites do seu próprio conhecimento.
- 💡 **Estratégias Especializadas:** Aplica o método de raciocínio mais adequado para cada tipo de pergunta.
- 📈 **Sistema de Cache, Saúde e Métricas:** Otimiza o desempenho e monitora a estabilidade.

---

## 🛠️ Tecnologias Usadas

- Python 3.8+
- [Ollama](https://ollama.com) (para executar LLMs locais)
- [OpenAI API](https://platform.openai.com/) (para LLMs de nuvem)
- Flask + Flask-CORS (API REST)
- python-dotenv (para gerenciamento de chaves de API)
- HTML/CSS/JS puro (interface interativa e responsiva)
- Programação Orientada a Objetos com `abc` para abstração de modelos.

---

## 🧪 Testes de Robustez

A arquitetura DREAM passou por uma evolução com várias fases, incluindo:

1.  Agente neuro-simbólico ingênuo
2.  Executor com correção simbólica
3.  Planejador generalista com classificação de domínio
4.  Virada epistêmica: admitir a ignorância e sugerir pesquisa
5.  **Refatoração Multi-Modelo: abstração para suportar diferentes provedores de LLM**

---

## 🧑‍💻 Contribuindo

Fique à vontade para abrir *Issues*, *Pull Requests*, ou sugerir melhorias.

Se quiser propor novas estratégias de raciocínio, handlers ou integrações com outros provedores de LLM, envie seu módulo e documentação.

---

## 📜 Licença

Este projeto é distribuído sob a Licença MIT.  
© 2025 — Felipe Cataneo