# 🧠 DREAM V12.2 — Sistema Cognitivo com Humildade Epistêmica

---

## 📘 Descrição

**DREAM (Dynamic Reasoning and Epistemic Metacognition)** é uma arquitetura cognitiva evolutiva que transforma um LLM tradicional em um agente autoconsciente de suas limitações. O sistema combate a "Ilusão de Pensamento" — respostas convincentes, mas logicamente falhas — utilizando uma camada metacognitiva de controle, validação simbólica e uma filosofia de **humildade epistêmica**: saber quando **não sabe**.

---

## 📁 Estrutura do Projeto

```bash
📦 dream-v12
├── arquiteto_final.py        # Núcleo cognitivo com pipeline completo
├── server.py                 # Backend Flask para servir o sistema via API
├── frontend.html             # Interface web interativa e responsiva
├── requirements.txt          # Dependências Python
└── README.md                 # Este arquivo
```

---

## 🚀 Como Rodar Localmente

### 1. Clone o repositório

```bash
git clone https://github.com/felipecataneo/arquitetura_dream.git
cd arquitetura_dream - main
```

### 2. Instale as dependências

> Recomendado usar um ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**`requirements.txt`**
```txt
ollama
numpy
networkx
Flask
flask-cors
```

### 3. Inicie o modelo com o Ollama

Certifique-se de ter o [Ollama](https://ollama.com) instalado e execute:

```bash
ollama run gemma3
```

> O modelo padrão utilizado é `gemma3`. Altere no `arquiteto_final.py` se necessário.

### 4. Execute o servidor

```bash
python server.py
```

A API estará disponível em:
```
http://127.0.0.1:5000/solve
```

### 5. Abra a interface web

Abra o arquivo `frontend.html` no navegador.

---

## 💡 Exemplos de Comando Rápido

Use os botões ou digite comandos na interface:

- 💻 `Crie uma calculadora Python simples com GUI usando tkinter`
- 🎪 `O que pesa mais, um quilograma de aço ou um quilograma de penas?`
- 🔢 `Quantas vezes a letra e aparece nesta frase?`
- 🔬 `Explique a relação entre relatividade geral e mecânica quântica.`

---

## 🧠 Principais Recursos

- ✅ Classificação de intenção (ex: código, lógica, fatos, pesquisa)
- 🧩 Executor simbólico (ex: valida Torre de Hanói passo a passo)
- 🔄 Feedback loop com auto-correção (com fallback algorítmico)
- 📚 Reconhecimento de limite de conhecimento (humildade epistêmica)
- 💡 Estratégias especializadas para cada tipo de pergunta
- 📈 Sistema de cache, saúde, métricas e aprendizado adaptativo

---

## 🛠️ Tecnologias Usadas

- Python 3.7+
- [Ollama](https://ollama.com) (para executar o modelo LLM local)
- Flask + Flask-CORS (API REST)
- HTML/CSS/JS puro (interface interativa)
- Enum, dataclasses, typing, JSON e execução de código seguro

---

## 🧪 Testes de Robustez

A arquitetura DREAM passou por uma evolução com várias fases, incluindo:

1. Agente neuro-simbólico ingênuo
2. Executor com correção simbólica
3. Planejador generalista com classificação de domínio
4. Virada epistêmica: admitir a ignorância e sugerir pesquisa

---

## 🧑‍💻 Contribuindo

Fique à vontade para abrir *Issues*, *Pull Requests*, ou sugerir melhorias.

Se quiser propor novas estratégias de raciocínio, handlers ou domínios, envie seu módulo e documentação.

---

## 📜 Licença

Este projeto é distribuído sob a Licença MIT.  
© 2025 — Felipe Cataneo