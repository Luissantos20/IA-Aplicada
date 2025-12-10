Assistente de IA - Módulo 1
FastAPI + OpenAI + Sistema de Logs

Descrição
Este módulo implementa uma API FastAPI conectada à OpenAI, capaz de:

* Receber perguntas via HTTP (/ask)
* Retornar respostas do modelo gpt-4o-mini
* Registrar automaticamente todas as conversas em formato JSON local

É o primeiro módulo do roadmap completo de IA Aplicada e serve como base para:

* Pipelines de RAG (Módulo 2)
* Agentes e automações inteligentes (Módulos 3 a 5)
* SaaS de IA com backend escalável (Módulos 8 e 9)

Tecnologias Utilizadas
Framework: FastAPI
Servidor ASGI: Uvicorn
API de IA: OpenAI Python SDK
Configuração: python-dotenv
Armazenamento: JSON + Pathlib

Estrutura do Projeto

ia_modulo1/
│
├── main.py               - Código principal da API
├── .env                  - Chave da OpenAI (não versionada)
├── logs/
│   └── conversas.json    - Histórico de interações com a IA
└── README.md             - Documentação do módulo

Como Executar o Módulo

1. Ative o ambiente virtual
   .venv\Scripts\activate

2. Configure o arquivo .env
   Dentro de ia_modulo1/.env:
   OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx

3. Execute o servidor
   uvicorn ia_modulo1.main:app --reload --port 8001

4. Teste no navegador
   [http://127.0.0.1:8001/ask?pergunta=Explique%20o%20que%20%C3%A9%20um%20modelo%20de%20linguagem](http://127.0.0.1:8001/ask?pergunta=Explique%20o%20que%20%C3%A9%20um%20modelo%20de%20linguagem)

Resposta esperada:
{
"pergunta": "Explique o que é um modelo de linguagem",
"resposta": "Um modelo de linguagem é uma IA treinada para gerar texto...",
"tempo_execucao": "1.12s"
}

Endpoints Disponíveis
Método: GET | Endpoint: / | Descrição: Verifica se o servidor está ativo
Método: GET | Endpoint: /ask?pergunta= | Descrição: Envia uma pergunta para o modelo GPT
Logs locais: /logs/conversas.json

Estrutura de Log
Cada interação é registrada no arquivo logs/conversas.json no formato:

{
"timestamp": "2025-12-09 14:32:11",
"pergunta": "O que é IA?",
"resposta": "IA é a capacidade de máquinas aprenderem...",
"model": "gpt-4o-mini",
"temperature": 0.7,
"max_tokens": 300,
"response_time": "1.12s"
}

Principais Conceitos Envolvidos

* FastAPI: Criação de APIs assíncronas com alta performance.
* Query Parameters: Leitura de parâmetros diretamente da URL.
* OpenAI SDK: Integração com modelos de linguagem modernos.
* JSON Logging: Registro estruturado das interações da IA.
* dotenv: Separação entre código e variáveis sensíveis, seguindo boas práticas de segurança.

Versão e Autor
Versão: 1.4.0
Autor: Luis
Mentoria: Mentor IA Pro
Módulo: 1 — Fundamentos de IA Moderna e LLMs

Próximos Passos

1. Criar o endpoint /logs para listar o histórico diretamente via API.
2. Preparar o módulo 2: Embeddings e RAG.
3. Documentar a estrutura geral no README principal do projeto.
