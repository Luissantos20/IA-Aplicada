"""
==============================================================
Assistente de IA - Módulo 1
--------------------------------------------------------------
Este módulo implementa uma API FastAPI conectada à OpenAI,
com sistema de logging automático para registrar perguntas,
respostas e metadados de interação.

Versão: 1.4.0
Autor: Luis
Mentoria: Mentor IA Pro
==============================================================
"""

from fastapi import FastAPI, Query
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
import os, json, time

# ============================================================
#  CONFIGURAÇÃO INICIAL
# ============================================================

# Caminho absoluto até o arquivo .env (variáveis de ambiente)
dotenv_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=dotenv_path)

# Inicializa o cliente da OpenAI com a chave da API
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Instância principal da aplicação FastAPI
app = FastAPI(
    title="Assistente de IA - Módulo 1",
    description="API conectada à OpenAI com sistema de logs e auditoria.",
    version="1.4.0"
)

# Caminho do arquivo onde as conversas serão armazenadas
LOG_FILE = Path(__file__).resolve().parent / "logs" / "conversas.json"
LOG_FILE.parent.mkdir(exist_ok=True)  # Garante que a pasta 'logs/' exista


# ============================================================
#  FUNÇÃO AUXILIAR DE LOG
# ============================================================

def registrar_conversa(pergunta: str, resposta: str, parametros: dict, tempo_execucao: float) -> None:
    """
    Registra uma interação de usuário com o assistente em formato JSON.

    Args:
        pergunta (str): Texto enviado pelo usuário.
        resposta (str): Resposta gerada pelo modelo da OpenAI.
        parametros (dict): Dicionário contendo modelo, temperature e max_tokens.
        tempo_execucao (float): Tempo total de processamento da requisição, em segundos.

    Estrutura do registro salvo:
        {
            "timestamp": "YYYY-MM-DD HH:MM:SS",
            "pergunta": "...",
            "resposta": "...",
            "model": "gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 300,
            "response_time": "1.21s"
        }

    O arquivo JSON é criado (ou atualizado) em:
        /ia_modulo1/logs/conversas.json
    """
    registro = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pergunta": pergunta,
        "resposta": resposta,
        "model": parametros.get("model"),
        "temperature": parametros.get("temperature"),
        "max_tokens": parametros.get("max_tokens"),
        "response_time": f"{tempo_execucao:.2f}s"
    }

    # Lê logs anteriores (se existirem)
    if LOG_FILE.exists():
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []  # Se o arquivo estiver vazio ou corrompido
    else:
        logs = []

    logs.append(registro)

    # Grava o novo registro no arquivo JSON
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=4, ensure_ascii=False)


# ============================================================
#  ENDPOINTS DA API
# ============================================================

@app.get("/")
def home():
    """
    Endpoint raiz de verificação de status do servidor.

    Returns:
        dict: Confirmação de que o servidor está ativo e a versão atual do módulo.
    """
    return {
        "status": "Servidor ativo",
        "modulo": "1.4 - Logs e Histórico de Conversas"
    }


@app.get("/ask")
def ask(pergunta: str = Query(..., description="Digite sua pergunta para o assistente")):
    """
    Endpoint principal da API de IA.

    Recebe uma pergunta via query string, envia para o modelo GPT e retorna a resposta.

    Exemplo:
        GET /ask?pergunta=Explique%20o%20que%20é%20um%20modelo%20de%20linguagem

    Args:
        pergunta (str): Texto enviado pelo usuário.

    Returns:
        dict: Contém a pergunta original, a resposta gerada e o tempo de execução.

    Observação:
        As interações são automaticamente registradas no arquivo de logs.
    """

    # Marca o tempo de início da execução
    inicio = time.time()

    # Parâmetros do modelo (podem ser ajustados conforme necessidade)
    parametros = {
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 300
    }

    # Envia a requisição para o modelo da OpenAI
    resposta = client.chat.completions.create(
        model=parametros["model"],
        messages=[
            {"role": "system", "content": "Você é um assistente técnico e explicativo."},
            {"role": "user", "content": pergunta}
        ],
        temperature=parametros["temperature"],
        max_tokens=parametros["max_tokens"]
    )

    # Extrai o conteúdo da resposta do modelo
    conteudo = resposta.choices[0].message.content

    # Calcula o tempo total de resposta
    fim = time.time()
    duracao = fim - inicio

    # Registra o log da conversa
    registrar_conversa(pergunta, conteudo, parametros, duracao)

    # Retorna o resultado da requisição
    return {
        "pergunta": pergunta,
        "resposta": conteudo,
        "tempo_execucao": f"{duracao:.2f}s"
    }
