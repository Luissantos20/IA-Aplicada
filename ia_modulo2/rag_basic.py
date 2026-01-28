"""
==============================================================
RAG Básico (Retrieval-Augmented Generation)
--------------------------------------------------------------
Demonstra o pipeline de busca + geração contextualizada.
==============================================================
"""

# ============================================================
#  IMPORTAÇÕES
# ============================================================

import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import os

# ============================================================
#  CONFIGURAÇÃO DE CLIENTES
# ============================================================

load_dotenv()

client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client_chroma = chromadb.PersistentClient(path="./chroma_db")

collection = client_chroma.get_or_create_collection(
    name="meus_textos",
    embedding_function=None  # Usaremos nossos próprios embeddings
)

# ============================================================
#  FUNÇÃO DE RAG
# ============================================================

def responder_com_contexto(pergunta: str, n_contextos: int = 2):
    """
    Busca os textos mais relevantes no Chroma e usa como contexto
    para gerar uma resposta inteligente via GPT.
    """

    # Gera embedding da pergunta
    embedding_pergunta = client_openai.embeddings.create(
        model="text-embedding-3-small",
        input=[pergunta]
    ).data[0].embedding

    # Busca os contextos mais próximos no Chroma
    resultados = collection.query(
        query_embeddings=[embedding_pergunta],
        n_results=n_contextos
    )

    # Extrai os textos relevantes
    contextos = resultados["documents"][0]
    contexto_concatenado = "\n".join(contextos)

    # Monta o prompt contextualizado
    prompt = f"""
    Você é um assistente inteligente.
    Use o contexto abaixo para responder à pergunta do usuário.

    CONTEXTO:
    {contexto_concatenado}

    PERGUNTA:
    {pergunta}

    Responda de forma clara e explicativa, com base apenas no contexto acima.
    """

    # Gera resposta com o modelo GPT
    resposta = client_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Você é um assistente técnico e explicativo."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.6,
        max_tokens=400
    )

    return resposta.choices[0].message.content

# ============================================================
#  TESTE DO PIPELINE RAG
# ============================================================

if __name__ == "__main__":
    pergunta = "qual animal está descansando?"
    resposta = responder_com_contexto(pergunta)
    print("\n💬 Pergunta:", pergunta)
    print("\n🧠 Resposta contextualizada:\n")
    print(resposta)
