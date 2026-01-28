"""
==============================================================
Armazenamento e Busca Semântica com ChromaDB
--------------------------------------------------------------
Este script demonstra como criar uma base vetorial local com
ChromaDB, inserir embeddings e realizar consultas semânticas.
==============================================================
"""

# ============================================================
#  IMPORTAÇÕES
# ============================================================

import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np

# ============================================================
#  CONFIGURAÇÃO DE CLIENTES
# ============================================================

# Carrega variáveis de ambiente (.env)
load_dotenv()

# Inicializa cliente da OpenAI
client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Inicializa cliente do ChromaDB (modo persistente)
client_chroma = chromadb.PersistentClient(path="./chroma_db")

# ============================================================
#  CRIAÇÃO OU ACESSO À COLEÇÃO
# ============================================================

collection = client_chroma.get_or_create_collection(
    name="meus_textos",
    metadata={
        "descricao": "Exemplo de coleção de textos semânticos",
        "hnsw:space": "cosine",       
    },
    embedding_function=None # Não usa o modelo iterno do chroma, vamos utilizar da openAI
)

# ============================================================
#  INSERÇÃO DE DADOS (TEXTOS E EMBEDDINGS)
# ============================================================

# Textos de exemplo
textos = [
    "Um cachorro brincando na grama",
    "Um carro esportivo acelerando na pista",
    "Um gato dormindo no sofá",
    "Um atleta correndo na praia"
]

# Gera embeddings para cada texto usando o modelo da OpenAI
resposta = client_openai.embeddings.create(
    model="text-embedding-3-small",
    input=textos
)

# Extrai os vetores (embeddings)
embeddings = [np.array(item.embedding) for item in resposta.data]

# Adiciona tudo à coleção
collection.add(
    ids=[f"id_{i}" for i in range(len(textos))],
    documents=textos,
    embeddings=embeddings,
    metadatas=[{"origem": "exemplo"} for _ in textos]
)

# ============================================================
#  CONSULTA SEMÂNTICA
# ============================================================

consulta = "animal descansando"
print(f"\n🔍 Consulta: {consulta}\n")

# Gera embedding da consulta usando o mesmo modelo
consulta_embedding = client_openai.embeddings.create(
    model="text-embedding-3-small",
    input=[consulta]
).data[0].embedding

# Faz a busca usando o embedding gerado
resultado = collection.query(
    query_embeddings=[consulta_embedding],
    n_results=2
)

# Exibe os resultados 
for doc, dist in zip(resultado["documents"][0], resultado["distances"][0]): 
    print(f"Texto encontrado: {doc}") 
    print(f"Distância semântica: {dist:.4f}\n") 
