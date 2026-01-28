"""
==============================================================
Embeddings e Similaridade Semântica
--------------------------------------------------------------
Este script demonstra como gerar embeddings de texto usando
a API da OpenAI e como calcular a similaridade de cosseno entre
diferentes sentenças.

Documentação de referência:
- https://platform.openai.com/docs/guides/embeddings

Autor: Luis
Mentoria: Mentor IA Pro
==============================================================
"""

# ============================================================
#  IMPORTAÇÕES
# ============================================================

from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import os

# ============================================================
#  CONFIGURAÇÃO INICIAL
# ============================================================

# Carrega variáveis de ambiente (.env deve conter OPENAI_API_KEY)
load_dotenv()

# Inicializa o cliente OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ============================================================
#  FUNÇÃO DE SIMILARIDADE
# ============================================================

def similaridade_cosseno(vetor_a, vetor_b):
    """
    Calcula a similaridade de cosseno entre dois vetores NumPy.

    Fórmula:
        sim(A, B) = (A · B) / (||A|| * ||B||)

    Retorna um valor entre -1 e 1:
        - 1  → vetores idênticos (mesmo significado)
        - 0  → sem relação
        - -1 → significados opostos
    """
    produto_escalar = np.dot(vetor_a, vetor_b)
    norma_a = np.linalg.norm(vetor_a)
    norma_b = np.linalg.norm(vetor_b)
    return produto_escalar / (norma_a * norma_b)

# ============================================================
#  TEXTOS DE TESTE
# ============================================================

frases = [
    "Um cachorro brincando no parque",
    "Um animal de estimação correndo na grama",
    "Um carro esportivo acelerando na pista"
]

print(" Gerando embeddings para as frases:")
for i, frase in enumerate(frases, start=1):
    print(f"   {i}. {frase}")

# ============================================================
#  GERAÇÃO DOS EMBEDDINGS
# ============================================================

resposta = client.embeddings.create(
    model="text-embedding-3-small",  # Modelo recomendado pela OpenAI
    input=frases
)

# Extrai os vetores
embeddings = [np.array(item.embedding) for item in resposta.data]

# ============================================================
#  CÁLCULO DAS SIMILARIDADES
# ============================================================

n = len(embeddings)
matriz_similaridade = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        matriz_similaridade[i, j] = similaridade_cosseno(embeddings[i], embeddings[j])

# ============================================================
#  EXIBIÇÃO DOS RESULTADOS
# ============================================================

print("\n🧮 Matriz de Similaridade (valores entre 0 e 1):\n")
for i in range(n):
    for j in range(n):
        print(f"{matriz_similaridade[i, j]:.3f}", end="\t")
    print()

# ============================================================
#  INTERPRETAÇÃO AUTOMÁTICA
# ============================================================

# Encontra o par de frases mais semelhantes (excluindo diagonais e evitando duplicats, já que sim(A,B)=sim(B,A))
max_val = -1
mais_proximo = (None, None)

for i in range(n):
    for j in range(i + 1, n): # Compara apenas a metade superior da matriz, ignorando diagonal
        if matriz_similaridade[i, j] > max_val:
            max_val = matriz_similaridade[i, j]
            mais_proximo = (i, j)

print("\n✅ Frases mais semanticamente semelhantes:")
print(f"  → \"{frases[mais_proximo[0]]}\"")
print(f"  → \"{frases[mais_proximo[1]]}\"")
print(f"  Similaridade: {max_val:.3f}")
