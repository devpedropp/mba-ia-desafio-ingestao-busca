import os
from utils import validate_envs
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

def _search_context(pergunta) -> list:
  validate_envs()
  
  model = os.getenv("OPENAI_EMBEDDING_MODEL","text-embedding-3-small")
  collection_name = os.getenv("PG_VECTOR_COLLECTION_NAME")
  connection=os.getenv("DATABASE_URL")
  embeddings = OpenAIEmbeddings(model=model)

  store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
  )

  rawResult = store.similarity_search(pergunta, k=10)

  result = []
  for doc in rawResult:
     result.append(doc.page_content)

  return result

def search_prompt(pergunta=None) -> str:
  contexto = _search_context(pergunta)
  return PromptTemplate.from_template(PROMPT_TEMPLATE).format(contexto, pergunta)

 