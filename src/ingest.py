import os
from utils import validate_envs
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector

load_dotenv()

def ingest_pdf() -> None:
    validate_envs()

    loadedPdf = _load_pdf()
    splitDocs = _split_pdf(loadedPdf)
    enrichedDocs = _enrich_documents(splitDocs)
    _save_on_db(enrichedDocs)

def _save_on_db(docs: list[Document]) -> None:
    try: 
        collectionName = os.getenv("PG_VECTOR_COLLECTION_NAME")
        dbConnection = os.getenv("DATABASE_URL")
        model = os.getenv("OPENAI_EMBEDDING_MODEL")

        embeddings = OpenAIEmbeddings(model=model)
        store = PGVector(
            embeddings=embeddings,
            collection_name=collectionName,
            connection=dbConnection,
            use_jsonb=True,
        )

        ids = [f"doc-{i}" for i in range(len(docs))]
        store.add_documents(documents=docs, ids=ids)
    except Exception as err:
        print(f"An unexpected error occurred: {err}")

def _enrich_documents(docs: list[Document]) -> list[Document]:
    return [
        Document(
            page_content=d.page_content,
            metadata={k: v for k, v in d.metadata.items() if v not in ("", None)}
        )
        for d in docs
    ]

def _split_pdf(pdf: list[Document]) -> list[Document]:
    split = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150, add_start_index=False).split_documents(pdf)
    if not split:
        print("Nothing to split.")
        raise SystemExit(0)
    
    return split

def _load_pdf() -> (list[Document] | None):
    try:
        pdfPath = os.getenv("PDF_PATH")
        return PyPDFLoader(pdfPath).load()
    except FileNotFoundError as err:
        print(f"PDF not found: {err}")

if __name__ == "__main__":
    ingest_pdf()