import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector

load_dotenv()

def ingest_pdf():
    _validate_envs()

    loadedPdf = _load_pdf()
    splitDocs = _split_pdf(loadedPdf)
    enrichedDocs = _enrich_documents(splitDocs)
    _save_on_db(enrichedDocs)

def _save_on_db(docs: list[Document]):
    try: 
        collectionName = os.getenv("PG_VECTOR_COLLECTION_NAME")
        dbConnection = os.getenv("DATABASE_URL")

        ids = [f"doc-{i}" for i in range(len(docs))]

        embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_MODEL","text-embedding-3-small"))

        store = PGVector(
            embeddings=embeddings,
            collection_name=collectionName,
            connection=dbConnection,
            use_jsonb=True,
        )

        store.add_documents(documents=docs, ids=ids)
    except Exception as err:
        print(f"An unexpected error occurred: {err}")

def _enrich_documents(docs: list[Document]):
    return [
        Document(
            page_content=d.page_content,
            metadata={k: v for k, v in d.metadata.items() if v not in ("", None)}
        )
        for d in docs
    ]

def _split_pdf(pdf: list[Document]):
    splitted = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150, add_start_index=False).split_documents(pdf)
    if not splitted:
        print("Nothing to split.")
        raise SystemExit(0)
    
    return splitted

def _load_pdf():
    try:
        pdfPath = os.getenv("PDF_PATH")
        return PyPDFLoader(pdfPath).load()
    except FileNotFoundError as err:
        print(f"PDF not found: {err}")
    

def _validate_envs():
    envKeys = ["OPENAI_API_KEY", "DATABASE_URL","PG_VECTOR_COLLECTION_NAME", "PDF_PATH"]
    for k in envKeys:
        if not os.getenv(k):
            raise RuntimeError(f"Missing environment variable {k}")

if __name__ == "__main__":
    ingest_pdf()