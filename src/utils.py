import os

def validate_envs() -> None:
    envKeys = ["OPENAI_API_KEY", "DATABASE_URL","PG_VECTOR_COLLECTION_NAME", "PDF_PATH", "OPENAI_EMBEDDING_MODEL"]
    for k in envKeys:
        if not os.getenv(k):
            raise RuntimeError(f"Missing environment variable {k}")