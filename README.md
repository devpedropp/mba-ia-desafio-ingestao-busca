# Desafio MBA Engenharia de Software com IA - Full Cycle

Crie e ative um ambiente virtual antes de instalar dependências:

```
python3 -m venv venv
source venv/bin/activate
```

## Ordem de execução
### 1. Subir o banco de dados:

```
docker compose up -d
```

### 2. Executar ingestão do PDF:

```
python3 src/ingest.py
```

### 3. Rodar o chat:

```
python3 src/chat.py
```