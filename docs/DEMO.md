# Local Demo

## Deterministic checks

```powershell
python -m unittest discover -s backend\banking -p "test_*unit.py"
python -m compileall backend rag_api RAG ml_pipeline archetype_data
```

The six research-dependent need-discovery tests are preserved under `tests\research_dependent` and are excluded from this essential suite. Their required research artifacts are documented in that directory.

## Django

```powershell
python backend\manage.py check
python backend\manage.py runserver
```

## RAG API

```powershell
uvicorn rag_api.main:app --reload --port 8001
```

The application is intended to run against synthetic/local demonstration resources. Do not point it at a production database or copy confidential artifacts into this repository.
