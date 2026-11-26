# Architecture Overview

## Data and modeling

- `archetype_data/data_simulation`: deterministic synthetic customer, transaction, and ownership generation.
- `archetype_data/static_clustering`: behavioral segmentation implementation and evaluation helpers.
- `archetype_data/ts`: temporal feature construction and forecasting workflows.
- `archetype_data/propensity`: feature contracts, training, evaluation, and inference.
- `archetype_data/recommendation`: ontology-aware hybrid candidate generation, eligibility, conflict handling, scoring, and explanations.

## Application

- `backend`: Django dashboard, APIs, authentication, customer-context isolation, and LangGraph orchestration.
- `rag_api`: FastAPI retrieval service boundary.
- `RAG`: retrieval, grounding, context construction, and evaluation code.
- `ml_pipeline`: bounded workflow entry points and artifact promotion logic.

## Safety boundaries

Customer context is server-side identity-bound. Public RAG retrieval is isolated from private customer context. Unsupported actions are routed to an escalation response. The portfolio edition contains no operational database loader or private ownership snapshot.
