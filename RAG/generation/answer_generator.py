import logging
import re
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAG_DIR = ROOT / "RAG"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

try:
    from RAG import config
    from RAG.generation.context_builder import build_contex
except ModuleNotFoundError:
    from rag import config
    from rag.generation.context_builder import build_contex


LOGGER = logging.getLogger(__name__)
NO_EVIDENCE_ANSWER = "I could not find production-safe Atlas Demo Bank information supporting an answer to this question."
CITATION_RE = re.compile(r"\[S\d+\]")

PROMPT_INJECTION_RE = re.compile(
    r"\b(ignore|disregard|override|forget)\b|don't\s+tell\s+me|do\s+not\s+tell\s+me",
    re.I,
)
GENERIC_HIGH_RISK_RE = re.compile(
    r"\b(fee|fees|tariff|rate|rates|limit|limits|withdrawal|approval|days?|processing|documents?|required|estimate|guess)\b",
    re.I,
)


def _plain_url(value):
    text = " ".join(str(value or "").split())
    urls = re.findall(r"https?://[^\s)\]]+", text)
    return urls[-1] if urls else tex


def clean_model_text(value):
    text = " ".join(str(value or "").split())
    replacements = {
        "adhÃ©sion": "adhésion",
        "Ã©": "é",
        "Ã¨": "è",
        "Ã ": "à",
        "Ã‰": "É",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return tex


def _status_label(status):
    return {
        "verified": "verified",
        "partially_verified": "partially verified",
        "official_information_not_public": "not publicly documented",
        "human_intervention": "requires bank support",
    }.get(status or "", "unknown")


class OllamaLLMClient:
    def __init__(self, model=None, temperature=None, timeout=None):
        self.model = model or config.LLM_MODEL
        self.temperature = config.LLM_TEMPERATURE if temperature is None else temperature
        self.timeout = timeout or getattr(config, "LLM_TIMEOUT_SECONDS", 120)

    def generate(self, prompt):
        try:
            import ollama
        except ImportError as exc:
            raise RuntimeError(
                "Missing Ollama Python client. Install it with `python -m pip install ollama`."
            ) from exc

        try:
            client = ollama.Client(host=getattr(config, "OLLAMA_HOST", None), timeout=self.timeout)
            response = client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": self.temperature},
            )
        except Exception as exc:
            raise RuntimeError(
                f"Ollama generation failed. Check that Ollama is running and model `{self.model}` is installed."
            ) from exc

        try:
            return response["message"]["content"].strip()
        except Exception as exc:
            raise RuntimeError("Ollama returned an unexpected response shape.") from exc


def get_llm_client(model=None):
    if config.LLM_PROVIDER != "ollama":
        raise ValueError(f"Unsupported LLM_PROVIDER: {config.LLM_PROVIDER}")
    return OllamaLLMClient(model=model)


def build_grounded_prompt(query, context, retry=False):
    evidence_blocks = []
    for item in context.get("evidence", []):
        source_urls = [_plain_url(url) for url in item.get("source_urls", [])]
        source = source_urls[0] if source_urls else ""
        evidence_blocks.append(
            "\n".join([
                f"[{item['citation_id']}]",
                f"Product: {item.get('product', '')}",
                f"Category: {item.get('category', '')}",
                f"Retrieved question: {item.get('question', '')}",
                f"Retrieved answer: {item.get('answer') or item.get('content', '')}",
                f"Question type: {item.get('question_type', '')}",
                f"Evidence status: {_status_label(item.get('verification_status'))}",
                f"Confidence: {item.get('confidence', '')}",
                f"Verification notes: {item.get('verification_notes', '')}",
                f"Source: {source}",
            ])
        )

    retry_rules = ""
    if retry:
        retry_rules = (
            "\nSTRICT RETRY RULE: Your previous answer used missing or invalid citation IDs. "
            "Use only the citation IDs listed in PRODUCTION-SAFE EVIDENCE. "
            "Every factual sentence must end with a valid citation ID. "
            "If you cannot cite a claim with those IDs, do not make that claim.\n"
        )

    return f"""SYSTEM RULES
You are generating a grounded banking answer for Atlas Demo Bank DemoCityia.
Answer ONLY from PRODUCTION-SAFE EVIDENCE supplied below.
Do not use your own banking knowledge.
Do not infer unsupported facts.
Do not guess.
Never invent fees, limits, rates, eligibility conditions, required documents, processing times, procedures, guarantees, or product conditions.
Never silently substitute a requested Atlas Demo product with a different product because its evidence is semantically similar.
You may paraphrase, summarize, combine evidence, explain terminology, and make straightforward comparisons directly supported by the cited evidence.
Verification status policy:
- verified: answer normally from the evidence.
- partially_verified: use confirmed details, and explicitly qualify any detail that is not officially confirmed.
- official_information_not_public: do not fill gaps; say the specific information is not publicly provided or confirmed by Atlas Demo Bank, while using any confirmed context in the evidence.
- human_intervention: provide safe general guidance from the evidence and route account-specific, transaction-specific, authentication, dispute, application-status, or customer-data requests to bank/customer-service intervention.
When multiple evidence items overlap, semantic relevance comes first; use status only to qualify how strongly claims may be stated.
Never treat partially_verified, official_information_not_public, or human_intervention evidence as verified.
Do not show internal status labels such as "partially_verified", "official_information_not_public", or "human_intervention" to customers.
If the evidence only answers part of the question, answer only that part and clearly say the evidence does not provide the remaining information.
If the evidence describes only access, sign-up, eligibility, or availability, do not expand it into a general product description.
If the question asks how a product works but the evidence only explains access or sign-up, say: "The verified evidence only explains how access/sign-up is set up..." and cite it.
Never interpret absence of evidence as evidence that something is unavailable.
Answer the exact question asked, not merely a related topic.
Do not substitute eligibility information for a definition.
Do not substitute application instructions for required documents.
For comparisons, discuss every requested product represented in the question.
If verified evidence is missing for one requested comparison side, state that explicitly.
A statement that a product exists or is listed on official pages is not sufficient as a product definition when stronger evidence is required.
Keep the answer concise and customer-friendly.
Answer directly in a natural banking-assistant tone.
Do not use mechanical evidence-report phrases such as "The cited source confirms", "the evidence states", or "Details not confirmed by the cited evidence are not included."
Do not expose verification workflow, retrieval, or evaluation language to the customer.
Cite factual statements using only citation IDs provided below, such as [S1].
Do not cite source URLs directly in the answer; use citation IDs only.
{retry_rules}
USER QUESTION
{query}

PRODUCTION-SAFE EVIDENCE
{chr(10).join(evidence_blocks)}

OUTPUT REQUIREMENTS
Return only the customer-facing answer.
Every factual sentence must include at least one citation ID.
Prefer one or two short sentences. Do not add uncited advice such as contacting the bank unless that advice appears in the evidence.
Use only these citation IDs: {", ".join(item["citation_id"] for item in context.get("evidence", []))}.
If the evidence is insufficient, say so directly and cite the evidence that is available.
"""


def extract_citation_ids(answer):
    return {match.strip("[]") for match in CITATION_RE.findall(answer or "")}


def validate_citations(answer, allowed_ids):
    answer = clean_model_text(answer)
    answer_for_sentence_check = re.sub(r"([.!?])\s*((?:\[S\d+\]\s*)+)", r" \2\1", answer)
    used_ids = extract_citation_ids(answer)
    invalid_ids = sorted(used_ids - set(allowed_ids))
    uncited_sentences = []
    for sentence in re.split(r"(?<=[.!?])\s+", answer_for_sentence_check):
        sentence = sentence.strip()
        if not sentence:
            continue
        if sentence.lower().startswith(("however", "the verified evidence does not", "verified evidence does not")):
            continue
        if not CITATION_RE.search(sentence):
            uncited_sentences.append(sentence)
    return {
        "passed": bool(used_ids) and not invalid_ids and not uncited_sentences,
        "used_ids": sorted(used_ids),
        "invalid_ids": invalid_ids,
        "uncited_sentences": uncited_sentences,
    }


def citations_for_used_ids(context, used_ids):
    by_id = {item["citation_id"]: item for item in context.get("evidence", [])}
    citations = []
    for citation_id in used_ids:
        item = by_id.get(citation_id)
        if not item:
            continue
        source_urls = [_plain_url(url) for url in item.get("source_urls", [])]
        source_titles = item.get("source_titles") or []
        citations.append({
            "citation_id": citation_id,
            "document_id": item.get("document_id"),
            "source_title": source_titles[0] if source_titles else "",
            "source_url": source_urls[0] if source_urls else "",
        })
    return citations


def cite_each_sentence(text, citation_id):
    text = clean_model_text(text)
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
    cited = []
    for sentence in sentences:
        if CITATION_RE.search(sentence):
            cited.append(sentence)
        else:
            cited.append(f"{sentence} [{citation_id}]")
    return " ".join(cited)


def _public_gap_phrase(item):
    product = clean_model_text(item.get("product") or "this product")
    question_type = clean_model_text(item.get("question_type") or "detail")
    return f"Atlas Demo Bank does not publicly document the specific {question_type} requested for {product}."


def deterministic_evidence_answer(query, context):
    evidence = context.get("evidence", [])
    if not evidence:
        return None
    analysis = context.get("analysis") or {}
    if (
        len(evidence) > 1
        and (analysis.get("is_comparison") or analysis.get("is_multi_part") or len(analysis.get("mentioned_products") or []) > 1)
    ):
        parts = []
        for item in evidence:
            citation_id = item["citation_id"]
            product = clean_model_text(item.get("product") or "Relevant evidence")
            content = cite_each_sentence(item.get("answer") or item.get("content", ""), citation_id)
            if content:
                parts.append(f"{product}: {content}")
        return " ".join(parts) if parts else None
    item = evidence[0]
    citation_id = item["citation_id"]
    content = cite_each_sentence(item.get("content", ""), citation_id)
    query_l = (query or "").lower()
    status = item.get("verification_status")
    if status == "partially_verified":
        answer = f"The cited source confirms: {content} Details not confirmed by the cited evidence are not included. [{citation_id}]"
    elif status == "official_information_not_public":
        answer = f"{_public_gap_phrase(item)} [{citation_id}] Useful confirmed context: {content} For the exact requirement or procedure, contact an Atlas Demo Bank advisor or branch. [{citation_id}]"
    elif status == "human_intervention":
        answer = f"Use this general guidance: {content} Because this may depend on your account, card, device, transaction, application, or personal banking data, contact Atlas Demo Bank support or your branch if it continues. [{citation_id}]"
    elif "how" in query_l and "cash management" in query_l:
        answer = (
            "The verified evidence only explains how access/sign-up is set up for Atlas Demo Cash Management: "
            f"{content}"
        )
    elif "carte gold" in query_l and "carte gold" not in content.lower():
        answer = f"Carte Gold can refer to Atlas Demo Gold card options. {content}"
    else:
        answer = conten
    return answer


def no_evidence_response(query, reason, sources=None):
    if reason == "required_documents_not_specified":
        answer = (
            "The verified information available to me does not specify the required documents for opening an account. "
            "Please confirm the current document requirements with Atlas Demo Bank or an advisor."
        )
    elif reason in {"comparison_missing_product_evidence", "requested_product_missing_evidence"}:
        answer = "I do not have verified Atlas Demo information for every requested product, so I cannot safely compare or describe them as if all were covered."
    else:
        answer = "I do not have verified Atlas Demo information for that specific detail."
    return {
        "query": query,
        "answered": False,
        "answer": answer,
        "reason": reason,
        "citations": [],
        "sources": sources or [],
    }


def should_refuse_query_before_retrieval(query, product=None):
    query = query or ""
    if PROMPT_INJECTION_RE.search(query):
        return "prompt_injection_or_instruction_override_request"

    return None


def should_refuse_before_llm(query, context, product=None):
    query = query or ""
    query_refusal = should_refuse_query_before_retrieval(query, product=product)
    if query_refusal:
        return query_refusal

    return None


def _is_exact_lookup(query, context):
    analysis = context.get("analysis") or {}
    query_l = (query or "").lower().strip(" ?!.")
    return (
        analysis.get("question_type") == "factual"
        and not analysis.get("is_comparison")
        and not analysis.get("is_multi_part")
        and len(analysis.get("mentioned_products") or []) == 1
        and bool(re.match(r"^(what is|what does|tell me about|define)\b", query_l))
    )


def generation_failure_response(query, reason, model=None, latency_ms=None):
    response = {
        "query": query,
        "answered": False,
        "answer": "I found verified evidence, but could not safely generate a cited answer from it.",
        "reason": reason,
        "citations": [],
        "sources": [],
        "model": model or config.LLM_MODEL,
    }
    if latency_ms is not None:
        response["generation_latency_ms"] = latency_ms
    return response


def _status_aware_fallback_response(query, context, reason, latency_ms, total_start, context_latency_ms, model=None):
    allowed_ids = [item["citation_id"] for item in context.get("evidence", [])]
    answer = deterministic_evidence_answer(query, context)
    validation = validate_citations(answer, allowed_ids) if answer else {"passed": False}
    if not validation["passed"]:
        response = generation_failure_response(query, reason, model=model, latency_ms=latency_ms)
        response.update({
            "context_latency_ms": context_latency_ms,
            "total_rag_latency_ms": (time.perf_counter() - total_start) * 1000,
            "has_evidence": True,
            "evidence_document_ids": [item.get("document_id") for item in context.get("evidence", [])],
            "llm_called": True,
        })
        return response

    citations = citations_for_used_ids(context, validation["used_ids"])
    return {
        "query": query,
        "answered": True,
        "answer": answer,
        "citations": citations,
        "sources": citations,
        "evidence_count": len(context.get("evidence", [])),
        "model": model or config.LLM_MODEL,
        "generation_latency_ms": latency_ms,
        "context_latency_ms": context_latency_ms,
        "total_rag_latency_ms": (time.perf_counter() - total_start) * 1000,
        "has_evidence": True,
        "evidence_document_ids": [item.get("document_id") for item in context.get("evidence", [])],
        "llm_called": True,
        "citation_validation_passed": True,
        "deterministic_fallback_used": True,
        "fallback_reason": reason,
    }


def generate_answer(
    query: str,
    top_k: int = 3,
    product: str | None = None,
    category: str | None = None,
    model: str | None = None,
):
    selected_model = model or config.LLM_MODEL
    total_start = time.perf_counter()
    pre_retrieval_refusal = should_refuse_query_before_retrieval(query, product=product)
    if pre_retrieval_refusal:
        LOGGER.info("generation_refused_before_retrieval query=%r reason=%s", query, pre_retrieval_refusal)
        response = no_evidence_response(query, pre_retrieval_refusal, [])
        response.update({
            "has_evidence": False,
            "evidence_document_ids": [],
            "context_latency_ms": 0.0,
            "total_rag_latency_ms": (time.perf_counter() - total_start) * 1000,
            "llm_called": False,
        })
        return response

    context_start = time.perf_counter()
    context = build_context(query=query, top_k=top_k, product=product, category=category)
    context_latency_ms = (time.perf_counter() - context_start) * 1000
    evidence_document_ids = [item.get("document_id") for item in context.get("evidence", [])]
    document_ids = [item.get("document_id") for item in context.get("evidence", [])]
    LOGGER.info(
        "generation_request query=%r has_evidence=%s document_ids=%s model=%s",
        query,
        context.get("has_evidence"),
        document_ids,
        selected_model,
    )

    if not context.get("has_evidence"):
        LOGGER.info("generation_refused_no_evidence query=%r reason=%s", query, context.get("reason"))
        response = no_evidence_response(query, context.get("reason", "no_trusted_verified_evidence"), context.get("sources", []))
        response.update({
            "has_evidence": False,
            "evidence_document_ids": evidence_document_ids,
            "context_latency_ms": context_latency_ms,
            "total_rag_latency_ms": (time.perf_counter() - total_start) * 1000,
            "llm_called": False,
        })
        return response

    pre_llm_refusal = should_refuse_before_llm(query, context, product=product)
    if pre_llm_refusal:
        LOGGER.info("generation_refused_pre_llm query=%r reason=%s", query, pre_llm_refusal)
        response = no_evidence_response(query, pre_llm_refusal, [])
        response.update({
            "has_evidence": True,
            "evidence_document_ids": evidence_document_ids,
            "context_latency_ms": context_latency_ms,
            "total_rag_latency_ms": (time.perf_counter() - total_start) * 1000,
            "llm_called": False,
        })
        return response

    allowed_ids = [item["citation_id"] for item in context.get("evidence", [])]
    analysis = context.get("analysis") or {}
    fast_path_allowed = (
        getattr(config, "DETERMINISTIC_SINGLE_EVIDENCE_FAST_PATH", True)
        and len(context.get("evidence", [])) == 1
        and _is_exact_lookup(query, context)
        and not analysis.get("is_comparison")
        and not analysis.get("is_multi_part")
        and len(analysis.get("mentioned_products") or []) <= 1
        and (context.get("adequacy") or {}).get("adequate", True)
    )
    if fast_path_allowed:
        start = time.perf_counter()
        answer = deterministic_evidence_answer(query, context)
        validation = validate_citations(answer, allowed_ids)
        latency_ms = (time.perf_counter() - start) * 1000
        if validation["passed"]:
            citations = citations_for_used_ids(context, validation["used_ids"])
            LOGGER.info(
                "generation_success_deterministic_fast_path query=%r citations=%s latency_ms=%.2f",
                query,
                validation["used_ids"],
                latency_ms,
            )
            return {
                "query": query,
                "answered": True,
                "answer": answer,
                "citations": citations,
                "sources": citations,
                "evidence_count": len(context.get("evidence", [])),
                "model": selected_model,
                "generation_latency_ms": latency_ms,
                "context_latency_ms": context_latency_ms,
                "total_rag_latency_ms": (time.perf_counter() - total_start) * 1000,
                "has_evidence": True,
                "evidence_document_ids": evidence_document_ids,
                "llm_called": False,
                "citation_validation_passed": True,
                "deterministic_fast_path_used": True,
            }

    client = get_llm_client(model=selected_model)
    start = time.perf_counter()

    try:
        answer = clean_model_text(client.generate(build_grounded_prompt(query, context)))
    except RuntimeError as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        LOGGER.info("generation_failure query=%r error=%s latency_ms=%.2f", query, exc, latency_ms)
        return _status_aware_fallback_response(
            query=query,
            context=context,
            reason=str(exc),
            latency_ms=latency_ms,
            total_start=total_start,
            context_latency_ms=context_latency_ms,
            model=selected_model,
        )

    validation = validate_citations(answer, allowed_ids)
    retries_remaining = getattr(config, "LLM_MAX_CITATION_RETRIES", 0)
    while not validation["passed"] and retries_remaining > 0:
        retries_remaining -= 1
        LOGGER.info(
            "citation_validation_retry query=%r used=%s invalid=%s",
            query,
            validation["used_ids"],
            validation["invalid_ids"],
        )
        try:
            answer = clean_model_text(client.generate(build_grounded_prompt(query, context, retry=True)))
        except RuntimeError as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            response = generation_failure_response(query, str(exc), model=selected_model, latency_ms=latency_ms)
            response.update({
                "context_latency_ms": context_latency_ms,
                "total_rag_latency_ms": (time.perf_counter() - total_start) * 1000,
                "has_evidence": True,
                "evidence_document_ids": evidence_document_ids,
                "llm_called": True,
            })
            return response
        validation = validate_citations(answer, allowed_ids)

    latency_ms = (time.perf_counter() - start) * 1000
    if not validation["passed"]:
        LOGGER.info(
            "generation_failure_invalid_citations query=%r used=%s invalid=%s latency_ms=%.2f",
            query,
            validation["used_ids"],
            validation["invalid_ids"],
            latency_ms,
        )
        fallback_answer = deterministic_evidence_answer(query, context)
        fallback_validation = validate_citations(fallback_answer, allowed_ids) if fallback_answer else {"passed": False}
        if not fallback_validation["passed"]:
            response = generation_failure_response(
                query,
                "invalid_or_missing_citations",
                model=selected_model,
                latency_ms=latency_ms,
            )
            response.update({
                "context_latency_ms": context_latency_ms,
                "total_rag_latency_ms": (time.perf_counter() - total_start) * 1000,
                "has_evidence": True,
                "evidence_document_ids": evidence_document_ids,
                "llm_called": True,
            })
            return response
        used_ids = fallback_validation["used_ids"]
        return {
            "query": query,
            "answered": True,
            "answer": fallback_answer,
            "citations": citations_for_used_ids(context, used_ids),
            "sources": citations_for_used_ids(context, used_ids),
            "evidence_count": len(context.get("evidence", [])),
            "model": selected_model,
            "generation_latency_ms": latency_ms,
            "context_latency_ms": context_latency_ms,
            "total_rag_latency_ms": (time.perf_counter() - total_start) * 1000,
            "has_evidence": True,
            "evidence_document_ids": evidence_document_ids,
            "llm_called": True,
            "citation_validation_passed": True,
            "deterministic_fallback_used": True,
        }

    citations = citations_for_used_ids(context, validation["used_ids"])
    LOGGER.info(
        "generation_success query=%r citations=%s latency_ms=%.2f",
        query,
        validation["used_ids"],
        latency_ms,
    )
    return {
        "query": query,
        "answered": True,
        "answer": answer,
        "citations": citations,
        "sources": citations,
        "evidence_count": len(context.get("evidence", [])),
        "model": selected_model,
        "generation_latency_ms": latency_ms,
        "context_latency_ms": context_latency_ms,
        "total_rag_latency_ms": (time.perf_counter() - total_start) * 1000,
        "has_evidence": True,
        "evidence_document_ids": evidence_document_ids,
        "llm_called": True,
        "citation_validation_passed": True,
    }
