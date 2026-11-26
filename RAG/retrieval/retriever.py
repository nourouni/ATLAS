import json
import importlib.util
import re
import sys
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAG_DIR = ROOT / "RAG"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

try:
    from RAG import config
    from RAG.retrieval.bm25_retriever import BM25Retriever
    from RAG.retrieval.common import PRODUCTION_STATUSES, clean, load_verified_documents
    from RAG.retrieval.dense_retriever import DenseRetriever
    from RAG.retrieval.hybrid_retriever import HybridRetriever
    from RAG.retrieval.reranking_retriever import RerankingRetriever
except ModuleNotFoundError:
    from rag import config
    from rag.retrieval.bm25_retriever import BM25Retriever
    from rag.retrieval.common import PRODUCTION_STATUSES, clean, load_verified_documents
    from rag.retrieval.dense_retriever import DenseRetriever
    from rag.retrieval.hybrid_retriever import HybridRetriever
    from rag.retrieval.reranking_retriever import RerankingRetriever


_RETRIEVER = None
_PRODUCT_GAZETTEER = None
_PRODUCTION_PRODUCTS = None
_DOCUMENTS_BY_ID = None


STOPWORDS = {
    "what", "does", "with", "from", "that", "this", "bank", "atlas_demo", "about",
    "should", "customer", "know", "difference", "between", "used", "use", "can",
    "how", "work", "works", "the", "and", "for", "outside", "abroad", "card",
    "carte", "visa", "gold",
}

QUESTION_TYPE_ALIASES = {
    "definition": "factual",
    "factual": "factual",
    "procedure": "procedure",
    "eligibility": "eligibility",
    "comparison": "comparison",
    "fees_limits": "fees_limits",
    "required_documents": "required_documents",
}

COMPATIBLE_QUESTION_TYPES = {
    "factual": {"factual", "recommendation"},
    "procedure": {"procedure", "security"},
    "eligibility": {"eligibility"},
    "comparison": {"comparison", "recommendation"},
    "fees_limits": {"factual"},
    "required_documents": {"procedure", "factual"},
}

WEAK_RECORD_RE = re.compile(r"\b(listed on .*official|officially list|still available|does atlas_demo bank officially list)\b", re.I)
FEE_RE = re.compile(r"\b(fee|fees|tariff|price|cost|cotisation|subscription|limit|limits|ceiling|plafond)\b", re.I)
INTERNAL_PLACEHOLDER_RE = re.compile(
    r"\b(No official supporting URL available|Related product not found in local structured product catalog|"
    r"do not repeatedly web-verify|specific factual requested for GENERAL)\b",
    re.I,
)

UNSUPPORTED_TOPIC_PATTERNS = []


def _json_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return [value] if value.startswith("http") else []
    return parsed if isinstance(parsed, list) else []


def _normalize_url(value):
    text = clean(value)
    if not text:
        return ""
    urls = re.findall(r"https?://[^\s)\]]+", text)
    return urls[-1] if urls else tex


def _match_key(value):
    text = clean(value).lower()
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def _json_url_list(value):
    urls = []
    for item in _json_list(value):
        url = _normalize_url(item)
        if url and url not in urls:
            urls.append(url)
    return urls


def _load_production_products():
    return {
        clean(document.get("product"))
        for document in load_verified_documents()
        if clean(document.get("product"))
    }


def _load_documents_by_id():
    global _DOCUMENTS_BY_ID
    if _DOCUMENTS_BY_ID is not None:
        return _DOCUMENTS_BY_ID
    documents = {}
    for document in load_verified_documents():
        if clean(document.get("id")):
            documents[clean(document.get("id"))] = documen
    _DOCUMENTS_BY_ID = documents
    return documents


def _load_kb_products():
    products = set()
    kb_path = Path(config.VERIFIED_KNOWLEDGE_PATH).with_name("kb_extended.py")
    if not kb_path.exists():
        return products
    spec = importlib.util.spec_from_file_location("kb_extended_for_retriever", kb_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for collection_name in ("PRODUCTS", "EXTENDED_PRODUCTS"):
        for product in getattr(module, collection_name, {}):
            if clean(product):
                products.add(clean(product))
    return products


def _aliases_for_product(product):
    product = clean(product)
    aliases = {product.lower(), _match_key(product)}
    lowered = product.lower()
    lowered_key = _match_key(product)
    for prefix in ("carte ", "pack ", "offre "):
        if lowered.startswith(prefix):
            aliases.add(lowered.removeprefix(prefix))
        if lowered_key.startswith(prefix):
            aliases.add(lowered_key.removeprefix(prefix))
    if "atlas_demo " in lowered:
        aliases.add(lowered.replace("atlas_demo ", ""))
    if "atlas_demo " in lowered_key:
        aliases.add(lowered_key.replace("atlas_demo ", ""))
    if lowered == "carte visa nationale":
        aliases.add(lowered.replace("carte ", ""))
        aliases.add(_match_key(lowered.replace("carte ", "")))
    return {alias for alias in aliases if len(alias) >= 5}


def _product_gazetteer():
    global _PRODUCT_GAZETTEER, _PRODUCTION_PRODUCTS
    if _PRODUCT_GAZETTEER is not None:
        return _PRODUCT_GAZETTEER, _PRODUCTION_PRODUCTS
    _PRODUCTION_PRODUCTS = _load_production_products()
    all_products = _PRODUCTION_PRODUCTS | _load_kb_products()
    gazetteer = []
    for product in sorted(all_products, key=len, reverse=True):
        for alias in _aliases_for_product(product):
            gazetteer.append((alias, product))
    _PRODUCT_GAZETTEER = gazetteer
    return _PRODUCT_GAZETTEER, _PRODUCTION_PRODUCTS


def _mentioned_products(query):
    query_l = clean(query).lower()
    query_key = _match_key(query)
    gazetteer, _ = _product_gazetteer()
    mentions = []
    for alias, product in gazetteer:
        if (alias in query_l or alias in query_key) and product not in mentions:
            mentions.append(product)
    if "carte gold" in query_l or re.search(r"\bgold\b", query_l):
        if "Carte Visa Nationale" not in mentions:
            mentions.append("Carte Visa Nationale")
    return mentions


def _product_aliases(product):
    aliases = _aliases_for_product(product)
    lowered = clean(product).lower()
    if lowered == "carte visa nationale":
        aliases.add("carte gold")
        aliases.add("gold")
        aliases.add(lowered.replace("carte ", ""))
        aliases.add(_match_key(lowered.replace("carte ", "")))
        aliases.add("visa nationale")
    return aliases


def analyze_public_question(query):
    query = clean(query)
    query_l = query.lower()
    retrieval_query = query
    comparison = bool(re.search(r"\b(compare|difference between|versus|vs\.?)\b", query_l))
    if comparison:
        question_type = "comparison"
    elif re.search(r"\b(document|documents|paperwork|required docs|required documents)\b", query_l):
        question_type = "required_documents"
    elif re.search(r"\b(lost|stolen|forgot.*pin|replace.*card|card.*opposition|opposition)\b", query_l):
        question_type = "procedure"
    elif re.search(r"\b(fee|fees|tariff|price|cost|cotisation|subscription|limit|limits|ceiling|plafond)\b", query_l):
        question_type = "fees_limits"
    elif re.search(r"\b(who can|eligible|eligibility|qualify|conditions?|requirements?)\b", query_l):
        question_type = "eligibility"
    elif re.search(r"\b(how|apply|open|request|subscribe|sign up|process|procedure|steps|order)\b", query_l):
        question_type = "procedure"
    else:
        question_type = "factual"
    products = _mentioned_products(query)
    if re.fullmatch(r"(what is |tell me more about |more about )?carte gold\??", query_l):
        retrieval_query = "Carte Visa Nationale"
        question_type = "factual"
        comparison = False
    category_hint = ""
    if re.search(r"\b(cards?|carte|cib|gold|visa|platinum)\b", query_l):
        category_hint = "cards"
    elif re.search(r"\b(loans?|credit|financing|mortgage|financement)\b", query_l):
        category_hint = "loans_financing"
    elif re.search(r"\b(account|compte|open|opening|sign up|signup|register|join)\b", query_l):
        category_hint = "accounts"
    elif re.search(r"\b(invest|investment|placements?|savings?|saving)\b", query_l):
        category_hint = "investments_savings"
    elif re.search(r"\b(business|company|corporate|sme|merchant|enterprise|entreprise)\b", query_l):
        category_hint = "business"
    elif re.search(r"\b(products?|offer|available|range|catalog)\b", query_l):
        category_hint = "products"
    return {
        "question_type": question_type,
        "mentioned_products": products,
        "category_hint": category_hint,
        "is_comparison": comparison or len(products) > 1 and re.search(r"\b(and|/|or)\b", query_l),
        "is_multi_part": query.count("?") > 1 or bool(re.search(r"\b(and also|also|both)\b", query_l)),
        "retrieval_query": retrieval_query,
    }


def _query_tokens(text):
    raw = re.findall(r"[a-zA-ZÀ-ÿ0-9]+", clean(text).lower())
    return {token for token in raw if len(token) > 2 and token not in STOPWORDS}


def _has_lexical_overlap(query, result):
    query_terms = _query_tokens(query)
    if not query_terms:
        return True
    result_text = " ".join([
        clean(result.get("document")),
        clean((result.get("metadata") or {}).get("product")),
        clean((result.get("metadata") or {}).get("category")),
    ])
    return len(query_terms & _query_tokens(result_text)) >= getattr(config, "MIN_LEXICAL_OVERLAP", 1)


def _preflight(query, product=None):
    _, production_products = _product_gazetteer()
    mentioned = [clean(product)] if product else _mentioned_products(query)
    missing = [item for item in mentioned if item not in production_products]
    if missing:
        return False, "no_production_safe_documents_for_product: " + ", ".join(missing), mentioned
    if not mentioned:
        for topic, pattern in UNSUPPORTED_TOPIC_PATTERNS:
            if pattern.search(query):
                return False, f"no_verified_documents_for_topic: {topic}", mentioned
    return True, "", mentioned


def _build_backend():
    retriever_type = config.RETRIEVER_TYPE
    candidate_k = getattr(config, "CANDIDATE_K", 20)
    if retriever_type == "dense":
        return DenseRetriever()
    if retriever_type == "bm25":
        return BM25Retriever()
    if retriever_type == "hybrid_rrf":
        dense = DenseRetriever()
        bm25 = BM25Retriever()
        return HybridRetriever(
            dense_retriever=dense,
            bm25_retriever=bm25,
            rrf_k=getattr(config, "RRF_K", 60),
            candidate_k=candidate_k,
        )
    if retriever_type == "dense_cross_encoder_rerank":
        return RerankingRetriever(candidate_k=candidate_k)
    raise ValueError(f"Unsupported RETRIEVER_TYPE: {retriever_type}")


def get_retriever():
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = _build_backend()
    return _RETRIEVER


def _passes_filters(result, product=None, category=None):
    metadata = result.get("metadata") or {}
    if clean(metadata.get("verification_status")) not in PRODUCTION_STATUSES:
        return False
    if product and clean(metadata.get("product")).lower() != clean(product).lower():
        return False
    if category and clean(metadata.get("category")).lower() != clean(category).lower():
        return False
    return True


def _result_product_matches(result_product, mentioned_products):
    if not mentioned_products:
        return True
    result_l = clean(result_product).lower()
    result_key = _match_key(result_product)
    for product in mentioned_products:
        product_l = clean(product).lower()
        product_key = _match_key(product)
        if product_l == result_l or product_key == result_key:
            return True
        if product_l in result_l or result_l in product_l or product_key in result_key or result_key in product_key:
            return True
    return False


def _metadata(result):
    return result.get("metadata") or {}


def _canonical_result(result):
    return _load_documents_by_id().get(clean(result.get("document_id")), {})


def _result_text(result):
    canonical = _canonical_result(result)
    return " ".join(
        clean(value)
        for value in [
            result.get("document"),
            canonical.get("question"),
            canonical.get("answer"),
            canonical.get("content"),
            (result.get("metadata") or {}).get("question_type"),
        ]
    )


def _is_useful_result(result):
    canonical = _canonical_result(result)
    content = clean(canonical.get("answer") or canonical.get("content") or result.get("document"))
    if INTERNAL_PLACEHOLDER_RE.search(content):
        return False
    return bool(content and content.lower() != "answer:")


def _category_matches_hint(category, hint):
    category_l = clean(category).lower()
    if not hint:
        return False
    if hint == "cards":
        return "card" in category_l or "carte" in category_l
    if hint == "loans_financing":
        return any(term in category_l for term in ("loan", "credit", "financ"))
    if hint == "accounts":
        return "account" in category_l or "compte" in category_l or "services digitaux" in category_l
    if hint == "investments_savings":
        return any(term in category_l for term in ("placement", "saving", "account"))
    if hint == "business":
        return any(term in category_l for term in ("business", "corporate", "sme", "financ", "cash", "cards", "package"))
    if hint == "products":
        return bool(category_l) and category_l not in {"fees & limits"}
    return False


def _is_broad_category_query(analysis):
    return bool(analysis.get("category_hint")) and not analysis.get("mentioned_products")


def _question_type_matches(requested, actual, text):
    actual = QUESTION_TYPE_ALIASES.get(clean(actual).lower(), clean(actual).lower())
    requested = QUESTION_TYPE_ALIASES.get(clean(requested).lower(), clean(requested).lower())
    if requested == "fees_limits":
        return actual == "factual" and bool(FEE_RE.search(text))
    if requested == "required_documents":
        return bool(re.search(r"\b(document|documents|paperwork|required)\b", text, re.I))
    return actual in COMPATIBLE_QUESTION_TYPES.get(requested, {requested})


def _covers_all_products(result, products):
    if not products:
        return True
    metadata = result.get("metadata") or {}
    canonical = _canonical_result(result)
    text = _match_key(" ".join([
        clean(metadata.get("product") or result.get("product")),
        clean(canonical.get("product")),
        clean(canonical.get("question")),
    ]))
    return all(any(alias in text for alias in _product_aliases(product)) for product in products)


def _rerank_results(results, analysis):
    reranked = []
    for result in results:
        metadata = result.get("metadata") or {}
        text = _result_text(result)
        adjusted = float(result.get("score") or 0.0)
        if _question_type_matches(analysis["question_type"], metadata.get("question_type"), text):
            adjusted += getattr(config, "QUESTION_TYPE_MATCH_BOOST", 0.035)
        else:
            adjusted -= getattr(config, "QUESTION_TYPE_MISMATCH_PENALTY", 0.012)
        if WEAK_RECORD_RE.search(text):
            adjusted -= getattr(config, "WEAK_RECORD_PENALTY", 0.08)
        if INTERNAL_PLACEHOLDER_RE.search(text):
            adjusted -= 0.20
        if _category_matches_hint(metadata.get("category"), analysis.get("category_hint")):
            adjusted += 0.045
        elif _is_broad_category_query(analysis):
            adjusted -= 0.035
        if analysis.get("category_hint") == "accounts" and clean(metadata.get("product")).lower() == "webank":
            adjusted += 0.04
        if analysis.get("mentioned_products") and not analysis.get("is_comparison"):
            product = clean(metadata.get("product") or result.get("product"))
            if any(_match_key(product) == _match_key(item) for item in analysis["mentioned_products"]):
                adjusted += 0.12
        if _is_broad_category_query(analysis) and re.search(r"\b(what are|list|types?|options?|products?)\b", clean(analysis.get("query", "")).lower()):
            adjusted += 0.01
        if analysis["is_comparison"] and metadata.get("question_type") == "comparison" and _covers_all_products(result, analysis["mentioned_products"]):
            adjusted += getattr(config, "COMPARISON_MATCH_BOOST", 0.05) + 0.14
        elif analysis["is_comparison"] and metadata.get("question_type") == "comparison" and len(analysis.get("mentioned_products") or []) >= 2:
            adjusted -= getattr(config, "COMPARISON_PARTIAL_MATCH_PENALTY", 0.20)
        if (
            analysis["question_type"] == "factual"
            and clean(metadata.get("question_type")).lower() == "comparison"
            and len(analysis.get("mentioned_products") or []) >= 2
            and _covers_all_products(result, analysis["mentioned_products"])
        ):
            adjusted += getattr(config, "FACTUAL_FAMILY_COMPARISON_BOOST", 0.16)
        question_text = " ".join([
            clean(_canonical_result(result).get("question")),
            clean(metadata.get("question")),
        ])
        if analysis["question_type"] == "factual" and FEE_RE.search(question_text):
            adjusted -= getattr(config, "FACTUAL_FEE_RECORD_PENALTY", 0.10)
        result = dict(result)
        result["score"] = adjusted
        reranked.append(result)
    return sorted(reranked, key=lambda item: item.get("score") or 0.0, reverse=True)


def _is_near_duplicate(result, seen_texts):
    text = clean(result.get("document")).lower()
    if not text:
        return False
    head = text[:280]
    if head in seen_texts:
        return True
    seen_texts.add(head)
    return False


def _diversify_products(results, analysis):
    if not _is_broad_category_query(analysis):
        return results
    selected = []
    deferred = []
    seen_products = set()
    for result in results:
        product = clean((result.get("metadata") or {}).get("product") or result.get("product"))
        if product and product not in seen_products:
            selected.append(result)
            seen_products.add(product)
        else:
            deferred.append(result)
    return selected + deferred


def _diversify_requested_products(results, analysis):
    products = analysis.get("mentioned_products") or []
    if len(products) < 2:
        return results
    selected = []
    selected_ids = set()
    for product in products:
        for result in results:
            document_id = result.get("document_id")
            if document_id in selected_ids:
                continue
            result_product = clean((result.get("metadata") or {}).get("product") or result.get("product"))
            if _result_product_matches(result_product, [product]):
                selected.append(result)
                selected_ids.add(document_id)
                break
    return selected + [result for result in results if result.get("document_id") not in selected_ids]


def _format_result(rank, result):
    metadata = result.get("metadata") or {}
    return {
        "rank": rank,
        "document_id": result.get("document_id"),
        "content": result.get("document"),
        "product": clean(metadata.get("product")),
        "question_id": clean(metadata.get("question_id") or result.get("document_id")),
        "category": clean(metadata.get("category")),
        "subcategory": clean(metadata.get("subcategory")),
        "question_type": clean(metadata.get("question_type")),
        "score": result.get("score"),
        "source_urls": _json_url_list(metadata.get("source_urls")),
        "source_titles": _json_list(metadata.get("source_titles")),
        "verification_status": clean(metadata.get("verification_status")),
        "confidence": clean(metadata.get("confidence")),
        "verification_notes": clean(metadata.get("verification_notes")),
        "original_rank": result.get("original_rank"),
        "original_score": result.get("original_score"),
    }


def retrieve(query: str, top_k: int = None, product: str | None = None, category: str | None = None):
    query = clean(query)
    if not query:
        raise ValueError("query must not be empty")

    requested_top_k = top_k or getattr(config, "TOP_K", 5)
    analysis = analyze_public_question(query)
    analysis["query"] = query
    allowed, reason, mentioned_products = _preflight(query, product=product)
    if not allowed:
        return {
            "query": query,
            "retriever_type": config.RETRIEVER_TYPE,
            "results": [],
            "no_results_reason": reason,
            "mentioned_products": mentioned_products,
        }

    candidate_floor = 80 if _is_broad_category_query(analysis) or len(analysis.get("mentioned_products") or []) >= 2 else getattr(config, "CANDIDATE_K", 20)
    candidate_k = max(requested_top_k, candidate_floor)
    backend = get_retriever()
    raw_results = backend.retrieve(analysis.get("retrieval_query") or query, top_k=candidate_k)
    raw_results = [
        {**result, "original_rank": index, "original_score": result.get("score")}
        for index, result in enumerate(raw_results, start=1)
    ]

    candidates = []
    seen_ids = set()
    seen_texts = set()
    for result in raw_results:
        document_id = result.get("document_id")
        if not document_id or document_id in seen_ids:
            continue
        if not _passes_filters(result, product=product, category=category):
            continue
        metadata = _metadata(result)
        result_product = result.get("product") or metadata.get("product")
        if (
            analysis.get("is_comparison")
            and clean(metadata.get("question_type")).lower() == "comparison"
            and len(analysis.get("mentioned_products") or []) >= 2
            and not _covers_all_products(result, analysis["mentioned_products"])
        ):
            continue
        if mentioned_products and not _result_product_matches(result_product, mentioned_products):
            continue
        if _is_broad_category_query(analysis) and not _category_matches_hint(metadata.get("category"), analysis.get("category_hint")):
            if analysis.get("category_hint") == "accounts" and clean(metadata.get("product")).lower() == "webank":
                pass
            else:
                continue
        if not mentioned_products and not _is_broad_category_query(analysis) and not _has_lexical_overlap(query, result):
            continue
        if not _is_useful_result(result):
            continue
        if _is_near_duplicate(result, seen_texts):
            continue
        seen_ids.add(document_id)
        candidates.append(result)

    reranked = _diversify_requested_products(_diversify_products(_rerank_results(candidates, analysis), analysis), analysis)
    results = [_format_result(index, result) for index, result in enumerate(reranked[:requested_top_k], start=1)]

    return {
        "query": query,
        "retriever_type": config.RETRIEVER_TYPE,
        "results": results,
        "mentioned_products": mentioned_products,
        "analysis": analysis,
        "before_rerank_ids": [result.get("document_id") for result in candidates[:requested_top_k]],
        "after_rerank_ids": [result.get("document_id") for result in reranked[:requested_top_k]],
    }


def _rank_map(results):
    return {
        result["document_id"]: {
            "rank": index,
            "score": result["score"],
        }
        for index, result in enumerate(results, start=1)
    }


def explain_retrieval(query: str, top_k: int = None):
    query = clean(query)
    if not query:
        raise ValueError("query must not be empty")

    requested_top_k = top_k or getattr(config, "TOP_K", 5)
    allowed, reason, _ = _preflight(query)
    if not allowed:
        print(f"Query: {query}")
        print(f"Retrieval strategy: {config.RETRIEVER_TYPE}")
        print(f"No trusted verified results: {reason}")
        return {
            "query": query,
            "retriever_type": config.RETRIEVER_TYPE,
            "results": [],
            "no_results_reason": reason,
        }

    candidate_k = max(requested_top_k, getattr(config, "CANDIDATE_K", 20))

    dense = DenseRetriever()
    bm25 = BM25Retriever()
    dense_results = dense.retrieve(query, top_k=candidate_k)
    bm25_results = bm25.retrieve(query, top_k=candidate_k)
    dense_ranks = _rank_map(dense_results)
    bm25_ranks = _rank_map(bm25_results)

    normal = retrieve(query, top_k=requested_top_k)
    print(f"Query: {query}")
    print(f"Retrieval strategy: {config.RETRIEVER_TYPE}")
    print("rank | document_id | product | score | dense_rank | bm25_rank | reranker_score")
    print("-----+-------------+---------+-------+------------+-----------+---------------")
    for result in normal["results"]:
        document_id = result["document_id"]
        dense_rank = dense_ranks.get(document_id, {}).get("rank", "")
        bm25_rank = bm25_ranks.get(document_id, {}).get("rank", "")
        reranker_score = result["score"] if config.RETRIEVER_TYPE == "dense_cross_encoder_rerank" else ""
        print(
            f"{result['rank']} | {document_id} | {result['product']} | "
            f"{result['score']:.6f} | {dense_rank} | {bm25_rank} | {reranker_score}"
        )
    return normal
