from __future__ import annotations

import argparse
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import polars as pl

try:
    from .config import DEFAULT_PATHS
except ImportError:
    from config import DEFAULT_PATHS

ONTOLOGY_PATH = DEFAULT_PATHS.ontology_path
PROPENSITY_PATH = DEFAULT_PATHS.propensity_recommendations_path
OWNERSHIP_PATH = DEFAULT_PATHS.ownership_long_path
RECOMMENDATION_PROVENANCE_PATH = (
    Path(__file__).resolve().parents[2]
    / "archetype_data"
    / "artifacts"
    / "need_discovery"
    / "recommendation_candidate_events.jsonl"
)

RDF_ABOUT = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about"
RDF_RESOURCE = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource"

SEGMENT_ARCHETYPE_ALIASES = {
    "digital active transactors": "Digital Power Retail",
    "thin buffer low income": "ATM Dependent Minimal Retail",
    "high spend salary users": "Stable Engaged Retail",
    "stable salary savers": "Stable Engaged Retail",
    "wealth investment clients": "Stable Engaged Retail",
    "tre remittance clients": "Omnichannel Retail Outliers",
    "irregular freelance income": "Moderate Activity Strained Retail",
    "credit dependent business": "Cashflow Strained Enterprises",
    "seasonal volatile business": "Volatile Low Activity Enterprises",
    "stable operating business": "Investment Active Enterprises",
    "cash management mature corp": "Investment Active Enterprises",
    "high throughput low margin": "Enterprise Power Users",
}

SEGMENT_LABEL_ALIASES = {
    "jeune": "Study Service ATM Retail",
    "salarie": "Stable Engaged Retail",
    "retraite": "Stable Engaged Retail",
    "sans profession": "ATM Dependent Minimal Retail",
    "profession liberale": "Moderate Activity Strained Retail",
    "pme": "Cashflow Strained Enterprises",
    "tpe": "Volatile Low Activity Enterprises",
    "grande entreprise": "Enterprise Power Users",
    "institution": "Investment Active Enterprises",
}

HARD_NEGATIVE_RELATIONS = {"negates", "incompatibleWith"}
ALTERNATIVE_RELATIONS = {"alternativeTo"}
REQUIRES_RELATION = "requires"
SOFT_RELATIONS = {"complements", "contains", "relatedService", "servesSameNeed", "upgradeOf"}
PRODUCT_RELATIONS = HARD_NEGATIVE_RELATIONS | SOFT_RELATIONS | {REQUIRES_RELATION}

PRODUCT_REPLACEMENTS = {}

NUMERIC_CANDIDATE_FIELDS = (
    "propensity_score",
    "segment_product_prior",
    "max_owned_to_candidate_affinity",
    "mean_owned_to_candidate_affinity",
)
REQUIRED_CANDIDATE_FIELDS = (
    "product",
    "category",
    "family",
    "propensity_score",
    "segment_product_prior",
    "max_owned_to_candidate_affinity",
    "mean_owned_to_candidate_affinity",
    "sources",
)
REQUIRED_RULE_FIELDS = REQUIRED_CANDIDATE_FIELDS + (
    "metadata",
    "need_match",
    "matched_needs",
    "eligible",
    "eligibility_basis",
    "eligibility_rules",
    "already_owned",
    "missing_requirements",
    "conflicts",
    "alternatives",
    "supporting_relations",
    "relation_evidence",
    "keep",
)
REQUIRED_RECOMMENDATION_FIELDS = REQUIRED_RULE_FIELDS + (
    "score_components",
    "final_score",
    "rank",
    "explanation",
)


def _candidate_status(candidate: dict[str, Any], selected_products: set[str]) -> str:
    """Map existing rule outcomes to provenance-only status labels."""
    product = str(candidate["product"])
    if product in selected_products:
        return "selected"
    if candidate.get("already_owned"):
        return "excluded_owned"
    if candidate.get("missing_requirements"):
        return "excluded_missing_requirement"
    if candidate.get("conflicts"):
        return "excluded_hard_conflict"
    if candidate.get("eligible") == "ineligible":
        return "excluded_eligibility"
    return "conditional"


def append_candidate_provenance(
    *,
    customer_id: int,
    checked_candidates: list[dict[str, Any]],
    ranked_candidates: list[dict[str, Any]],
    selected_candidates: list[dict[str, Any]],
    path: Path = RECOMMENDATION_PROVENANCE_PATH,
) -> None:
    """Append candidate decisions already computed by the recommender.

    This function only serializes existing fields; it does not participate in
    candidate generation, rule checks, scoring, sorting, or top-k selection.
    """
    selected_products = {str(row["product"]) for row in selected_candidates}
    ranked_by_product = {str(row["product"]): row for row in ranked_candidates}
    run_id = f"recrun_{uuid4().hex}"
    timestamp = datetime.now(timezone.utc).isoformat()
    events: list[dict[str, Any]] = []
    for candidate in checked_candidates:
        product = str(candidate["product"])
        ranked = ranked_by_product.get(product, {})
        hard_conflicts = [
            conflict for conflict in (candidate.get("conflicts") or []) if conflict != "already_owned"
        ]
        events.append(
            {
                "event_id": f"recgap_{uuid4().hex}",
                "recommendation_run_id": run_id,
                "timestamp": timestamp,
                "customer_id": customer_id,
                "product": product,
                "matched_needs": candidate.get("matched_needs") or [],
                "candidate_status": _candidate_status(candidate, selected_products),
                "requirement_status": "missing" if candidate.get("missing_requirements") else "satisfied_or_not_required",
                "missing_requirements": candidate.get("missing_requirements") or [],
                "eligibility_status": candidate.get("eligible"),
                "ownership_status": "owned" if candidate.get("already_owned") else "not_owned",
                "hard_conflicts": hard_conflicts,
                "conflicts": candidate.get("conflicts") or [],
                "alternative_relationships": candidate.get("alternatives") or [],
                "affinity": candidate.get("max_owned_to_candidate_affinity"),
                "segment_prior": candidate.get("segment_product_prior"),
                "propensity_score": candidate.get("propensity_score"),
                "need_match": candidate.get("need_match"),
                "score_components": ranked.get("score_components"),
                "final_score": ranked.get("final_score"),
                "rank": ranked.get("rank"),
            }
        )
    if not events:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")


def _local_name(value: str | None) -> str:
    if not value:
        return ""
    return value.rsplit("#", 1)[-1].rsplit("/", 1)[-1]


def _relation_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _normalize(value: str | None) -> str:
    if value is None:
        return ""
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _clean_string(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text or None


def _recommendation_product(value: Any) -> str:
    cleaned = _clean_string(value) or ""
    return PRODUCT_REPLACEMENTS.get(cleaned.casefold(), cleaned)


def _clean_number(value: Any, default: float = 0.0) -> float:
    if value is None:
        return defaul
    try:
        number = float(value)
    except (TypeError, ValueError):
        return defaul
    if math.isnan(number) or math.isinf(number):
        return defaul
    return number


@dataclass
class OntologyIndex:
    label_by_uri: dict[str, str] = field(default_factory=dict)
    uri_by_label: dict[str, str] = field(default_factory=dict)
    product_uri_by_label: dict[str, str] = field(default_factory=dict)
    types_by_uri: dict[str, set[str]] = field(default_factory=dict)
    props_by_uri: dict[str, dict[str, set[str]]] = field(default_factory=dict)
    literals_by_uri: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    need_description_by_uri: dict[str, str] = field(default_factory=dict)
    description_by_uri: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_owl(cls, path: Path = ONTOLOGY_PATH) -> "OntologyIndex":
        index = cls()
        if not path.exists():
            return index

        root = ET.parse(path).getroot()
        for desc in root.findall(".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}Description"):
            uri = desc.attrib.get(RDF_ABOUT)
            if not uri:
                continue

            props = index.props_by_uri.setdefault(uri, {})
            literals = index.literals_by_uri.setdefault(uri, {})
            for child in desc:
                rel = _relation_name(child.tag)
                resource = child.attrib.get(RDF_RESOURCE)

                if rel == "type" and resource:
                    index.types_by_uri.setdefault(uri, set()).add(_local_name(resource))
                elif rel in {"label", "productName"} and child.text:
                    label = child.text.strip()
                    index.label_by_uri.setdefault(uri, label)
                    index.uri_by_label[_normalize(label)] = uri
                elif rel == "description" and child.text:
                    description = child.text.strip()
                    index.description_by_uri[uri] = description
                    index.need_description_by_uri[uri] = description
                elif resource:
                    props.setdefault(rel, set()).add(resource)
                elif child.text and child.text.strip():
                    literals.setdefault(rel, []).append(child.text.strip())

        for uri in index.props_by_uri:
            index.label_by_uri.setdefault(uri, _local_name(uri))
            index.uri_by_label.setdefault(_normalize(_local_name(uri)), uri)

        for uri, types in index.types_by_uri.items():
            if "Product" not in types:
                continue
            label = index.label_by_uri.get(uri, _local_name(uri))
            index.product_uri_by_label[_normalize(label)] = uri
            index.product_uri_by_label.setdefault(_normalize(_local_name(uri)), uri)
        return index

    def product_uri(self, product_name: str) -> str | None:
        normalized = _normalize(product_name)
        return self.product_uri_by_label.get(normalized) or self.uri_by_label.get(normalized)

    def label(self, uri: str | None) -> str:
        if not uri:
            return ""
        return self.label_by_uri.get(uri, _local_name(uri))

    def product_need_uris(self, product_name: str) -> set[str]:
        uri = self.product_uri(product_name)
        if not uri:
            return set()
        return self.props_by_uri.get(uri, {}).get("servesNeed", set())

    def segment_need_uris(self, segment_label: str) -> set[str]:
        uri = self.uri_by_label.get(_normalize(segment_label))
        if not uri:
            return set()
        return self.props_by_uri.get(uri, {}).get("likelyNeeds", set())

    def segment_need_uris_for_context(
        self, segment: str | None, latent_archetype: str | None
    ) -> set[str]:
        candidate_labels = [
            segment,
            SEGMENT_LABEL_ALIASES.get(_normalize(segment)),
            SEGMENT_ARCHETYPE_ALIASES.get(_normalize(latent_archetype)),
        ]
        need_uris: set[str] = set()
        for label in candidate_labels:
            if label:
                need_uris.update(self.segment_need_uris(str(label)))
        return need_uris

    def segment_need_evidence_for_context(
        self, segment: str | None, latent_archetype: str | None
    ) -> dict[str, list[dict[str, Any]]]:
        sources = [
            ("behavioral_segment", segment, segment),
            (
                "alias_mapping",
                SEGMENT_LABEL_ALIASES.get(_normalize(segment)),
                segment,
            ),
            (
                "latent_archetype",
                SEGMENT_ARCHETYPE_ALIASES.get(_normalize(latent_archetype)),
                latent_archetype,
            ),
        ]
        evidence_by_need: dict[str, list[dict[str, Any]]] = {}
        seen: set[tuple[str, str, str, str]] = set()
        for source_type, ontology_label, observed_value in sources:
            if not ontology_label:
                continue
            for need_uri in self.segment_need_uris(str(ontology_label)):
                key = (need_uri, source_type, str(ontology_label), str(observed_value))
                if key in seen:
                    continue
                seen.add(key)
                evidence_by_need.setdefault(need_uri, []).append(
                    {
                        "source_type": source_type,
                        "observed_value": observed_value,
                        "ontology_segment": ontology_label,
                    }
                )
        return evidence_by_need

    def products_serving_needs(self, need_uris: set[str]) -> list[str]:
        products: list[str] = []
        for uri, props in self.props_by_uri.items():
            if not props.get("servesNeed", set()).intersection(need_uris):
                continue
            if "Product" in self.types_by_uri.get(uri, set()) or uri in self.label_by_uri:
                products.append(self.label(uri))
        return sorted(set(products))

    def relations_between(self, source_product: str, target_product: str) -> list[str]:
        source_uri = self.product_uri(source_product)
        target_uri = self.product_uri(target_product)
        if not source_uri or not target_uri:
            return []

        relations: list[str] = []
        source_props = self.props_by_uri.get(source_uri, {})
        target_props = self.props_by_uri.get(target_uri, {})
        for relation in SOFT_RELATIONS:
            if target_uri in source_props.get(relation, set()):
                relations.append(f"{source_product} {relation} {target_product}")
            if source_uri in target_props.get(relation, set()):
                relations.append(f"{target_product} {relation} {source_product}")
        return relations

    def relation_evidence_between(self, source_product: str, target_product: str) -> list[dict[str, str]]:
        source_uri = self.product_uri(source_product)
        target_uri = self.product_uri(target_product)
        if not source_uri or not target_uri:
            return []

        evidence: list[dict[str, str]] = []
        source_props = self.props_by_uri.get(source_uri, {})
        target_props = self.props_by_uri.get(target_uri, {})
        for relation in SOFT_RELATIONS:
            if target_uri in source_props.get(relation, set()):
                evidence.append(
                    {"relation": relation, "source_product": source_product, "target_product": target_product}
                )
            if source_uri in target_props.get(relation, set()):
                evidence.append(
                    {"relation": relation, "source_product": target_product, "target_product": source_product}
                )
        return evidence

    def eligibility_rules(self, product_uri: str | None) -> list[dict[str, Any]]:
        if not product_uri:
            return []

        rules = []
        for rule_uri in sorted(self.props_by_uri.get(product_uri, {}).get("hasEligibilityRule", set())):
            literals = self.literals_by_uri.get(rule_uri, {})
            rules.append(
                {
                    "rule": self.label(rule_uri),
                    "evidence": " ".join(literals.get("supportingEvidence", [])),
                    "requires_manual_review": any(
                        value.lower() == "true" for value in literals.get("requiresManualReview", [])
                    ),
                    "validation_status": next(iter(literals.get("validationStatus", [])), None),
                    "confidence": next(iter(literals.get("confidence", [])), None),
                }
            )
        return rules

    def product_constraints(self, product_uri: str | None) -> list[str]:
        if not product_uri:
            return []
        descriptions = []
        for constraint_uri in sorted(self.props_by_uri.get(product_uri, {}).get("hasOwnershipConstraint", set())):
            description = self.description_by_uri.get(constraint_uri)
            if description:
                descriptions.append(f"{self.label(constraint_uri)}: {description}")
        return descriptions


@dataclass
class RecommendationConfig:
    propensity_weight: float = 0.70
    need_weight: float = 0.30
    candidate_limit: int = 50


class HybridRecommender:
    def __init__(
        self,
        propensity_path: Path = PROPENSITY_PATH,
        ownership_path: Path = OWNERSHIP_PATH,
        ontology_path: Path = ONTOLOGY_PATH,
        config: RecommendationConfig | None = None,
    ) -> None:
        self.propensity_path = propensity_path
        self.ownership_path = ownership_path
        self.ontology = OntologyIndex.from_owl(ontology_path)
        self.config = config or RecommendationConfig()
        catalog = pl.read_csv(DEFAULT_PATHS.product_catalog_path).select(["product_id", "product_name"])
        self._canonical_product_names = {
            str(row["product_id"]): str(row["product_name"])
            for row in catalog.iter_rows(named=True)
        }
        self._canonical_product_ids = set(self._canonical_product_names)
        # Index the two large input artifacts once.  The previous implementation
        # opened and filtered both Parquet files for every customer, turning the
        # recommendation run into hundreds of thousands of repeated disk scans.
        self._propensity_by_customer = self._index_customer_frames(self.propensity_path)
        self._owned_by_customer = self._index_customer_frames(self.ownership_path)

    @staticmethod
    def _index_customer_frames(path: Path) -> dict[int, pl.DataFrame]:
        frame = pl.read_parquet(path)
        if "client_id" not in frame.columns:
            raise ValueError(f"Recommendation source is missing client_id: {path}")
        return {
            int(key[0] if isinstance(key, tuple) else key): customer_frame
            for key, customer_frame in frame.partition_by("client_id", as_dict=True).items()
        }

    def get_customer_state(self, customer_id: int) -> dict[str, Any]:
        recs = self._customer_propensity_rows(customer_id)
        owned = self._customer_owned_rows(customer_id)

        if recs.height:
            first = recs.row(0, named=True)
            behavioral_segment = first.get("behavioral_segment")
            customer_category = first.get("segment")
            latent_archetype = first.get("latent_archetype")
        elif owned.height:
            first = owned.row(0, named=True)
            behavioral_segment = first.get("behavioral_segment")
            customer_category = first.get("segment")
            latent_archetype = None
        else:
            raise ValueError(f"No recommendation or ownership rows found for customer {customer_id}.")

        owned_product_column = "product_name" if "product_name" in owned.columns else "product"
        owned_products = owned.get_column(owned_product_column).unique().sort().to_list() if owned.height else []
        inferred_need_uris = self.ontology.segment_need_uris_for_context(
            str(behavioral_segment) if behavioral_segment is not None else None,
            str(latent_archetype) if latent_archetype is not None else None,
        )
        inferred_need_evidence = self.ontology.segment_need_evidence_for_context(
            str(behavioral_segment) if behavioral_segment is not None else None,
            str(latent_archetype) if latent_archetype is not None else None,
        )
        return {
            "client_id": customer_id,
            "segment": behavioral_segment,
            "behavioral_segment": behavioral_segment,
            "customer_category": customer_category,
            "latent_archetype": latent_archetype,
            "owned_products": owned_products,
            "inferred_needs": [self._format_need(uri) for uri in sorted(inferred_need_uris)],
            "inferred_need_provenance": {
                self._format_need(uri): evidence for uri, evidence in sorted(inferred_need_evidence.items())
            },
            "inferred_need_uris": sorted(inferred_need_uris),
        }

    def get_ontology_candidates(self, customer_state: dict[str, Any]) -> list[dict[str, Any]]:
        need_uris = set(customer_state["inferred_need_uris"])
        products = self.ontology.products_serving_needs(need_uris)
        candidates = []
        for product in products:
            product_uri = self.ontology.product_uri(product)
            product_id = _local_name(product_uri) if product_uri else None
            if product_id not in self._canonical_product_ids:
                raise ValueError(
                    f"Ontology product does not resolve to a verified canonical product_id: {product}"
                )
            candidates.append(
                {
                    "product": self._canonical_product_names[product_id],
                    "product_id": product_id,
                    "source": "ontology",
                }
            )
        return candidates

    def get_propensity_candidates(self, customer_id: int) -> list[dict[str, Any]]:
        rows = self._customer_propensity_rows(customer_id)
        rows = rows.sort("propensity_score", descending=True).head(self.config.candidate_limit)
        candidates = []
        for row in rows.iter_rows(named=True):
            product = _clean_string(row.get("product_name")) or _clean_string(row.get("product"))
            if not product:
                continue
            candidates.append(
                {
                    "product_id": _clean_string(row.get("product_id")),
                    "product": product,
                    "category": _clean_string(row.get("category")),
                    "family": _clean_string(row.get("family")),
                    "propensity_score": _clean_number(row.get("propensity_score")),
                    "segment_product_prior": _clean_number(row.get("segment_product_prior")),
                    "max_owned_to_candidate_affinity": _clean_number(
                        row.get("max_owned_to_candidate_affinity")
                    ),
                    "mean_owned_to_candidate_affinity": _clean_number(
                        row.get("mean_owned_to_candidate_affinity")
                    ),
                    "source": "propensity",
                }
            )
        return candidates

    def generate_candidates(self, customer_state: dict[str, Any]) -> list[dict[str, Any]]:
        by_product: dict[str, dict[str, Any]] = {}
        for candidate in self.get_ontology_candidates(customer_state):
            product = _recommendation_product(candidate["product"])
            row = by_product.setdefault(product, self._default_candidate(product))
            row["product_id"] = candidate["product_id"]
            by_product[product]["sources"].append("ontology")

        for candidate in self.get_propensity_candidates(int(customer_state["client_id"])):
            product = _recommendation_product(candidate["product"])
            row = by_product.setdefault(product, self._default_candidate(product))
            row.update({k: v for k, v in candidate.items() if k != "source"})
            row["product"] = produc
            row["sources"].append("propensity")

        candidates = [self._normalize_candidate(row) for row in by_product.values()]
        self._validate_rows(candidates, REQUIRED_CANDIDATE_FIELDS, "candidate")
        return candidates

    def apply_ontology_rules(
        self, customer_state: dict[str, Any], candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        owned_products = set(customer_state["owned_products"])
        owned_uris = {self.ontology.product_uri(product) for product in owned_products}
        owned_uris.discard(None)
        inferred_need_uris = set(customer_state["inferred_need_uris"])

        checked = []
        for candidate in candidates:
            product = candidate["product"]
            product_uri = self.ontology.product_uri(product)
            product_props = self.ontology.props_by_uri.get(product_uri or "", {})
            product_needs = self.ontology.product_need_uris(product)
            matched_needs = product_needs.intersection(inferred_need_uris)

            conflicts: list[str] = []
            alternatives: list[str] = []
            missing_requirements: list[str] = []
            supporting_relations: list[str] = []
            relation_evidence: list[dict[str, str]] = []

            already_owned = product in owned_products
            if already_owned:
                conflicts.append("already_owned")

            for required_uri in sorted(product_props.get(REQUIRES_RELATION, set())):
                if required_uri not in owned_uris:
                    missing_requirements.append(self.ontology.label(required_uri))

            for relation in HARD_NEGATIVE_RELATIONS:
                for blocked_uri in product_props.get(relation, set()):
                    if blocked_uri in owned_uris:
                        conflicts.append(f"{relation}: {product} -> {self.ontology.label(blocked_uri)}")
                for owned_uri in owned_uris:
                    owned_props = self.ontology.props_by_uri.get(owned_uri, {})
                    if product_uri and product_uri in owned_props.get(relation, set()):
                        conflicts.append(f"{relation}: {self.ontology.label(owned_uri)} -> {product}")

            for relation in ALTERNATIVE_RELATIONS:
                for related_uri in product_props.get(relation, set()):
                    alternatives.append(f"{product} alternativeTo {self.ontology.label(related_uri)}")
                for owned_uri in owned_uris:
                    owned_props = self.ontology.props_by_uri.get(owned_uri, {})
                    if product_uri and product_uri in owned_props.get(relation, set()):
                        alternatives.append(f"{self.ontology.label(owned_uri)} alternativeTo {product}")

            for owned_product in owned_products:
                supporting_relations.extend(self.ontology.relations_between(owned_product, product))
                relation_evidence.extend(self.ontology.relation_evidence_between(owned_product, product))
            supporting_relations.extend(self.ontology.product_constraints(product_uri))

            affinity = _clean_number(candidate.get("max_owned_to_candidate_affinity"))
            if affinity > 0:
                supporting_relations.append(
                    "propensity portfolio affinity "
                    f"{affinity:.3f}"
                )

            need_match = 1 if matched_needs else 0
            keep = not already_owned and not conflicts and not missing_requirements
            eligibility_rules = self.ontology.eligibility_rules(product_uri)

            checked.append(
                {
                    **candidate,
                    "metadata": self._candidate_metadata(candidate, product_uri),
                    "need_match": need_match,
                    "matched_needs": [self._format_need(uri) for uri in sorted(matched_needs)],
                    "eligible": self._eligibility_status(
                        already_owned, conflicts, missing_requirements, eligibility_rules
                    ),
                    "eligibility_basis": (
                        "No public eligibility rule found in ontology."
                        if not eligibility_rules
                        else "Public eligibility rule found; not customer-evaluated."
                    ),
                    "eligibility_rules": eligibility_rules,
                    "already_owned": already_owned,
                    "missing_requirements": missing_requirements,
                    "conflicts": conflicts,
                    "alternatives": sorted(set(alternatives)),
                    "supporting_relations": sorted(set(supporting_relations)),
                    "relation_evidence": relation_evidence,
                    "keep": keep,
                }
            )
        self._validate_rows(checked, REQUIRED_RULE_FIELDS, "rule check")
        return checked

    def rank_candidates(self, checked_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        valid = [candidate for candidate in checked_candidates if candidate["keep"]]
        ranked = []
        for candidate in valid:
            propensity = _clean_number(candidate.get("propensity_score"))
            need_score = _clean_number(candidate["need_match"])
            final_score = (
                self.config.propensity_weight * propensity
                + self.config.need_weight * need_score
            )
            score_components = {
                "propensity": propensity,
                "need": need_score,
                "weights": {
                    "propensity": self.config.propensity_weight,
                    "need": self.config.need_weight,
                },
            }
            ranked.append(
                {
                    **candidate,
                    "score_components": score_components,
                    "final_score": final_score,
                }
            )

        ranked.sort(
            key=lambda row: (
                -_clean_number(row["final_score"]),
                -_clean_number(row.get("propensity_score")),
                -_clean_number(row.get("need_match")),
                str(row["product"]).lower(),
            )
        )
        for rank, candidate in enumerate(ranked, start=1):
            candidate["rank"] = rank
            candidate["explanation"] = self.build_explanation(candidate)
        self._validate_rows(ranked, REQUIRED_RECOMMENDATION_FIELDS, "recommendation")
        return ranked

    def build_explanation(self, candidate: dict[str, Any]) -> str:
        propensity = candidate.get("propensity_score")
        if not candidate.get("metadata", {}).get("propensity_available", False):
            parts = [f"{candidate['product']} is recommended from ontology evidence"]
        else:
            parts = [
                f"{candidate['product']} is recommended with a "
                f"{_clean_number(propensity):.1%} product propensity score"
            ]
        if candidate["matched_needs"]:
            parts.append("It serves inferred need(s): " + ", ".join(candidate["matched_needs"]))
        if candidate["supporting_relations"]:
            parts.append("Supporting evidence: " + "; ".join(candidate["supporting_relations"][:3]))
        if candidate.get("eligibility_rules"):
            parts.append("Public eligibility rule evidence is attached for manual review")
        parts.append("No blocking ownership, requirement, or conflict rule was found")
        return ". ".join(parts) + "."

    def recommend_customer(
        self,
        customer_id: int,
        top_k: int = 5,
        *,
        persist_provenance: bool = True,
        provenance_path: Path | None = None,
    ) -> dict[str, Any]:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        customer_state = self.get_customer_state(customer_id)
        candidates = self.generate_candidates(customer_state)
        checked = self.apply_ontology_rules(customer_state, candidates)
        ranked = self.rank_candidates(checked)
        selected = ranked[:top_k]
        if persist_provenance:
            append_candidate_provenance(
                customer_id=customer_id,
                checked_candidates=checked,
                ranked_candidates=ranked,
                selected_candidates=selected,
                path=provenance_path or RECOMMENDATION_PROVENANCE_PATH,
            )
        return {
            "customer_state": {
                key: value for key, value in customer_state.items() if key != "inferred_need_uris"
            },
            "candidate_generation": {
                "n_candidates": len(candidates),
                "n_ontology_candidates": sum("ontology" in row["sources"] for row in candidates),
                "n_propensity_candidates": sum("propensity" in row["sources"] for row in candidates),
            },
            "rule_checks": checked,
            "recommendations": selected,
        }

    def _customer_propensity_rows(self, customer_id: int) -> pl.DataFrame:
        return self._propensity_by_customer.get(customer_id, pl.DataFrame())

    def _customer_owned_rows(self, customer_id: int) -> pl.DataFrame:
        return self._owned_by_customer.get(customer_id, pl.DataFrame())

    def _format_need(self, need_uri: str) -> str:
        label = self.ontology.label(need_uri)
        description = self.ontology.need_description_by_uri.get(need_uri)
        if description and description != label:
            return f"{label}: {description}"
        return label

    def _default_candidate(self, product: str) -> dict[str, Any]:
        return {
            "product": product,
            "product_id": None,
            "category": None,
            "family": None,
            "propensity_score": None,
            "segment_product_prior": 0.0,
            "max_owned_to_candidate_affinity": 0.0,
            "mean_owned_to_candidate_affinity": 0.0,
            "sources": [],
        }

    def _normalize_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        normalized = self._default_candidate(str(candidate["product"]))
        normalized.update(candidate)
        normalized["product"] = str(normalized["product"])
        normalized["product_id"] = _clean_string(normalized.get("product_id"))
        normalized["category"] = _clean_string(normalized.get("category"))
        normalized["family"] = _clean_string(normalized.get("family"))
        normalized["sources"] = sorted(set(str(source) for source in normalized.get("sources", []) if source))
        for field_name in NUMERIC_CANDIDATE_FIELDS:
            if field_name == "propensity_score" and "propensity" not in normalized["sources"]:
                normalized[field_name] = None
                continue
            normalized[field_name] = _clean_number(normalized.get(field_name))
        return normalized

    def _candidate_metadata(self, candidate: dict[str, Any], product_uri: str | None) -> dict[str, Any]:
        return {
            "product_uri": product_uri,
            "ontology_label": self.ontology.label(product_uri) if product_uri else None,
            "source_flags": {
                "ontology": "ontology" in candidate["sources"],
                "propensity": "propensity" in candidate["sources"],
            },
            "propensity_available": "propensity" in candidate["sources"],
        }

    def _eligibility_status(
        self,
        already_owned: bool,
        conflicts: list[str],
        missing_requirements: list[str],
        eligibility_rules: list[dict[str, Any]],
    ) -> str:
        if already_owned or conflicts or missing_requirements:
            return "ineligible"
        if eligibility_rules:
            return "unknown"
        return "eligible"

    def _validate_rows(
        self, rows: list[dict[str, Any]], required_fields: tuple[str, ...], row_type: str
    ) -> None:
        for index, row in enumerate(rows):
            missing = [field_name for field_name in required_fields if field_name not in row]
            if missing:
                raise ValueError(f"{row_type} row {index} is missing required field(s): {missing}")
            if not row.get("product"):
                raise ValueError(f"{row_type} row {index} has an empty product.")
            for field_name in NUMERIC_CANDIDATE_FIELDS:
                value = row.get(field_name)
                if field_name == "propensity_score" and value is None:
                    continue
                if not isinstance(value, float) or math.isnan(value) or math.isinf(value):
                    raise ValueError(f"{row_type} row {index} has invalid numeric field {field_name}: {value}")


def _print_text_result(result: dict[str, Any]) -> None:
    state = result["customer_state"]
    print(f"Customer {state['client_id']}")
    print(f"Segment: {state.get('segment')}")
    print(f"Archetype: {state.get('latent_archetype')}")
    print(f"Owned products: {', '.join(state['owned_products']) or 'none'}")
    print()

    for rec in result["recommendations"]:
        print(f"{rec['rank']}. {rec['product']}")
        propensity = rec.get("propensity_score")
        print(f"   propensity: {'none' if propensity is None else f'{_clean_number(propensity):.3f}'}")
        print(f"   need_match: {'yes' if rec['need_match'] else 'no'}")
        print(f"   eligible: {rec['eligible']}")
        print(f"   conflicts: {', '.join(rec['conflicts']) or 'none'}")
        print(f"   final_score: {rec['final_score']:.3f}")
        print(f"   explanation: {rec['explanation']}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid ontology + product propensity recommendations.")
    parser.add_argument("--customer-id", type=int, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    recommender = HybridRecommender()
    result = recommender.recommend_customer(args.customer_id, top_k=args.top_k)

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
    _print_text_result(result)


if __name__ == "__main__":
    main()
