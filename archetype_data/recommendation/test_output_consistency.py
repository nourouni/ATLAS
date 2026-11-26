from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import polars as pl

from hybrid_recommender import (
    PROPENSITY_PATH,
    REQUIRED_RECOMMENDATION_FIELDS,
    REQUIRED_RULE_FIELDS,
    HybridRecommender,
)


def _assert_fields(rows: list[dict[str, Any]], required_fields: tuple[str, ...], label: str) -> None:
    if not rows:
        raise AssertionError(f"Expected at least one {label} row.")

    expected_keys = set(rows[0])
    for index, row in enumerate(rows):
        missing = [field for field in required_fields if field not in row]
        if missing:
            raise AssertionError(f"{label} row {index} is missing required fields: {missing}")
        if set(row) != expected_keys:
            raise AssertionError(f"{label} row {index} has inconsistent output keys.")
        if not isinstance(row["sources"], list) or not row["sources"]:
            raise AssertionError(f"{label} row {index} has invalid sources.")
        if not isinstance(row["metadata"], dict):
            raise AssertionError(f"{label} row {index} has invalid metadata.")


def _assert_json_finite(value: Any, path: str = "result") -> None:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise AssertionError(f"{path} contains non-finite value {value}.")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_json_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_json_finite(item, f"{path}[{index}]")


def _assert_rule_consistency(result: dict[str, Any], customer_id: int) -> None:
    owned_products = set(result["customer_state"]["owned_products"])
    for row in result["rule_checks"]:
        source_flags = row["metadata"]["source_flags"]
        if row["product"] in owned_products and not row["already_owned"]:
            raise AssertionError(f"Owned product was not flagged for customer {customer_id}: {row['product']}.")
        if row["already_owned"] and row["eligible"] != "ineligible":
            raise AssertionError(f"Owned product is not ineligible for customer {customer_id}: {row['product']}.")
        if row["conflicts"] and row["eligible"] != "ineligible":
            raise AssertionError(f"Hard conflict is not ineligible for customer {customer_id}: {row['product']}.")
        if row["missing_requirements"] and row["eligible"] != "ineligible":
            raise AssertionError(f"Missing requirement is not ineligible for customer {customer_id}: {row['product']}.")
        if source_flags["ontology"] and not source_flags["propensity"] and row["propensity_score"] is not None:
            raise AssertionError(f"Ontology-only row has non-null propensity for customer {customer_id}: {row['product']}.")
        if source_flags["propensity"] and not source_flags["ontology"] and row["need_match"] != 0:
            raise AssertionError(f"Propensity-only row has non-zero need match for customer {customer_id}: {row['product']}.")


def _assert_rank_consistency(rows: list[dict[str, Any]], customer_id: int) -> None:
    ranks = [row["rank"] for row in rows]
    if ranks != list(range(1, len(rows) + 1)):
        raise AssertionError(f"Recommendation ranks are not sequential for customer {customer_id}: {ranks}.")
    if len(ranks) != len(set(ranks)):
        raise AssertionError(f"Recommendation ranks are not unique for customer {customer_id}: {ranks}.")
    for row in rows:
        score_components = row["score_components"]
        if "relation" in score_components or "affinity" in score_components:
            raise AssertionError(
                f"Relation or affinity leaked into scoring for customer {customer_id}: {row['product']}."
            )


def _sample_customer_ids(limit: int, path: Path = PROPENSITY_PATH) -> list[int]:
    return (
        pl.scan_parquet(path)
        .select("client_id")
        .unique()
        .sort("client_id")
        .limit(limit)
        .collect()
        .get_column("client_id")
        .to_list()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate hybrid recommender output consistency.")
    parser.add_argument("--sample-size", type=int, default=100)
    args = parser.parse_args()
    if args.sample_size < 1:
        raise ValueError("--sample-size must be at least 1.")

    recommender = HybridRecommender()
    sample_ids = _sample_customer_ids(limit=args.sample_size)
    if not sample_ids:
        raise AssertionError(f"No sample customer IDs found in {Path(PROPENSITY_PATH)}.")

    for customer_id in sample_ids:
        result = recommender.recommend_customer(int(customer_id), top_k=5)
        json.dumps(result, allow_nan=False)
        _assert_json_finite(result)
        _assert_fields(result["rule_checks"], REQUIRED_RULE_FIELDS, "rule check")
        _assert_fields(result["recommendations"], REQUIRED_RECOMMENDATION_FIELDS, "recommendation")
        _assert_rule_consistency(result, int(customer_id))
        _assert_rank_consistency(result["recommendations"], int(customer_id))

        scores = [row["final_score"] for row in result["recommendations"]]
        if scores != sorted(scores, reverse=True):
            raise AssertionError(f"Recommendations are not sorted for customer {customer_id}.")

    print(f"Output consistency checks passed for {len(sample_ids)} customers.")


if __name__ == "__main__":
    main()
