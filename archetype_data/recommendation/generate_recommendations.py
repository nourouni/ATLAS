from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from .config import DEFAULT_PATHS
from .hybrid_recommender import HybridRecommender


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def generate(output_path: Path, limit: int | None = None) -> Path:
    customer_ids = (
        pl.scan_parquet(DEFAULT_PATHS.propensity_recommendations_path)
        .select("client_id")
        .unique()
        .sort("client_id")
        .collect()
        .get_column("client_id")
        .to_list()
    )
    if limit is not None:
        customer_ids = customer_ids[:limit]

    recommender = HybridRecommender()
    rows: list[dict[str, object]] = []
    for index, customer_id in enumerate(customer_ids, start=1):
        result = recommender.recommend_customer(
            int(customer_id), top_k=5, persist_provenance=False
        )
        state = result["customer_state"]
        for recommendation in result["recommendations"]:
            rows.append(
                {
                    "client_id": int(customer_id),
                    "product_id": recommendation["product_id"],
                    "segment": state.get("segment"),
                    "behavioral_segment": state.get("behavioral_segment"),
                    "latent_archetype": state.get("latent_archetype"),
                    "product_name": recommendation["product"],
                    "category": recommendation.get("category"),
                    "classification": recommendation.get("classification"),
                    "type": recommendation.get("type"),
                    "rank": recommendation["rank"],
                    "propensity_score": recommendation.get("propensity_score"),
                    "final_score": recommendation["final_score"],
                    "need_match": recommendation["need_match"],
                    "matched_needs": _json(recommendation["matched_needs"]),
                    "eligible": recommendation["eligible"],
                    "already_owned": recommendation["already_owned"],
                    "missing_requirements": _json(recommendation["missing_requirements"]),
                    "conflicts": _json(recommendation["conflicts"]),
                    "alternatives": _json(recommendation["alternatives"]),
                    "supporting_relations": _json(recommendation["supporting_relations"]),
                    "sources": _json(recommendation["sources"]),
                    "explanation": recommendation["explanation"],
                }
            )
        if index % 10000 == 0 or index == len(customer_ids):
            print(f"Processed {index}/{len(customer_ids)} customers", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the current hybrid recommendation artifact.")
    parser.add_argument("--output", type=Path, default=DEFAULT_PATHS.final_recommendations_path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(generate(args.output, args.limit))


if __name__ == "__main__":
    main()
