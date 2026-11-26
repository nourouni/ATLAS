from __future__ import annotations

import argparse

from propensity_model import PropensityConfig, SchemaConfig, run_propensity_experimen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run customer engagement propensity modeling.")
    parser.add_argument("--segment", choices=["retail", "enterprise", "all"], default="retail")
    parser.add_argument("--lookback-weeks", type=int, default=8)
    parser.add_argument("--horizon-weeks", type=int, default=8)
    parser.add_argument("--min-relative-growth", type=float, default=0.20)
    parser.add_argument("--min-absolute-growth", type=float, default=3.0)
    parser.add_argument("--min-future-activity-when-past-zero", type=float, default=3.0)
    parser.add_argument("--composite-min-relative-improvement", type=float, default=0.10)
    parser.add_argument("--composite-min-absolute-improvement", type=float, default=0.25)
    parser.add_argument("--min-precision", type=float, default=0.50)
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--max-clients", type=int, default=20_000)
    parser.add_argument("--strict-client-holdout", action="store_true")
    parser.add_argument("--client-id", default="client_id")
    parser.add_argument("--cutoff-index", default="week_idx")
    parser.add_argument("--cutoff-start", default="week_start")
    parser.add_argument("--cutoff-end", default="week_end")
    parser.add_argument("--segment-column", default="segment")
    parser.add_argument("--activity-column", default="period_frequency")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_clients = None if args.max_clients <= 0 else args.max_clients
    config = PropensityConfig(
        segment=args.segment,
        lookback_weeks=args.lookback_weeks,
        horizon_weeks=args.horizon_weeks,
        min_relative_growth=args.min_relative_growth,
        min_absolute_growth=args.min_absolute_growth,
        min_future_activity_when_past_zero=args.min_future_activity_when_past_zero,
        composite_min_relative_improvement=args.composite_min_relative_improvement,
        composite_min_absolute_improvement=args.composite_min_absolute_improvement,
        min_precision=args.min_precision,
        validation_size=args.validation_size,
        test_size=args.test_size,
        max_clients=max_clients,
        strict_client_holdout=args.strict_client_holdout,
        schema=SchemaConfig(
            client_id=args.client_id,
            cutoff_index=args.cutoff_index,
            cutoff_start=args.cutoff_start,
            cutoff_end=args.cutoff_end,
            segment=args.segment_column,
            activity=args.activity_column,
        ),
    )
    results = run_propensity_experiment(config)
    print("Training frame:", results["training_frame"].shape)
    print("Selected target:", results["selected_target"])
    print("Selected model:", results["selected_model_name"])
    print("Selected feature set:", results["selected_feature_set"])
    print("Selected top 2 models")
    for row in results["selected_top2_models"]:
        print(
            f"  rank {row['rank']}: {row['selected_target']} / "
            f"{row['selected_feature_set']} / {row['selected_model']} -> {row['artifact']}"
        )
    print(results["split_summary"])
    print("Validation comparison")
    print(results["validation_comparison"])
    print("Final test comparison")
    print(results["test_comparison"])


if __name__ == "__main__":
    main()
