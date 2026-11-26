# Product Propensity Workflow

This folder documents the product propensity workflow used for produc
recommendations in the archetype dataset.

Product propensity estimates how likely a customer is to fit or adopt a produc
they do not already own. It uses only `archetype_data` outputs and does not use
`initial_data`.

Main files:

- `product_propensity_model.py`
- `product_propensity_workflow.ipynb`
- `product_propensity_report_notes.md`
- `outputs/product_propensity/`

The modeling unit is:

```tex
one row = one customer + one candidate produc
```

The target is built from simulated product ownership:

```tex
target = 1 if the customer already owns the produc
target = 0 if the product is segment-eligible but not owned by the customer
```

So yes: the supervised model is trained by learning the patterns behind wha
products each customer already owns. At recommendation time, it scores only
products with `target = 0`, meaning products the customer does not already own.
This makes the score a cross-sell/adoption propensity proxy rather than a direc
observed future purchase event.

Product propensity source files:

- `../outputs/archetype_customers.csv`
- `../outputs/archetype_features.csv`
- `../products/simulation/outputs/product_ownership_long.parquet`
- `../products/simulation/outputs/product_ownership_features.parquet`
- `../products/simulation/outputs/segment_product_probabilities.parquet`
- `../products/simulation/outputs/conditional_product_links.parquet`

Current product propensity outputs:

- `outputs/product_propensity/product_propensity_training_frame.parquet`
- `outputs/product_propensity/product_propensity_scoring_frame.parquet`
- `outputs/product_propensity/product_model_comparison.csv`
- `outputs/product_propensity/product_feature_list.csv`
- `outputs/product_propensity/best_product_propensity_model.joblib`
- `outputs/product_propensity/product_holdout_predictions.parquet`
- `outputs/product_propensity/top_non_owned_product_recommendations.parquet`

Important output columns:

- `target`: ownership label used for training; `1` owned, `0` not owned.
- `propensity_score`: model probability that the customer-product pair resembles
  the positive class.
- `rank`: product rank within the customer's recommendation list.
- `segment_product_prior`: baseline product ownership probability for the
  customer segment.
- `max_owned_to_candidate_affinity`: strongest conditional link from the
  customer's existing products to the candidate product.
- `mean_owned_to_candidate_affinity`: average conditional link from the existing
  portfolio to the candidate product.

Model metrics:

- `average_precision`: ranking quality for finding owned/positive examples.
  Higher is better and it is especially useful for recommendation-style models.
- `precision_at_0_5`: among rows scored at least `0.5`, how many were actually
  positive.
- `recall_at_0_5`: among all positive rows, how many received a score at leas
  `0.5`.
- `f1_at_0_5`: balance of precision and recall at the `0.5` cutoff.
- `roc_auc`: general positive-vs-negative separation.
- `brier`: probability calibration error; lower is better.

The `0.5` cutoff means a probability threshold of 50 percent. If
`propensity_score >= 0.5`, the model is saying that the pair is more likely to
belong to the owned/positive class than the non-owned/negative class. For final
recommendations, the score is mainly used for ranking; the threshold is a
diagnostic, not a hard business rule.

Current recommendation score distribution:

| Statistic | Value | Meaning |
|---|---:|---|
| min | 0.000161 | Lowest recommendation score; almost no modeled fit. |
| median | 0.064692 | Half of recommendation rows are below this score. Most candidates are low-fit. |
| p90 | 0.710515 | 10 percent of recommendation rows score above this level. |
| p99 | 0.935525 | Top 1 percent of recommendation rows are very high-confidence fits. |
| max | 0.994671 | Highest observed recommendation score. |

Current inspected run:

- Training frame: `2,547,576` rows and `90` columns.
- Positive rows: `907,555`.
- Negative rows: `1,640,021`.
- Positive rate: `35.62%`.
- Scoring frame: `2,660,338` rows.
- Holdout predictions: `510,785` rows.
- Top recommendation rows: `615,166`.
- Customers with recommendations: `208,000`.
- Products appearing in recommendations: `77`.
- Rows above `0.5`: `100,782`.
- Products with score above `0.5`: `63`.
- Selected model: `hist_gradient_boosting`.
