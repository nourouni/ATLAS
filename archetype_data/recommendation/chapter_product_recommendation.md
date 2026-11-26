# Hybrid Product Recommendation Chapter

## Objective

The final recommendation stage combines the product propensity model with the
OWL ontology. Product propensity gives an individual customer-product likelihood
score. The ontology adds symbolic reasoning: segment needs, product-need links,
product relations, requirements, conflicts, and explanation evidence.

The goal is not to let either component dominate blindly. The statistical model
answers:

```tex
How likely does this product look for this customer?
```

The ontology answers:

```tex
Is this product logically valid, semantically relevant, and explainable?
```

## Architecture

The final stage is split into four layers:

```tex
Customer
  -> candidate generation
  -> rule-based reasoning
  -> ranking layer
  -> explanation layer
  -> recommendation
```

This separation is important because hard business logic must happen before
ranking. A product with a high propensity score should still be removed if it is
already owned, incompatible, or missing a required prerequisite.

## Candidate Generation

Candidate generation creates the initial pool of possible products.

The ontology contributes products through paths such as:

```tex
Behavioral segment -> likely need -> product serving that need
```

The propensity model contributes products through predicted scores for products
the customer does not already own.

The final candidate pool is the union of both sources. This means the system can
recommend products that are semantically relevant and can also include
high-propensity products that were not reached through an ontology need path.

## Rule-Based Reasoning

After candidate generation, the system applies symbolic checks. This layer
answers:

```tex
Is this candidate logically valid?
```

The implemented hard checks include:

- remove products already owned by the customer;
- remove products whose required prerequisite product is not owned;
- remove products negated by, or incompatible with, an owned product when such
  ontology relations exist.

The layer also records soft supporting evidence:

- the product serves an inferred customer need;
- the product is related to an owned product;
- the product is part of a package or contains a product already owned;
- the product has an upgrade relation with an owned product;
- the product has a high portfolio affinity signal.

Hard evidence decides whether to keep or remove a candidate. Soft evidence is
used to explain and support recommendations.

## Ranking

Only valid candidates are ranked.

The baseline ranking formula is:

```tex
final_score =
  0.62 * propensity_score
  + 0.25 * need_match
  + 0.08 * relation_score
  + 0.05 * affinity_score
```

where:

- `propensity_score` is the product propensity model score;
- `need_match` is `1` if the ontology says the candidate serves an inferred
  need for the customer, otherwise `0`.
- `relation_score` rewards ontology product-product evidence between an owned
  product and the candidate;
- `affinity_score` uses the product affinity signal from the propensity output.

This baseline demonstrates the value of a hybrid recommender. A product with a
slightly lower propensity can rank above another product when it directly serves
an inferred need, has clear ontology support from the existing portfolio, or has
strong product-affinity evidence.

## Explanation

The explanation layer uses stored reasoning evidence. It does not invent a
description after the fact.

Each recommendation includes:

- product name;
- rank;
- propensity score;
- final score;
- need match;
- eligibility status;
- public eligibility-rule evidence when such rules are present;
- conflicts;
- supporting relations;
- a natural-language explanation.

Example explanation:

```tex
Term Deposit is recommended because it serves a need inferred for the customer
segment, is related to a product already owned by the customer, and has a 72%
product propensity score. No blocking conflict was found.
```

## Evaluation Plan

The next experiment should compare:

- propensity-only recommendations;
- ontology-only recommendations;
- hybrid ontology + propensity recommendations.

The hybrid system should demonstrate that the ontology adds value by removing
impossible recommendations, prioritizing need-relevant products, and producing
traceable explanations.
