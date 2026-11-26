# Research-dependent tests

These six tests are preserved separately from the essential portfolio suite:

- `test_group_gap_evidence.py`: two tests requiring the excluded generated gap-evidence inputs.
- `test_label_candidate_needs.py`: two tests requiring the excluded semantic-group and ontology inspection artifacts.
- `test_review_candidate_needs.py`: two tests requiring the excluded taxonomy/review artifacts and a writable generated-output directory.

They are not part of the bounded portfolio smoke command because those original research artifacts are intentionally excluded. Run them only after supplying the original research inputs in an isolated working copy; do not point them at an operational database.
