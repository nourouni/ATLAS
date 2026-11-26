from __future__ import annotations

from django.core.management.base import BaseCommand

from banking.evaluation.dataset import DEFAULT_DATASET_PATH, write_datase


class Command(BaseCommand):
    help = "Build the reproducible chatbot evaluation dataset from verified RAG and safe Django services."

    def add_arguments(self, parser):
        parser.add_argument("--customer-id", type=int, default=169887)
        parser.add_argument("--other-customer-id", type=int, default=135553)
        parser.add_argument("--output", default=str(DEFAULT_DATASET_PATH))

    def handle(self, *args, **options):
        cases = write_dataset(path=DEFAULT_DATASET_PATH.__class__(options["output"]), customer_id=options["customer_id"], other_customer_id=options["other_customer_id"])
        self.stdout.write(f"Wrote {len(cases)} cases to {options['output']}")
