from __future__ import annotations

import json
from unittest.mock import patch

from django.core.management.base import BaseCommand

from banking.orchestration.graph import graph


class Command(BaseCommand):
    help = "Run sanitized sample banking chatbot graph requests."

    def add_arguments(self, parser):
        parser.add_argument("--live", action="store_true", help="Use live Ollama, RAG, and Django services.")

    def handle(self, *args, **options):
        samples = [
            {"message": "What documents are needed to open an account?", "customer_id": None, "authenticated": False},
            {"message": "What product do you recommend for me?", "customer_id": 1, "authenticated": True},
            {"message": "Why was my loan rejected?", "customer_id": 1, "authenticated": True},
        ]
        if not options["live"]:
            with patch("banking.orchestration.nodes._llm_generate", side_effect=self._fake_llm),
                patch("banking.orchestration.nodes.rag_tool") as mocked_rag,
                patch("banking.orchestration.nodes.services.get_customer_recommendations") as mocked_recommendations:
                mocked_rag.query.return_value = {
                    "answered": True,
                    "answer": "Verified public account-opening information.",
                    "citations": [{"citation_id": "S1"}],
                    "evidence_count": 1,
                }
                mocked_recommendations.return_value = [
                    {"product_id": "P2", "product_name": "Carte CIB", "rank": 1, "score": 0.84, "reason": "Stored recommendation evidence."}
                ]
                self._run_samples(samples)
            return

        self._run_samples(samples)

    def _run_samples(self, samples):
        for sample in samples:
            result = graph.invoke(sample)
            self.stdout.write(
                " | ".join(
                    [
                        f"intent={result.get('intent')}",
                        f"route={result.get('route')}",
                        f"escalated={bool(result.get('blocked') or result.get('requires_human') or result.get('insufficient_evidence'))}",
                        f"final_response={result.get('final_answer')}",
                    ]
                )
            )

    def _fake_llm(self, prompt):
        if "Return only JSON" in prompt:
            if "documents are needed" in prompt:
                intent = "PUBLIC_INFO"
            elif "recommend for me" in prompt:
                intent = "RECOMMENDATION"
            else:
                intent = "UNKNOWN"
            return json.dumps({"intent": intent, "confidence": 0.95, "reason": "demo"})
        return "Grounded demo answer from selected evidence."
