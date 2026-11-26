from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import ollama
from django.conf import settings
from django.contrib.auth.models import AnonymousUser, User
from django.core.management.base import BaseCommand
from django.db import connection
from django.test import RequestFactory

from banking.chatbot_views import chatbot_graph
from banking.evaluation.dataset import load_datase
from banking.orchestration import nodes


ARTIFACT_DIR = Path("artifacts")
EVAL_DIR = ARTIFACT_DIR / "chatbot_evaluation"
DEMO_DIR = ARTIFACT_DIR / "demo"
SELECTED_CASE_IDS = ["PUB-001", "CUST-001", "REC-001", "REX-001", "UNAUTH-001", "TXN-001", "HUM-001"]
SENSITIVE_NUMBER_RE = re.compile(r"\b\d{6,}\b")


class Command(BaseCommand):
    help = "Run the frozen chatbot v3 demonstration examples with sanitized output."

    def add_arguments(self, parser):
        parser.add_argument("--dataset", default=str(EVAL_DIR / "chatbot_eval_cases.jsonl"))
        parser.add_argument("--fallback-results", default=str(EVAL_DIR / "chatbot_eval_results_targeted_fixes_v3.csv"))
        parser.add_argument("--output", default=str(DEMO_DIR / "chatbot_v3_demo_outputs.json"))

    def handle(self, *args, **options):
        output_path = Path(options["output"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        health = self._health()
        live_available = all(health.values())
        cases = {case["case_id"]: case for case in load_dataset(Path(options["dataset"]))}
        rows = []
        if live_available:
            for case_id in SELECTED_CASE_IDS:
                rows.append(self._run_case(cases[case_id]))
        else:
            rows = self._fallback_rows(Path(options["fallback_results"]))
        payload = {
            "run_identifier": "chatbot-v3-final-demo",
            "health": health,
            "mode": "live" if live_available else "preserved-v3-output-fallback",
            "examples": rows,
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        for row in rows:
            self.stdout.write(
                f"{row['case_id']} | intent={row['intent']} | route={row['route']} | escalated={row['escalated']}\n"
                f"Q: {row['question']}\nA: {row['answer']}\n"
            )
        self.stdout.write(f"Saved {output_path}")

    def _health(self) -> dict[str, bool]:
        health = {"django": True, "postgresql": False, "rag": False, "ollama": False}
        try:
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
                health["postgresql"] = cursor.fetchone()[0] == 1
        except Exception:
            health["postgresql"] = False
        try:
            health["rag"] = nodes.rag_tool.health().get("status") in {"ok", "degraded"}
        except Exception:
            health["rag"] = False
        try:
            client = ollama.Client(host=getattr(settings, "CHATBOT_OLLAMA_HOST", None), timeout=5)
            models = client.list()
            installed = {
                item.get("name") or item.get("model")
                for item in getattr(models, "models", None) or models.get("models", [])
            }
            model = getattr(settings, "CHATBOT_OLLAMA_MODEL", "phi4-mini")
            health["ollama"] = any(name == model or str(name).startswith(f"{model}:") for name in installed)
        except Exception:
            health["ollama"] = False
        return health

    def _run_case(self, case: dict[str, Any]) -> dict[str, Any]:
        factory = RequestFactory()
        auth = case["authentication"]
        body = {"message": case["question"]}
        if auth.get("body_customer_id") is not None:
            body["customer_id"] = auth["body_customer_id"]
        request = factory.post("/api/chatbot/graph/", data=json.dumps(body), content_type="application/json", HTTP_X_REQUEST_ID=f"demo-{case['case_id']}-{time.time_ns()}")
        request.user = self._user(auth.get("customer_id") if auth.get("authenticated") else None)
        usage = {"rag": 0, "customer_context": 0, "recommendations": 0}
        real_rag = nodes.rag_tool.query
        real_profile = nodes.services.get_customer_profile
        real_recs = nodes.services.get_customer_recommendations

        def rag_wrap(*args, **kwargs):
            usage["rag"] += 1
            return real_rag(*args, **kwargs)

        def profile_wrap(*args, **kwargs):
            usage["customer_context"] += 1
            return real_profile(*args, **kwargs)

        def recs_wrap(*args, **kwargs):
            usage["recommendations"] += 1
            return real_recs(*args, **kwargs)

        with patch("banking.orchestration.nodes.rag_tool.query", side_effect=rag_wrap),
            patch("banking.orchestration.nodes.services.get_customer_profile", side_effect=profile_wrap),
            patch("banking.orchestration.nodes.services.get_customer_recommendations", side_effect=recs_wrap):
            response = chatbot_graph(request, customer_id=auth.get("path_customer_id"))
        payload = json.loads(response.content.decode("utf-8"))
        return self._demo_row(case, payload, usage)

    def _fallback_rows(self, path: Path) -> list[dict[str, Any]]:
        import csv

        rows = []
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row["case_id"] not in SELECTED_CASE_IDS:
                    continue
                source_usage = json.loads(row.get("source_usage") or "{}")
                rows.append(
                    {
                        "case_id": row["case_id"],
                        "question": row["question"],
                        "intent": row.get("predicted_intent") or "",
                        "route": row.get("actual_route") or "",
                        "escalated": row.get("actual_escalated") == "True",
                        "answer": self._sanitize(row.get("final_answer") or ""),
                        "source_type": self._source_type(source_usage, row.get("actual_route") or ""),
                        "mode": "preserved-v3-output-fallback",
                    }
                )
        return sorted(rows, key=lambda item: SELECTED_CASE_IDS.index(item["case_id"]))

    def _demo_row(self, case: dict[str, Any], payload: dict[str, Any], usage: dict[str, int]) -> dict[str, Any]:
        return {
            "case_id": case["case_id"],
            "question": case["question"],
            "intent": payload.get("intent") or "",
            "route": payload.get("route") or "",
            "escalated": bool(payload.get("escalated")),
            "answer": self._sanitize(payload.get("answer") or ""),
            "source_type": self._source_type(usage, payload.get("route") or ""),
            "mode": "live",
        }

    def _source_type(self, usage: dict[str, int], route: str) -> str:
        if usage.get("rag"):
            return "rag"
        if usage.get("customer_context"):
            return "customer_context"
        if usage.get("recommendations"):
            return "recommendations"
        if route == "ESCALATE":
            return "deterministic_policy"
        return "none"

    def _user(self, customer_id: int | None):
        if customer_id is None:
            return AnonymousUser()
        user = User(username=f"demo-{customer_id}")
        user.customer_id = customer_id
        return user

    def _sanitize(self, text: str) -> str:
        return SENSITIVE_NUMBER_RE.sub("EVAL_CUSTOMER", text)
