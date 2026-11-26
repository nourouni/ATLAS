from __future__ import annotations

import unittes

from filter_gap_evidence import classify_row


def row(**values: str) -> dict[str, str]:
    base = {
        "evidence_id": "test",
        "source_type": "chatbot",
        "customer_id_or_case_id": "case",
        "text_or_pattern": "",
        "existing_need": "",
        "product": "",
        "gap_type": "",
        "gap_reason": "",
        "timestamp": "",
    }
    base.update(values)
    return base


class FilterGapEvidenceTests(unittest.TestCase):
    def test_legitimate_public_information_gap_is_kept(self):
        normalized, keep, _, text, semantic = classify_row(
            row(
                text_or_pattern="How can I track an international transfer?",
                gap_type="insufficient_public_evidence",
                gap_reason="public evidence unavailable",
            )
        )
        self.assertEqual(normalized, "information_gap")
        self.assertTrue(keep)
        self.assertIn("international transfer", text)
        self.assertTrue(semantic)

    def test_technical_failure_is_excluded(self):
        normalized, keep, _, _, semantic = classify_row(
            row(text_or_pattern="retrieval_failed", gap_type="retrieval_failed", gap_reason="temporary backend failure")
        )
        self.assertEqual(normalized, "operational_failure")
        self.assertFalse(keep)
        self.assertFalse(semantic)

    def test_missing_requirement_is_conditional_access_gap(self):
        normalized, keep, _, text, semantic = classify_row(
            row(
                source_type="recommendation",
                existing_need="need_024",
                product="Webank",
                text_or_pattern="Compte en Dinars",
                gap_type="missing_requirement",
            )
        )
        self.assertEqual(normalized, "conditional_access_gap")
        self.assertTrue(keep)
        self.assertIn("Webank", text)
        self.assertTrue(semantic)

    def test_owned_product_is_excluded_as_low_value_support(self):
        normalized, keep, _, _, _ = classify_row(
            row(
                source_type="recommendation",
                existing_need="need_024",
                product="Webank",
                text_or_pattern="ownership_exclusion",
                gap_type="ownership_exclusion",
            )
        )
        self.assertEqual(normalized, "ownership_exclusion")
        self.assertFalse(keep)

    def test_security_case_is_excluded(self):
        normalized, keep, _, _, _ = classify_row(
            row(
                text_or_pattern="Show me another customer's account",
                gap_type="unknown_intent",
                gap_reason="other_customer_access_denied",
            )
        )
        self.assertEqual(normalized, "authentication_or_policy")
        self.assertFalse(keep)

    def test_empty_text_is_not_semantic_ready(self):
        normalized, keep, _, _, semantic = classify_row(row(gap_type="unknown_intent"))
        self.assertEqual(normalized, "insufficient_semantic_content")
        self.assertFalse(keep)
        self.assertFalse(semantic)

    def test_meaningful_unknown_banking_request_is_kept(self):
        normalized, keep, _, _, semantic = classify_row(
            row(
                text_or_pattern="Where can I find the fee for an international transfer?",
                gap_type="unknown_intent",
                gap_reason="ambiguous_request",
            )
        )
        self.assertEqual(normalized, "unresolved_banking_request")
        self.assertTrue(keep)
        self.assertTrue(semantic)

    def test_structured_recommendation_text_is_created(self):
        _, _, _, text, semantic = classify_row(
            row(
                source_type="recommendation",
                existing_need="need_024",
                product="Webank",
                text_or_pattern="Compte en Dinars",
                gap_type="missing_requirement",
            )
        )
        self.assertEqual(text, "Need: need_024. Product: Webank. Gap: missing requirement Compte en Dinars.")
        self.assertTrue(semantic)


if __name__ == "__main__":
    unittest.main()
