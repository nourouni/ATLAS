from __future__ import annotations

import unittes

from .behavioral_segment_presentation import (
    ACTIVE_BEHAVIORAL_SEGMENT_IDS,
    behavioral_segment_definitions,
    resolve_behavioral_segment,
)


class BehavioralSegmentPresentationTests(unittest.TestCase):
    def test_all_current_retail_and_enterprise_ids_resolve(self):
        definitions = behavioral_segment_definitions()
        for segment_id in ACTIVE_BEHAVIORAL_SEGMENT_IDS:
            self.assertIn(segment_id, definitions)
            self.assertTrue(definitions[segment_id]["display_name"])
            self.assertTrue(definitions[segment_id]["description"])

    def test_obsolete_retail_ids_are_not_active(self):
        definitions = behavioral_segment_definitions()
        self.assertNotIn("seg_ret_5", definitions)
        self.assertNotIn("seg_ret_6", definitions)

    def test_unknown_ids_use_neutral_fallback(self):
        resolved = resolve_behavioral_segment("seg_unknown")
        self.assertEqual(resolved["segment_id"], "seg_unknown")
        self.assertEqual(resolved["display_name"], "Unclassified behavior")

    def test_business_category_and_behavioral_id_are_separate(self):
        resolved = resolve_behavioral_segment("seg_ret_2", {"segment": "Jeune"})
        self.assertEqual(resolved["segment"], "Jeune")
        self.assertEqual(resolved["segment_id"], "seg_ret_2")
        self.assertNotEqual(resolved["display_name"], "Jeune")\n