from django.test import SimpleTestCase

from banking.orchestration.conversation_state import resolve_follow_up


class ConversationStateTests(SimpleTestCase):
    def test_for_jeune_resolves_against_current_topic(self):
        resolved = resolve_follow_up(
            "for jeune",
            history=[
                {"role": "user", "content": "I am 22 and want a savings account"},
                {"role": "assistant", "content": "I could not verify a specific savings recommendation."},
            ],
            conversation_state={
                "current_topic": "savings",
                "current_audience": "young_retail",
                "last_public_query": "I am 22 and want a savings account",
            },
        )

        self.assertEqual(resolved, "I am 22 and want a savings account for young retail customers")
