from __future__ import annotations

from django.test.runner import DiscoverRunner


class ExistingDatabaseTestRunner(DiscoverRunner):
    """Run smoke tests against the configured PostgreSQL database.

    The application database role intentionally cannot create databases, so
    these tests validate the Docker database connection in place.
    """

    def setup_databases(self, **kwargs):
        return None

    def teardown_databases(self, old_config, **kwargs):
        return None
