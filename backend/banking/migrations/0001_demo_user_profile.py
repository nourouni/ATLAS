from __future__ import annotations

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="DemoUserProfile",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                (
                    "role",
                    models.CharField(
                        choices=[("customer", "Customer"), ("staff", "Staff/Advisor"), ("admin", "Admin")],
                        max_length=20,
                    ),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("customer_id", models.BigIntegerField(blank=True, null=True)),
                (
                    "user",
                    models.OneToOneField(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="demo_profile",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "db_table": "banking_user_profile",
            },
        ),
        migrations.AddConstraint(
            model_name="demouserprofile",
            constraint=models.UniqueConstraint(fields=("customer_id",), name="unique_demo_customer_mapping"),
        ),
        migrations.AddConstraint(
            model_name="demouserprofile",
            constraint=models.CheckConstraint(
                condition=(
                    models.Q(("customer_id__isnull", False), ("role", "customer"))
                    | models.Q(("customer_id__isnull", True), ("role__in", ["staff", "admin"]))
                ),
                name="valid_demo_user_profile_role_customer",
            ),
        ),
    ]
