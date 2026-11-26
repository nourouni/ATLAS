from __future__ import annotations

from django.conf import settings
from django.db import models


class ModelRun(models.Model):
    run_id = models.BigAutoField(primary_key=True)
    run_name = models.TextField()
    model_type = models.TextField()
    source_path = models.TextField(blank=True, null=True)
    artifact_version = models.TextField(blank=True, null=True)
    parameters = models.JSONField()
    metrics = models.JSONField()
    started_at = models.DateTimeField(blank=True, null=True)
    completed_at = models.DateTimeField(blank=True, null=True)
    created_at = models.DateTimeField()

    class Meta:
        managed = False
        db_table = "model_runs"


class Product(models.Model):
    product_id = models.TextField(primary_key=True)
    product_name = models.TextField(unique=True)
    alias_notes = models.TextField(blank=True, null=True)
    category = models.TextField(blank=True, null=True)
    classification = models.TextField(blank=True, null=True)
    product_type = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField()
    updated_at = models.DateTimeField()

    class Meta:
        managed = False
        db_table = "products"


class Customer(models.Model):
    customer_id = models.BigIntegerField(primary_key=True)
    segment_label = models.TextField(blank=True, null=True)
    behavioral_segment = models.ForeignKey(
        "CustomerSegment",
        models.DO_NOTHING,
        db_column="behavioral_segment_id",
        blank=True,
        null=True,
    )
    forecast_pool = models.TextField(blank=True, null=True)
    latent_archetype = models.TextField(blank=True, null=True)
    customer_class = models.TextField(blank=True, null=True)
    employment = models.TextField(blank=True, null=True)
    income_decile = models.IntegerField(blank=True, null=True)
    tenure_days = models.IntegerField(blank=True, null=True)
    source_system = models.TextField()
    created_at = models.DateTimeField()
    updated_at = models.DateTimeField()

    class Meta:
        managed = False
        db_table = "customers"


class CustomerSegment(models.Model):
    segment_id = models.TextField(primary_key=True)
    segment_name = models.TextField()
    source_group = models.TextField(blank=True, null=True)
    cluster_id = models.IntegerField(blank=True, null=True)
    description = models.TextField(blank=True, null=True)
    transaction_behavior = models.TextField(blank=True, null=True)
    balance_and_cashflow_behavior = models.TextField(blank=True, null=True)
    digital_channel_behavior = models.TextField(blank=True, null=True)
    likely_customer_objectives = models.TextField(blank=True, null=True)
    potential_risks = models.TextField(blank=True, null=True)
    supporting_statistics = models.TextField(blank=True, null=True)
    interpretation_confidence = models.TextField(blank=True, null=True)
    requires_manual_review = models.BooleanField()
    created_at = models.DateTimeField()

    class Meta:
        managed = False
        db_table = "customer_segments"


class CustomerProduct(models.Model):
    customer_id = models.BigIntegerField(primary_key=True, db_column="customer_id")
    product = models.ForeignKey(Product, models.DO_NOTHING, db_column="product_id")
    product_model_segment = models.TextField(blank=True, null=True)
    family = models.TextField(blank=True, null=True)
    run = models.ForeignKey(ModelRun, models.DO_NOTHING, db_column="run_id")
    loaded_at = models.DateTimeField()

    class Meta:
        managed = False
        db_table = "customer_products"


class PropensityPrediction(models.Model):
    customer_id = models.BigIntegerField(primary_key=True, db_column="customer_id")
    product = models.ForeignKey(Product, models.DO_NOTHING, db_column="product_id")
    propensity_score = models.DecimalField(max_digits=8, decimal_places=6)
    segment_product_prior = models.DecimalField(max_digits=8, decimal_places=6, blank=True, null=True)
    max_owned_to_candidate_affinity = models.DecimalField(max_digits=8, decimal_places=6, blank=True, null=True)
    mean_owned_to_candidate_affinity = models.DecimalField(max_digits=8, decimal_places=6, blank=True, null=True)
    rank = models.IntegerField(blank=True, null=True)
    product_model_segment = models.TextField(blank=True, null=True)
    family = models.TextField(blank=True, null=True)
    run = models.ForeignKey(ModelRun, models.DO_NOTHING, db_column="run_id")
    scored_at = models.DateTimeField()

    class Meta:
        managed = False
        db_table = "propensity_predictions"


class Recommendation(models.Model):
    customer_id = models.BigIntegerField(primary_key=True, db_column="customer_id")
    product = models.ForeignKey(Product, models.DO_NOTHING, db_column="product_id")
    rank = models.IntegerField()
    score = models.DecimalField(max_digits=8, decimal_places=6)
    reason = models.TextField(blank=True, null=True)
    run = models.ForeignKey(ModelRun, models.DO_NOTHING, db_column="run_id")
    created_at = models.DateTimeField()

    class Meta:
        managed = False
        db_table = "recommendations"


class SegmentForecast(models.Model):
    forecast_id = models.BigAutoField(primary_key=True)
    pool = models.TextField()
    metric = models.TextField()
    forecast_date = models.DateField()
    scenario_case = models.TextField()
    model_name = models.TextField()
    actual_value = models.DecimalField(max_digits=20, decimal_places=6, blank=True, null=True)
    predicted_value = models.DecimalField(max_digits=20, decimal_places=6)
    abs_error = models.DecimalField(max_digits=20, decimal_places=6, blank=True, null=True)
    run = models.ForeignKey(ModelRun, models.DO_NOTHING, db_column="run_id")
    created_at = models.DateTimeField()

    class Meta:
        managed = False
        db_table = "segment_forecasts"


class DemoUserProfile(models.Model):
    ROLE_CUSTOMER = "customer"
    ROLE_STAFF = "staff"
    ROLE_ADMIN = "admin"
    ROLE_CHOICES = [
        (ROLE_CUSTOMER, "Customer"),
        (ROLE_STAFF, "Staff/Advisor"),
        (ROLE_ADMIN, "Admin"),
    ]

    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="demo_profile")
    customer_id = models.BigIntegerField(blank=True, null=True)
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "banking_user_profile"
        constraints = [
            models.UniqueConstraint(fields=["customer_id"], name="unique_demo_customer_mapping"),
            models.CheckConstraint(
                check=(
                    models.Q(role="customer", customer_id__isnull=False)
                    | models.Q(role__in=["staff", "admin"], customer_id__isnull=True)
                ),
                name="valid_demo_user_profile_role_customer",
            ),
        ]
