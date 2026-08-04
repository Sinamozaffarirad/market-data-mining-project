# Generated manually for the additive churn experiment feature.
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [("dunnhumby", "0002_alter_customersegment_churn_probability")]

    operations = [
        migrations.CreateModel(
            name="ChurnExperiment",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("method", models.CharField(max_length=24)), ("observation_window_days", models.PositiveIntegerField()),
                ("prediction_horizon_days", models.PositiveIntegerField()), ("step_size_days", models.PositiveIntegerField()),
                ("accuracy", models.FloatField(blank=True, null=True)), ("precision", models.FloatField(blank=True, null=True)),
                ("recall", models.FloatField(blank=True, null=True)), ("f1", models.FloatField(blank=True, null=True)),
                ("roc_auc", models.FloatField(blank=True, null=True)), ("pr_auc", models.FloatField(blank=True, null=True)),
                ("training_samples", models.PositiveIntegerField(default=0)), ("validation_samples", models.PositiveIntegerField(default=0)),
                ("test_samples", models.PositiveIntegerField(default=0)), ("churn_rate", models.FloatField(blank=True, null=True)),
                ("training_time_seconds", models.FloatField(blank=True, null=True)), ("prediction_time_seconds", models.FloatField(blank=True, null=True)),
                ("current_cutoff_day", models.IntegerField(blank=True, null=True)), ("is_active", models.BooleanField(default=False)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={"ordering": ["-created_at"]},
        ),
        migrations.CreateModel(
            name="ChurnCustomerScore",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("household_key", models.BigIntegerField()), ("churn_probability", models.FloatField()),
                ("experiment", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="scores", to="dunnhumby.churnexperiment")),
            ],
        ),
        migrations.AddConstraint(model_name="churncustomerscore", constraint=models.UniqueConstraint(fields=("experiment", "household_key"), name="unique_churn_score_per_experiment")),
        migrations.AddIndex(model_name="churncustomerscore", index=models.Index(fields=["experiment", "churn_probability"], name="dunnhumby__experi_5be373_idx")),
    ]
