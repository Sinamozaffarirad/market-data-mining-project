from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [("dunnhumby", "0004_rename_dunnhumby__experi_5be373_idx_dunnhumby_c_experim_7067eb_idx")]

    operations = [
        migrations.CreateModel(name="CustomerStateSnapshot", fields=[
            ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
            ("household_key", models.BigIntegerField()), ("cutoff_day", models.IntegerField()),
            ("observation_window_days", models.PositiveIntegerField()), ("recency_days", models.FloatField()),
            ("frequency", models.FloatField()), ("monetary", models.FloatField()),
            ("r_score", models.PositiveSmallIntegerField()), ("f_score", models.PositiveSmallIntegerField()),
            ("m_score", models.PositiveSmallIntegerField()), ("rfm_segment", models.CharField(max_length=32)),
            ("created_at", models.DateTimeField(auto_now_add=True)),
        ]),
        migrations.CreateModel(name="CustomerChurnOutcome", fields=[
            ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
            ("prediction_horizon_days", models.PositiveIntegerField()), ("is_churn", models.BooleanField()),
            ("snapshot", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="outcomes", to="dunnhumby.customerstatesnapshot")),
        ]),
        migrations.CreateModel(name="CustomerChurnPrediction", fields=[
            ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
            ("churn_probability", models.FloatField()),
            ("prediction_type", models.CharField(choices=[("historical", "Historical walk-forward"), ("current", "Current forecast")], max_length=16)),
            ("created_at", models.DateTimeField(auto_now_add=True)),
            ("experiment", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="history_predictions", to="dunnhumby.churnexperiment")),
            ("snapshot", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="predictions", to="dunnhumby.customerstatesnapshot")),
        ]),
        migrations.AddConstraint(model_name="customerstatesnapshot", constraint=models.UniqueConstraint(fields=("household_key", "cutoff_day", "observation_window_days"), name="unique_customer_state_snapshot")),
        migrations.AddIndex(model_name="customerstatesnapshot", index=models.Index(fields=["household_key", "cutoff_day"], name="dunnhumby_c_househo_2182d3_idx")),
        migrations.AddConstraint(model_name="customerchurnoutcome", constraint=models.UniqueConstraint(fields=("snapshot", "prediction_horizon_days"), name="unique_snapshot_churn_outcome")),
        migrations.AddConstraint(model_name="customerchurnprediction", constraint=models.UniqueConstraint(fields=("experiment", "snapshot", "prediction_type"), name="unique_experiment_snapshot_prediction")),
        migrations.AddIndex(model_name="customerchurnprediction", index=models.Index(fields=["experiment", "prediction_type"], name="dunnhumby_c_experim_91fdb1_idx")),
    ]
