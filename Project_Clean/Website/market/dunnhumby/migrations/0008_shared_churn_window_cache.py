from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [("dunnhumby", "0007_customerwindowhistory")]

    operations = [
        migrations.CreateModel(
            name="ChurnWindowCache",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("method", models.CharField(max_length=24)),
                ("observation_window_days", models.PositiveIntegerField()),
                ("prediction_horizon_days", models.PositiveIntegerField()),
                ("step_size_days", models.PositiveIntegerField()),
                ("dataset_signature", models.CharField(max_length=128)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.AddConstraint(
            model_name="churnwindowcache",
            constraint=models.UniqueConstraint(fields=("method", "observation_window_days", "prediction_horizon_days", "step_size_days", "dataset_signature"), name="unique_churn_window_cache"),
        ),
        migrations.AddField(
            model_name="churnexperiment",
            name="window_cache",
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name="experiments", to="dunnhumby.churnwindowcache"),
        ),
        migrations.CreateModel(
            name="CachedCustomerWindow",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("household_key", models.BigIntegerField()),
                ("observation_start", models.IntegerField()),
                ("observation_end", models.IntegerField()),
                ("cutoff_day", models.IntegerField()),
                ("label_start", models.IntegerField()),
                ("label_end", models.IntegerField()),
                ("recency_days", models.FloatField()),
                ("frequency", models.FloatField()),
                ("monetary", models.FloatField()),
                ("r_score", models.PositiveSmallIntegerField()),
                ("f_score", models.PositiveSmallIntegerField()),
                ("m_score", models.PositiveSmallIntegerField()),
                ("rfm_segment", models.CharField(max_length=32)),
                ("is_churn", models.BooleanField(help_text="No purchase occurred during this row's complete prediction horizon.")),
                ("cache", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="customer_windows", to="dunnhumby.churnwindowcache")),
            ],
            options={"ordering": ["cutoff_day", "household_key"]},
        ),
        migrations.AddConstraint(
            model_name="cachedcustomerwindow",
            constraint=models.UniqueConstraint(fields=("cache", "household_key", "cutoff_day"), name="unique_cached_customer_window"),
        ),
        migrations.AddIndex(
            model_name="cachedcustomerwindow",
            index=models.Index(fields=["cache", "household_key", "cutoff_day"], name="dunnhumby_cachehhcut_idx"),
        ),
        migrations.CreateModel(
            name="ChurnExperimentWindowPrediction",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("churn_probability", models.FloatField()),
                ("experiment", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="cached_history_predictions", to="dunnhumby.churnexperiment")),
                ("window", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="experiment_predictions", to="dunnhumby.cachedcustomerwindow")),
            ],
        ),
        migrations.AddConstraint(
            model_name="churnexperimentwindowprediction",
            constraint=models.UniqueConstraint(fields=("experiment", "window"), name="unique_cached_window_prediction"),
        ),
        migrations.AddIndex(
            model_name="churnexperimentwindowprediction",
            index=models.Index(fields=["experiment", "window"], name="dunnhumby_expwin_idx"),
        ),
    ]
