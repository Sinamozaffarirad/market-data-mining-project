from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [("dunnhumby", "0006_rename_dunnhumby_c_experim_91fdb1_idx_dunnhumby_c_experim_2e793c_idx_and_more")]

    operations = [
        migrations.CreateModel(
            name="CustomerWindowHistory",
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
                ("churn_probability", models.FloatField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("experiment", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="window_history", to="dunnhumby.churnexperiment")),
            ],
            options={"ordering": ["cutoff_day", "household_key"]},
        ),
        migrations.AddConstraint(
            model_name="customerwindowhistory",
            constraint=models.UniqueConstraint(fields=("experiment", "household_key", "cutoff_day"), name="unique_window_history_per_experiment"),
        ),
        migrations.AddIndex(
            model_name="customerwindowhistory",
            index=models.Index(fields=["experiment", "household_key", "cutoff_day"], name="dunnhumby_c_exphhcut_idx"),
        ),
    ]
