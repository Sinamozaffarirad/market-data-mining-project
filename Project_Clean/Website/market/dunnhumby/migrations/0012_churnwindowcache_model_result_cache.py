from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("dunnhumby", "0011_churnexperiment_training_metadata")]

    operations = [
        migrations.AddField(
            model_name="churnwindowcache",
            name="current_scores_blob",
            field=models.BinaryField(blank=True, editable=False, null=True),
        ),
        migrations.AddField(
            model_name="churnwindowcache",
            name="historical_predictions_blob",
            field=models.BinaryField(blank=True, editable=False, null=True),
        ),
        migrations.AddField(
            model_name="churnwindowcache",
            name="model_metrics_json",
            field=models.TextField(default="{}", editable=False),
        ),
        migrations.AddField(
            model_name="churnwindowcache",
            name="model_result_cached_at",
            field=models.DateTimeField(blank=True, editable=False, null=True),
        ),
        migrations.AddField(
            model_name="churnwindowcache",
            name="model_result_signature",
            field=models.CharField(blank=True, default="", editable=False, max_length=64),
        ),
    ]
