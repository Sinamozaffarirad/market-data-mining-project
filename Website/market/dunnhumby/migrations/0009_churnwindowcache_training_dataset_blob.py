from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("dunnhumby", "0008_shared_churn_window_cache")]

    operations = [
        migrations.AddField(
            model_name="churnwindowcache",
            name="training_dataset_blob",
            field=models.BinaryField(blank=True, editable=False, null=True),
        ),
    ]
