from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("dunnhumby", "0009_churnwindowcache_training_dataset_blob")]

    operations = [
        migrations.AddField(
            model_name="churnexperiment",
            name="classification_threshold",
            field=models.FloatField(default=0.50),
        ),
        migrations.AddField(
            model_name="churnexperiment",
            name="true_positive",
            field=models.PositiveIntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="churnexperiment",
            name="false_positive",
            field=models.PositiveIntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="churnexperiment",
            name="true_negative",
            field=models.PositiveIntegerField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="churnexperiment",
            name="false_negative",
            field=models.PositiveIntegerField(blank=True, null=True),
        ),
    ]
