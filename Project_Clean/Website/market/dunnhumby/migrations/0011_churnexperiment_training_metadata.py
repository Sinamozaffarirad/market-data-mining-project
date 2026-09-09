from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("dunnhumby", "0010_churnexperiment_evaluation_fields")]

    operations = [
        migrations.AddField(
            model_name="churnexperiment",
            name="training_metadata",
            field=models.TextField(default="{}", editable=False),
        ),
    ]
