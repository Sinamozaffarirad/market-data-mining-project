from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dunnhumby', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='associationrule',
            name='metadata',
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name='associationrule',
            name='min_confidence_threshold',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='associationrule',
            name='min_lift_threshold',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='associationrule',
            name='source_view',
            field=models.CharField(blank=True, default='', max_length=64),
        ),
    ]
