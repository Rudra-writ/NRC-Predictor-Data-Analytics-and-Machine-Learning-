# Generated by Django 4.0.1 on 2022-02-09 12:59

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('NRCpredictor', '0012_yearlygraphdata'),
    ]

    operations = [
        migrations.AddField(
            model_name='yearlygraphdata',
            name='total',
            field=models.FloatField(blank=True, null=True),
        ),
    ]
