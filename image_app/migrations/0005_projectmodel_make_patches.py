# Generated by Django 4.0.4 on 2022-06-10 16:18

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('image_app', '0004_remove_roismodel_name_projectmodel_embed_iteration_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='projectmodel',
            name='make_patches',
            field=models.BooleanField(default=False),
        ),
    ]
