# Generated by Django 4.0.4 on 2022-06-12 22:17

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('image_app', '0008_imagemodel_save_in_db'),
    ]

    operations = [
        migrations.AddField(
            model_name='imagemodel',
            name='superpixel_created',
            field=models.BooleanField(default=False),
        ),
    ]
