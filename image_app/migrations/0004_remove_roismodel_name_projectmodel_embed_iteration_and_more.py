# Generated by Django 4.0.4 on 2022-06-09 17:59

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('image_app', '0003_rename_nopixels_imagemodel_noobjects'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='roismodel',
            name='name',
        ),
        migrations.AddField(
            model_name='projectmodel',
            name='embed_iteration',
            field=models.IntegerField(default=-1),
        ),
        migrations.AddField(
            model_name='projectmodel',
            name='iteration',
            field=models.IntegerField(default=-1),
        ),
    ]
