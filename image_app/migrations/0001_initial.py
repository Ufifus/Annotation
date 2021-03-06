# Generated by Django 4.0.4 on 2022-05-30 19:52

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ImageModel',
            fields=[
                ('imageId', models.BigAutoField(primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=200)),
                ('height', models.IntegerField()),
                ('width', models.IntegerField()),
                ('date', models.DateTimeField(auto_now=True)),
                ('ppixels', models.IntegerField(default=0)),
                ('npixels', models.IntegerField(default=0)),
                ('nopixels', models.IntegerField(default=0)),
                ('superpixel_modelId', models.IntegerField(default=-1)),
                ('image', models.ImageField(height_field='height', upload_to='images', width_field='width')),
            ],
        ),
        migrations.CreateModel(
            name='ProjectModel',
            fields=[
                ('projId', models.BigAutoField(primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=150, unique=True)),
                ('description', models.TextField()),
                ('date', models.DateTimeField(auto_now=True)),
            ],
        ),
        migrations.CreateModel(
            name='RoisModel',
            fields=[
                ('roiId', models.BigAutoField(primary_key=True, serialize=False)),
                ('name', models.CharField(max_length=200)),
                ('height', models.IntegerField()),
                ('width', models.IntegerField()),
                ('date', models.DateTimeField(auto_now=True)),
                ('x', models.IntegerField()),
                ('y', models.IntegerField()),
                ('nobjects', models.IntegerField(default=0)),
                ('testingROI', models.IntegerField(default=-1)),
                ('imageROI', models.ImageField(height_field='height', upload_to='images', width_field='width')),
                ('imageId', models.ForeignKey(db_column='id', null=True, on_delete=django.db.models.deletion.CASCADE, to='image_app.imagemodel')),
            ],
        ),
        migrations.AddField(
            model_name='imagemodel',
            name='projId',
            field=models.ForeignKey(db_column='id', null=True, on_delete=django.db.models.deletion.CASCADE, to='image_app.projectmodel'),
        ),
    ]
