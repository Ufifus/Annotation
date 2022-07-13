from django.db import models

from PIL import Image
import logging
import os
import glob

from Histology_Project.settings import MEDIA_ROOT


logger = logging.getLogger(__name__)

images_root = os.path.join(MEDIA_ROOT, 'images').replace('\\', '/')
proj_derictories = ['mask', 'models', 'patches', 'pred', 'roi', 'roi_mask', 'superpixels', 'superpixels_boundary']

class ProjectModel(models.Model):
    "Модель для описания проекта"
    projId = models.BigAutoField(primary_key=True)
    name = models.CharField(max_length=150, unique=True)
    description = models.TextField()
    date = models.DateTimeField(auto_now=True)
    make_patches = models.BooleanField(default=False)
    train_ae = models.BooleanField(default=False)
    iteration = models.IntegerField(default=-1)
    embed_iteration = models.IntegerField(default=-1)

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        proj_root = get_proj_root(self.name)
        if not os.path.exists(images_root):
            os.mkdir(images_root)
        if not os.path.exists(proj_root):
            os.mkdir(proj_root)
            logger.warning(f'Создаем деррикторию проекта: ' + proj_root)
            for dir in proj_derictories:
                dir_root = os.path.join('{0}/', '{1}').format(proj_root, dir)
                if not os.path.exists(dir_root):
                    logger.warning(f'Создаем деррикторию {dir_root}')
                    os.mkdir(dir_root)

    def delete(self, *args, **kwargs):
        proj_root = get_proj_root(self.name)
        logger.warning(f'Удаляем дирректорию: ' + proj_root)
        for image in ImageModel.objects.filter(projId=self):
            image.delete()
        super(ProjectModel, self).delete(*args, **kwargs)
        os.remove(proj_root)


class ImageModel(models.Model):
    imageId = models.BigAutoField(primary_key=True)
    projId = models.ForeignKey('ProjectModel', on_delete=models.CASCADE, db_column='id', null=True)
    height = models.IntegerField()
    width = models.IntegerField()
    date = models.DateTimeField(auto_now=True)
    ppixels = models.IntegerField(default=0)
    npixels = models.IntegerField(default=0)
    noobjects = models.IntegerField(default=0)
    make_batches = models.BooleanField(default=False)
    superpixel_modelId = models.IntegerField(default=-1)
    superpixel_created = models.BooleanField(default=False)
    save_in_db = models.BooleanField(default=False)
    image = models.ImageField(
        upload_to='images',
        height_field='height',
        width_field='width',
    )

    def save(self, *args, **kwargs):
        """Сохраняем изображение и его маску если формат *.png иначе конвертируем"""
        image_name, image_format = self.image.name.split('.')
        if self.save_in_db:
            super().save(*args, **kwargs)
        else:
            self.image.name = self.create_path(image_name)
            self.save_in_db = True
            super().save(*args, **kwargs)
            image = Image.open(self.image.path)
            if image_format != 'png':
                image.save(self.image.path, 'png', quality=100)
            else:
                image.save(self.image.path)
            logger.warning(f'Куда сохраняем изображение: {self.image.path}')

            logger.warning('Сохраняем маску изображения')
            mask_root = self.create_mask_root(MEDIA_ROOT)
            mask_path = self.create_mask_path(image_name, mask_root)
            self.save_mask(mask_path)

    def delete(self, using=None, keep_parents=False, *args, **kwargs):
        proj_root = get_proj_root(self.projId.name)
        image_name = self.image.name.split('/')[-1]
        for dir in proj_derictories:
            dir_root = os.path.join('{0}/', '{1}').format(proj_root, dir)
            if os.path.exists(dir_root):
                files_path = self.create_product_image_path(MEDIA_ROOT, dir, image_name)
                files = glob.glob(files_path)
                logger.warning(f'Найденно {len(files)} файлов от изображения')
                for file in files:
                    os.remove(file)

        for roi in RoisModel.objects.filter(imageId=self):
            roi.delete()
        os.remove(self.image.path)
        super().delete(*args, **kwargs)

    def save_mask(self, mask_path):
        if os.path.isfile(mask_path):
            logger.warning('Маска уже существует')
        else:
            logger.warning(f'Путь куда сохраняем mask: {mask_path}')
            mask = Image.new('RGB', (self.image.height, self.image.width))
            mask.save(mask_path, 'png')

    def create_path(self, image_name):
        """Создаем путь до нашего файла: media/images/<project_name>/<image.png>"""
        return os.path.join('{0}/', '{1}.png').format(self.projId.name, image_name)

    def create_mask_path(self, image_name, mask_root):
        """Создаем путь для маски: media/images/<project_name>/masks/<image_mask.png>"""
        return os.path.join('{0}/', '{1}_mask.png').format(mask_root, image_name)

    def create_mask_root(self, media_root):
        """Создаем путь до каталога с масками: media/images/<project_name>/masks"""
        return os.path.join('{0}/', 'images/{1}/mask').format(media_root, self.projId.name).replace('\\', '/')

    def create_product_image_path(self, media_root, dir_name, image_name):
        """Создаем путь который находит все производные от нашего изображения
                media/images/<project_name>/*/<image_name>_*.png"""
        return os.path.join(media_root, 'images/{0}/{1}/{2}_*.png').format(media_root, dir_name, image_name)


class RoisModel(models.Model):
    roiId = models.BigAutoField(primary_key=True)
    imageId = models.ForeignKey('ImageModel', on_delete=models.CASCADE, db_column='id', null=True)
    height = models.IntegerField()
    width = models.IntegerField()
    date = models.DateTimeField(auto_now=True)
    x = models.IntegerField()
    y = models.IntegerField()
    nobjects = models.IntegerField(default=0)
    testingROI = models.IntegerField(default=-1)
    roi_path = models.TextField(default='path')
    roi_name = models.TextField(default='name')

    def save(self, *args, **kwargs):
        """Сохраняем Rois"""
        super().save(*args, **kwargs)


    def delete(self, using=None, keep_parents=False, *args, **kwargs):
        super().delete(*args, **kwargs)


def get_proj_root(proj_name):
    """Получаем путь до директории с проектом"""
    proj_root = os.path.join('{0}/', 'images/{1}').format(MEDIA_ROOT, proj_name).replace('\\', '/')
    return proj_root


# class MaskModel(models.Model):
#     maskId = models.BigAutoField(primary_key=True)
#     imageId = models.ForeignKey('ImageModel', on_delete=models.CASCADE, db_column='imageId')
#     height = models.IntegerField()
#     width = models.IntegerField()
#     mask = models.ImageField(
#         upload_to='images',
#         height_field='height',
#         width_field='width',
#     )
#
# class SuperpixelsModel(models.Model):
#     superpixId = models.BigAutoField(primary_key=True)
#     imageId = models.ForeignKey('ImageModel', on_delete=models.CASCADE, db_column='imageId')
#     height = models.IntegerField()
#     width = models.IntegerField()
#     superpixel = models.ImageField(
#         upload_to='images',
#         height_field='height',
#         width_field='width',
#     )
#
# class SuperpixelBoundaryModel(models.Model):
#     superpix_boundId = models.BigAutoField(primary_key=True)
#     imageId = models.ForeignKey('ImageModel', on_delete=models.CASCADE, db_column='imageId')
#     height = models.IntegerField()
#     width = models.IntegerField()
#     superpixel_bound = models.ImageField(
#         upload_to='images',
#         height_field='height',
#         width_field='width',
#     )