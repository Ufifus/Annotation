import logging
import os
import glob
import cv2
import numpy as np
import sklearn.feature_extraction.image
from skimage.measure import label

from .models import ImageModel, RoisModel, ProjectModel
from Histology_Project.settings import MEDIA_ROOT
from Annotation_model import run_base_train, make_superpixel, make_prediction, make_superpixel_dl, retrain_model, make_embed


logger = logging.getLogger(__name__)


def get_images(proj):
    """Получаем все изображения в проекте"""
    images = ImageModel.objects.filter(projId=proj)
    for i, image in enumerate(images):
        images[i].ROIs = RoisModel.objects.filter(imageId=image).count()
        images[i].trainingROIs = RoisModel.objects.filter(imageId=image, testingROI=1).count()
    return images


def check_exist_image(image_name, proj_name):
    """Проверяем нет ли файла с таким именем в каталоге"""
    name, format = image_name.split('.')
    format = 'png'
    file = os.path.join(MEDIA_ROOT, 'images/', proj_name, '{0}.{1}'.format(name, format))
    logger.warning(file)
    return os.path.isfile(file)


def get_image_info(image_id):
    """Получаем информацию о изображении"""
    q_image = ImageModel.objects.filter(imageId=image_id)
    image_url = q_image[0].image.url
    logger.error(image_url)
    image = q_image.values()[0]
    image['date'] = None
    image['url'] = image_url
    logger.warning(image)
    return image


def get_mask_path(proj_name, image_id):
    """Возвращаем путь до маски изображения"""
    logger.warning('Запускаем подзагурзку маски №{0} из {1}'.format(image_id, proj_name))
    mask_name = ImageModel.objects.get(imageId=image_id).image.name.split('/')[-1]
    mask_name = mask_name.replace('.png', '_mask.png')
    mask_path = os.path.join(MEDIA_ROOT, 'images/{0}/mask/{1}').format(proj_name, mask_name).replace('\\', '/')
    logger.warning('Загружаем маску из пути ' + mask_path)
    return mask_path


def get_rois_for_image(image_id):
    """Возвразаем список Rois изображения"""
    image = ImageModel.objects.get(imageId=image_id)
    rois = RoisModel.objects.filter(imageId=image).values()
    return list(rois)


def get_latest_modelId(proj_name):
    proj = ProjectModel.objects.get(name=proj_name)
    modelId = proj.iteration
    return modelId


def check_need_patches(proj_name, image):
    needs_calculated = False
    patches_root = get_patches_root(proj_name)
    logger.warning('Ищем батчи для изображения №' + image.image.name)
    if not image.make_batches:
        needs_calculated = True
    else:
        logger.warning('Изображение было батчено. Ищем батчи')
        image_name = image.image.name.split('/')[-1].split('.')[0]
        patches_files = os.path.join(patches_root, '{0}*.png').format(image_name)
        number_of_patches = len(glob.glob(patches_files))
        logger.warning('Кол-во найденных батчей: ' + str(number_of_patches))
        if number_of_patches == 0:
            logger.warning('Батчей не найдено')
            needs_calculated = True
    return needs_calculated


def get_patches_root(proj_name):
    return os.path.join(MEDIA_ROOT, 'images/{0}/patches').format(proj_name).replace('\\', '/')


def make_patches_worker(**kwargs):
    """Проходим по всем созраненным изображениям в файле и создаем батчи каждого а потом изменяем в модели были ли батчи"""
    logger.warning('Начинаем создавать батчи...')
    path_to_list_images = kwargs['patches_file']
    count_added = 0
    with open(path_to_list_images, 'r') as file:
        for i, line in enumerate(file):
            logger.warning(f'Выполняем {i} изображение')
            image_id, path_to_image = line.rstrip().split(' ')
            try:
                kwargs['patches_file'] = path_to_image.replace('\\', '/')
                logger.warning(kwargs)
                separate_image(domask=False, **kwargs)
                logger.warning(f'Успешно обработали {i} изображение')
                image = ImageModel.objects.get(imageId=image_id)
                image.make_batches = True
                image.save()
                count_added += 1
            except Exception as e:
                logger.warning(f'Произошла ошибка при выполнении {i} изображения \n {e}')

    logger.warning(f'Разбились на батчи {count_added} изображений из {i+1}')

    if int(count_added) == int(i+1):
        project = image.projId
        project.make_patches = True
        project.save()



def separate_image(patches_file, patches_root, patch_size, bgremoved, domask=False):
    """Разбиваем изображение на батчи"""
    image_name = patches_file.split('/')[-1]
    mask_name = image_name.replace('.png', '_mask.png')
    image_name = image_name.split('.')[0]
    mask_path = os.path.join(patches_root.replace('patches', 'mask'), mask_name).replace('\\', '/')
    if os.path.isfile(mask_path) and domask:
        logger.warning('Путь до маски' + mask_path)
        ismask = True
    else:
        ismask = False

    logger.warning(f'Открываем изображение {patches_file}...')
    image = cv2.imread(patches_file)  # NOTE: this image is in BGR not RGB format
    if ismask:
        image = np.dstack([(image[:, :, 0] == 0) * 255,
                           (image[:, :, 0] > 0) * 255,
                           np.ones(image.shape[0:2]) * 255])

    idxs = np.asarray(range(image.shape[0] * image.shape[1])).reshape(image.shape[0:2])

    patch_out = sklearn.feature_extraction.image._extract_patches(image, (patch_size, patch_size, 3), patch_size)
    patch_out = patch_out.reshape(-1, patch_size, patch_size, 3)

    idx_out = sklearn.feature_extraction.image._extract_patches(idxs, (patch_size, patch_size), patch_size)
    idx_out = idx_out[:, :, 0, 0]
    idx_out = idx_out.reshape(-1)
    rs, cs = np.unravel_index(idx_out, idxs.shape)

    for r, c, patch in zip(rs, cs, patch_out):

        gpatch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        if (bgremoved and gpatch.mean() > 240):
            continue
        patch_name = f"{'mask/' if ismask else ''}{image_name}_{c}_{r}{'_mask' if ismask else ''}.png"
        logger.warning(f'Записываем файл батч в ' + patches_root + ' c названием ' + patch_name)
        patch_path = os.path.join(patches_root, f'{patch_name}').replace('\\', '/')
        cv2.imwrite(patch_path, patch)
    return


def training_base_model_worker(proj_name, **kwargs):
    """Запускаем обучение модели"""
    project = ProjectModel.objects.get(name=proj_name)
    try:
        run_base_train(**kwargs)
        project.train_ae = True
        project.iteration = 0
    except Exception as e:
        logger.exception(f'Произошла ошибка при тренировке базовой модели {proj_name}')
        logger.exception(e)
    finally:
        project.save()


def make_superpixels_worker(image_id, modelreq, **kwargs):
    """Запускаем создание суперпикселя"""
    image = ImageModel.objects.get(imageId=image_id)
    if modelreq < 0:
        logger.warning('У проекта еще нет модели так что создаем обычный суперпиксель')
        try:
            make_superpixel(**kwargs)
            image.superpixel_created = True
            image.superpixel_modelId = -1
        except Exception as e:
            logger.exception(f'Произошла ошибка при создании суперпикселя {kwargs["image_name"]}')
            logger.exception(e)
    else:
        logger.warning('У проекта есть модель аннотатора так что создаем суперпиксель на его основе')
        try:
            make_superpixel_dl(**kwargs)
            image.superpixel_created = True
            image.superpixel_modelId = kwargs['model']
        except Exception as e:
            logger.exception(f'Произошла ошибка при создании суперпикселя {kwargs["image_name"]}')
            logger.exception(e)
    image.save()


def make_prediction_worker(**kwargs):
    logger.warning('Начинаем создание предсказания...')
    try:
        make_prediction(**kwargs)
    except Exception as e:
        logger.warning(f'Произошла ошибка при создании предсказания!!!')
        logger.warning(e)


def get_number_of_objects(img):
    _, nobjects = label(img[:, :, 1], return_num=True)
    return nobjects


def populate_training_files(proj_name, train_file_path, test_file_path):
    """Наполняем файлы с тренировочными и тестовыми данными информацией о наших изображениях"""

    testfp = open(test_file_path, "w")
    trainfp = open(train_file_path, "w")

    # loop through the images in the database:
    project = ProjectModel.objects.get(name=proj_name)
    for img in ImageModel.objects.filter(projId=project):  # TODO can improve this
        logger.warning(f'Проверяем Rois у изображения: {img.name}')
        for roi in RoisModel.objects.filter(imageId=img):
            logger.warning(f'Roi path = {roi.roi_path}')
            # check if this image roi exists:
            if not os.path.isfile(roi.path):
                logger.warning(f'Не найденно Roi по пути {roi.path}')
                continue

            # append this roi to the appropriate txt file:
            logger.warning(f'Testing ROI = {str(roi.testingROI)}')
            if roi.testingROI:
                testfp.write(f"{roi.path}\n")
            elif roi.testingROI == 0:
                trainfp.write(f"{roi.path}\n")

    # close the files:
    testfp.close()
    trainfp.close()


def retrain_model_worker(proj_name, **kwargs):
    """Дообучиваем модель и обновляем проект в бд """
    project = ProjectModel.objects.get(name=proj_name)
    newmodelid = retrain_model(**kwargs)

    project.iteration = newmodelid
    project.save()
    logger.warning(f'Теперь у проекта {proj_name} новая модель аннотаций под номером {newmodelid}')


def make_embed_worker(proj_name, **kwargs):
    """Создаем эмбединги на основе модели"""
    logger.warning('Начинаем создание эмбедингов')

    newembed_model = make_embed(proj_name, **kwargs)
    project = ProjectModel.objects.get(name=proj_name)
    project.embed_iteration = newembed_model
    project.save()
    logger.warning(f'Теперь у проекта {proj_name} есть эмбединги на основе модели {newembed_model}')


