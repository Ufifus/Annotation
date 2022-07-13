import glob

from django.shortcuts import render, redirect, reverse
from django.http import JsonResponse, HttpResponse, FileResponse
from django.views.generic import TemplateView

from PIL import Image
import logging
import os
import re
import cv2
import numpy as np
import base64
import json
from datetime import datetime
from dotenv import load_dotenv

from .models import ProjectModel, ImageModel, RoisModel
from .services import get_images, check_exist_image, get_image_info, get_mask_path, get_rois_for_image, \
    get_latest_modelId, check_need_patches, get_patches_root, make_patches_worker, MEDIA_ROOT, \
    training_base_model_worker, make_superpixels_worker, make_prediction_worker, get_number_of_objects, \
    populate_training_files, retrain_model_worker, make_embed_worker


load_dotenv()
logger = logging.getLogger(__name__)


class ListProjects(TemplateView):
    template_name = 'image_app/main.html'

    def get(self, request, *args, **kwargs):
        """Выводим все имеющиеся поекты"""
        projects = ProjectModel.objects.all()
        context = {
            'projects': projects,
        }
        return render(request, self.template_name, context=context)


def add_project(request):
    """Добавление проекта в бд"""
    if request.POST:
        data = request.POST
        logger.warning(data)
        if data['name'] is None or '':
            pass
        else:
            project = ProjectModel(name=data['name'], description=data['description'], date=datetime.now())
            project.save()
        return redirect('image_app:list-projects')



def delete_project(request, proj_name):
    """Удаление проекта из бд вместе со всеми файлами"""
    if request.method == 'GET':
        proj = ProjectModel.objects.get(name=proj_name)
        logger.warning(f'delete project {proj.projId}')
        proj.delete()
        return redirect('image_app:list-projects')
    else:
        return HttpResponse('<h1>Вы не можете удалить данный проект</h1>')


class DetailProject(TemplateView):
    template_name = 'image_app/project.html'

    def get(self, request, proj_name, *args, **kwargs):
        """Выводит список изображений и доступные ф-ю с ними"""
        proj = ProjectModel.objects.get(name=proj_name)
        images = get_images(proj)
        context = {
                    'images': images,
                    'project': proj,
                   }
        return render(request, self.template_name, context)

    def post(self, request, proj_name):
        """Загрузка новых изображений в проект"""
        if request.POST:
            proj = ProjectModel.objects.get(name=proj_name)
            for new_image in request.FILES.getlist('file'):
                logger.warning(new_image)
                if check_exist_image(new_image.name, proj_name):
                    logger.warning('file is exist')
                else:
                    image = ImageModel(image=new_image, projId=proj)
                    image.save()
        images = get_images(proj)
        context = {
            'project': proj,
            'images': images
        }
        return render(request, self.template_name, context=context)


def get_db_data(request):
    """Получаем на страницу данные о изображении или проекте"""
    ag = request.GET.get('q')
    ag = json.loads(ag)
    logger.warning(ag)
    filters = ag['filters'][0]
    if filters['name'] == 'project':
        """Выводим кол-во изображений"""
        project = ProjectModel.objects.get(projId=filters['val'])
        num_images = ImageModel.objects.filter(projId=project).count()
        data = {
            'num_results': num_images
        }
        return JsonResponse(data, status=200)
    elif filters['name'] == 'projId':
        project = ProjectModel.objects.filter(projId=filters['val']).values()[0]
        return JsonResponse(project, status=200)
    if filters['name'] == 'id':
        images = ImageModel.objects.filter(imageId=filters['val']).values()[0]
        num_images = len(images)
        data = {
            'num_results': num_images,
            'objects': images
        }
        return JsonResponse(data, status=200)


def make_patches(request, proj_name):
    """Ф-я для создания патчей изображения"""
    logger.warning('Создаем батчи для проекта ' + proj_name)
    project = ProjectModel.objects.get(name=proj_name)
    if project is None:
        return JsonResponse('Project doesnt exist', safe=False, status=400)

    target_files = []
    logger.warning('Пробегаем по списку изображений прикрепленных к проекту...')
    for image in ImageModel.objects.filter(projId=project):
        if check_need_patches(proj_name, image):
            logger.warning('Нужно создать батчи для изображения ' + image.image.name)
            target_files.append((image.imageId, image.image.path))

    if not target_files:
        logger.warning('Нет изображений для создания  батчей')
        return JsonResponse('None images for patching', safe=False, status=400)

    logger.warning('Сохраняем название изображений для батчинга: ' + str(len(target_files)) + ' - кол-во')
    patches_root = get_patches_root(proj_name)
    logger.warning(f'Путь до папки с батчами {patches_root}')
    patches_file = os.path.join(patches_root, 'imgs.txt')
    with open(patches_file, 'w') as file:
        for fid, fpath in target_files:
            file.write(f"{fid} {fpath}\n")

    patchsize = int(os.getenv('patchsize', 256))
    logger.warning('Размер батча ' + str(patchsize))

    whiteBG = str(request.GET.get('whiteBG', default='keep'))
    if whiteBG == 'remove':
        bgremoved = True
    else:
        bgremoved = False

    need_data = {
        'patches_root': patches_root,
        'patches_file': patches_file,
        'bgremoved': bgremoved,
        'patch_size': patchsize,
    }
    make_patches_worker(**need_data)
    return JsonResponse(data='Patches have made', status=200)


def train_autoencoder(request, proj_name):
    """Ф-я для запуска обучения модели аннотаций"""
    try:
        project = ProjectModel.objects.get(name=proj_name)
    except ProjectModel.DoesNotExist:
        return JsonResponse(f'Данного проекта под названием {proj_name} не существует', status=400)

    logger.warning(f'Начинаем тренировать модель автокодирования для проекта {proj_name}')
    logger.warning(f'Достаем данные из конфигов')
    train_config_ae = {
        'num_images': int(os.getenv('trainae_num_images', default=-1)),
        'batchsize': int(os.getenv('trainae_batchsize', default=32)),
        'patchsize': int(os.getenv('trainae_patchsize', default=32)),
        'num_workers': int(os.getenv('trainae_num_workers', default=0)),
        'num_epochs': int(os.getenv('trainae_num_epochs', default=1000)),
        'num_epochs_earlystop': int(os.getenv('trainae_num_epochs_earlystop', default=100)),
        'num_min_epochs': int(os.getenv('trainae_num_min_epochs', default=300))
    }

    proj_root = os.path.join(MEDIA_ROOT, 'images/{0}').format(proj_name).replace('\\', '/')
    train_config_ae['model_root'] = os.path.join(proj_root, 'models/0')
    train_config_ae['patches_files'] = os.path.join(proj_root, 'patches/*.png')

    for k, v in train_config_ae.items():
        logger.warning(f'{k} == {v}')

    training_base_model_worker(proj_name=proj_name, **train_config_ae)
    return JsonResponse(data='Model have trained', status=200, safe=False)


def retrain_model(request, proj_name):
    """Дооубучиваем модель на созданных нами аннотациях"""

    project = ProjectModel.objects.get(name=proj_name)

    logger.warning(f'Начинаем дообучивать модель проекта {proj_name}')

    frommodelid = int(request.GET.get('frommodelid', default=0))
    if frommodelid == -1:
        frommodelid = get_latest_modelId(proj_name)

    logger.warning(f'Начинаем дообучивать модель {frommodelid}')

    model_path = os.path.join(MEDIA_ROOT, f'images/{proj_name}/models/{frommodelid}/best_model.pth').replace('\\', '/')
    if frommodelid > project.iteration or not os.path.exists(model_path):
        return JsonResponse(data='Request model doesnt exists', status=404, safe=False)

    if project.train_ae_time is None and frommodelid == 0:
        error_message = f'The base model 0 of project {proj_name} was overwritten when Retrain Model 0 started.\n ' \
                        f'Please wait until the Retrain Model 0 finishes. '
        logger.warning(f'Базовая модель 0 из проекта {proj_name} еще недообучилась так что подождите')
        return JsonResponse(data=error_message, status=400, safe=False)

    new_modelid = get_latest_modelId(proj_name) + 1
    new_model_path = os.path.join(MEDIA_ROOT, f'images/{proj_name}/models/{new_modelid}').replace('\\', '/')
    if not os.path.exists(new_model_path):
        os.mkdir(new_model_path)

    logger.warning(f'Новый путь до модели: {new_model_path}')

    train_file_path = os.path.join(MEDIA_ROOT, f'images/{proj_name}/train_imgs.txt').replace('\\', '/')
    test_file_path = os.path.join(MEDIA_ROOT, f'images/{proj_name}/test_imgs.txt').replace('\\', '/')

    populate_training_files(proj_name, train_file_path, test_file_path)

    empty_training = not os.path.exists(test_file_path) or os.stat(test_file_path).st_size == 0
    empty_testing = not os.path.exists(test_file_path) or os.stat(test_file_path).st_size == 0
    if empty_training or empty_testing:  # TODO can improve this by simply counting ROIs in the db
        error_message = f'Not enough training/test images for project {proj_name}. You need at least 1 of each.'
        logger.warning(f'Не достаточно данных для дообучения модели проекта {proj_name}')
        return JsonResponse(data=error_message, status=400, safe=False)

    config_retrain = {
        'num_epochs': os.getenv('retrainae_num_epochs', default=1000),
        'num_epochs_earlystop': os.getenv('retrainae_num_epochs_earlystop', default=-1),
        'num_min_epochs': os.getenv('retrainae_num_min_epochs', default=300),
        'batch_size': os.getenv('retrainae_batchsize', default=32),
        'patch_size': os.getenv('retrainae_patchsize', default=256),
        'num_workers': os.getenv('retrainae_numworkers', default=0),
        'edge_weight': os.getenv('retrainae_edgeweight', default=2),
        'pclass_weight': os.getenv('retrainae_pclass_weight', default=.5),
        'fillbatch': os.getenv('retrainae_fillbatch', default=False),
        'train_file': train_file_path,
        'test_file': test_file_path,
        'model': model_path,
        'new_model_path': new_model_path,
        'project_name': proj_name
    }

    if config_retrain['pclass_weight'] == -1:
        proj_ppixel = 0
        proj_npixel = 0
        for image in ImageModel.objects.filter(projId=project):
            proj_npixel += image.npixels
            proj_ppixel += image.ppixels
        total = proj_npixel + proj_ppixel
        if total:
            pclass_weight = 1 - proj_ppixel / total
        else:
            pclass_weight = 0

    config_retrain['pclass_weigh'] = pclass_weight
    if bool(os.getenv('retrainae_fillbatch')):
        config_retrain['fillbatch'] = bool(os.getenv('retrainae_fillbatch'))

    retrain_model_worker(proj_name, **config_retrain)

    return JsonResponse(data='Success retrain', status=200, safe=False)


def make_embed(request, proj_name):
    """Создаем ембединги на основе изображений"""
    try:
        project = ProjectModel.objects.get(name=proj_name)
    except:
        return JsonResponse('This projects is unable', safe=False, status=500)

    # Проверяем на наличие существующей модели
    path_to_base_model = os.path.join(MEDIA_ROOT, f'images/{proj_name}/models/0/best_model.pth').replace('\\', '/')
    base_model_exist = os.path.exists(path_to_base_model)
    logger.warning(f'Базовая модель для проекта {proj_name} = {base_model_exist}')
    if not base_model_exist:
        return JsonResponse('Вы не можете создать ембединги так как не существует базовой модели',  safe=False, status=500)

    if project.train_ae is False and project.iteration == 0:
        return JsonResponse(f'Модель 0 обучается дождитесь окончания', safe=False, status=404)

    latest_modelId = get_latest_modelId(proj_name)
    modelid = int(request.GET.get('modelid', default=latest_modelId))

    make_embed_dict = {
        'batchsize': int(os.getenv('make_embed_batchsize', default=32)),
        'patchsize': int(os.getenv('make_embed_patchsize', default=256)),
        'num_imgs': int(request.GET.get('numimgs', default=-1)),
        'modelid': modelid,
        'outdir': os.path.join(MEDIA_ROOT, f'images/{proj_name}/models/{modelid}').replace('\\', '/'),
    }

    if modelid < 0 or modelid > latest_modelId:
        return JsonResponse(f'Ошибка, модели или не существует или вы выбрали не существующую модель', safe=False, status=404)

    make_embed_worker(proj_name, **make_embed_dict)

    return JsonResponse('Great success to make embed', status=200, safe=False)


class PlotEmbeding(TemplateView):
    """Рисуем эмбединги"""
    template_name = 'image_app/embed.html'

    def get(self, request, proj_name):
        proj = ProjectModel.objects.get(name=proj_name)
        project_iteration = proj.iteration
        selected_modelid = request.GET.get('modelid', default=get_latest_modelId(proj_name))
        if selected_modelid > get_latest_modelId(proj_name) or selected_modelid < 0:
            return JsonResponse('None avalable model', status=404, safe=False)

        context = {
            'project_iteration': project_iteration,
            'selected_modelid': selected_modelid,
            'project_name': proj_name
        }
        return render(request, self.template_name, context)


def get_embed(request, proj_name, image_id):
    """Отправляем эмбединги на сайт"""
    logger.warning(f'Image_Id = {image_id}')
    image_name = ImageModel.objects.get(imageId=image_id).image.name
    embed_path = os.path.join(MEDIA_ROOT, f'images/{proj_name}/patches').replace('\\', '/')
    files = glob.glob(f'{embed_path}/{image_name}*.png')
    if len(files) != 0:
        return FileResponse(open(files[0], 'rb'), status=200)


def get_embed_csv(request, proj_name):
    """Передаем на сайт csv файл с эмбедингами"""
    project = ProjectModel.objects.get(name=proj_name)

    latest_modelid = get_latest_modelId(proj_name)
    selected_modelid = int(request.GET.get('modelid', default=latest_modelid))
    logger.warning(f'Выбранная модель под номером {selected_modelid}')

    path_embed = os.path.join(MEDIA_ROOT, f'images/{proj_name}/models/{selected_modelid}/embedding.csv').replace('\\', '/')
    if selected_modelid > latest_modelid or selected_modelid < 0:
        return JsonResponse('You selected model is worth', status=403, safe=False)

    if not os.path.exists(path_embed):
        return JsonResponse('Embeddings do not exist', status=403, safe=False)

    return FileResponse(open(path_embed, 'rb'), status=200)


def get_roi(request, proj_name, roi_id):
    """Отправляем ройсы на сайт"""
    logger.warning(f'Roi_Id = {roi_id}')
    roi = RoisModel.objects.get(roiId=roi_id)
    if os.path.exists(roi.roi_path):
        return FileResponse(open(roi.roi_path, 'rb'), status=200)


def delete_image(request, image_id, proj_name):
    """Удаляем ненужные изображения и их Rois"""
    logger.warning('Удаляем ужображение' + image_id + 'из проекта' + proj_name)
    image = ImageModel.objects.get(imageId=image_id)
    image.delete()
    return JsonResponse(success=True, status=200)


class DetailImages(TemplateView):
    """Выводим страницу с аннотацией изображения"""
    template_name = 'image_app/image.html'

    def get(self, request, proj_name, image_id, *args, **kwargs):
        logger.warning(f'Изображение = {image_id}')
        project = ProjectModel.objects.filter(name=proj_name).values()[0]
        project['date'] = None
        logger.warning(project)
        image = get_image_info(image_id)
        startX = request.GET.get('startX', default='#')
        startY = request.GET.get('startY', default='#')
        defaultCropSize = request.GET.get('cropSize', default='256 ')
        context = {
                    'project': json.dumps(project),
                    'image': json.dumps(image),
                    'startX': json.dumps(startX),
                    'startY': json.dumps(startY),
                    'defaultCropSize': defaultCropSize,
                    'proj_name': proj_name,
                    'image_id': image_id
        }
        return render(request, self.template_name, context=context)


def get_prednext_image(request, proj_name, image_id, direction):
    """Выводим следующую страницу с изображеним"""
    image_id = int(image_id)
    project = ProjectModel.objects.get(name=proj_name)
    images_id = sorted([ID['imageId'] for ID in ImageModel.objects.filter(projId=project).values('imageId')])
    if len(images_id) == 1:
        return JsonResponse(data='Only one image upload for project', status=404, safe=False)

    logger.warning(f"{images_id}")
    current_image_id_index = images_id.index(image_id)

    if direction == 'previous':
        image_id_index = current_image_id_index - 1
        try:
            new_image_id = images_id[image_id_index]
        except IndexError:
            new_image_id = images_id[-1]
        logger.warning('Переходим на следущую страницу с изображением...')
    else:
        image_id_index = current_image_id_index + 1
        try:
            new_image_id = images_id[image_id_index]
        except IndexError:
            new_image_id = images_id[0]
        logger.warning('Переходим на прошлую страницу с изображением...')
    # return redirect('image_app:annotation', proj_name=proj_name, image_id=new_image_id)
    kwargs_url = {
        'proj_name': str(proj_name),
        'image_id': str(new_image_id),
    }
    logger.warning(f'Переходим на страницу с проектом {proj_name} и изображением {new_image_id}')
    logger.warning(f"URl = {reverse('image_app:annotation', kwargs=kwargs_url)}")
    return JsonResponse(data={'url': reverse('image_app:annotation', kwargs=kwargs_url)}, status=200)


def get_image(request, proj_name, image_id):
    logger.warning(f'Запускаем страницу с изображением c его аннотацией \n -------------- \n изображение №{image_id} '
                   f'проекта {proj_name}')
    image_path = ImageModel.objects.get(imageId=image_id).image.path
    logger.warning('Path изображения ' + image_path)
    image = open(image_path, 'rb')
    return FileResponse(image, status=200)


def get_mask(request, proj_name, image_id):
    """Загужаем маску из хранилища и отправляем на страницу"""
    logger.warning('MASK ----------')
    mask_path = get_mask_path(proj_name, image_id)
    mask = open(mask_path, 'rb')
    return FileResponse(mask, status=200)


def get_rois(request, proj_name, image_id):
    """загружаем Rois на страницу"""
    logger.warning(f'Выгружаем на страницу Rois {image_id} проекта {proj_name}')
    rois = get_rois_for_image(image_id)
    return JsonResponse(rois, safe=False, status=200)


def get_superpixels(request, proj_name, image_id):
    """Выгружаем на страницу superpixels изображения предварительно создав его если нет нового или модель обновилась
    Если модели нет то создаем стандартный superpixels"""
    logger.warning('SUPERPIXEL ---------')
    logger.warning(f'Получаем суперпиксель изображения {image_id} из проекта {proj_name}')

    force = bool(request.GET.get('force', False))
    logger.warning(f'Пересоздать суперпиксель --> {force}')

    latest_modelId = get_latest_modelId(proj_name)

    modelidreq = int(request.GET.get('superpixel_run_id', latest_modelId))
    logger.warning(f'Модель для создания суперпикселей проекта: {modelidreq}')

    if modelidreq > latest_modelId:
        return JsonResponse('Вызываемая модель лучше имеющихся', safe=False, status=400)

    project = ProjectModel.objects.get(name=proj_name)
    image = ImageModel.objects.get(imageId=image_id)
    image_name = image.image.name.split('/')[-1]
    superpixel_modelid = image.superpixel_modelId

    base_root = os.path.join(MEDIA_ROOT, f'images/{project.name}').replace('\\', '/')
    superpixel_root = os.path.join(base_root, 'superpixels/').replace('\\', '/')
    logger.warning(f'Путь до директории с суперпикселями: {superpixel_root}')

    superpixel_image_root = os.path.join(superpixel_root, image_name.replace('.png', '_superpixel.png')).replace('\\', '/')
    logger.warning(f'Путь до файла суперпикселя: {superpixel_image_root}')

    if os.path.exists(superpixel_image_root):
        logger.warning(f'У изображения {image_name} есть суперпиксель созданный моделью {superpixel_modelid}')
    else:
        logger.warning(f'У изображения {image_name} нет суперпикселя')

    config_superpixel = {
        'batchsize': int(os.getenv('superpixel_batchsize', 32)),
        'patchsize': int(os.getenv('superpixel_patchsize', 256)),
        'approxcellsize': int(os.getenv('superpixel_approxcellsize', 20)),
        'compactness': float(os.getenv('superpixel_compactness', .01)),
        'base_path': base_root,
        'image_name': image_name,
        'image_path': image.image.path,
        'model': os.path.join(base_root, '{modelidreq}/best_model.pth',).replace('\\', '/'),
        'force': force,
    }

    if modelidreq > superpixel_modelid or force:
        try:
            logger.warning('У изображения устаревший способ создания суперпикселя, так как появилась обновленная модель')
            os.remove(superpixel_image_root)
        except:
            pass

    make_superpixels_worker(image_id, modelidreq, **config_superpixel)

    superpixel = open(superpixel_image_root, 'rb')
    return FileResponse(superpixel, status=200)


def get_superpixels_boundary(request, proj_name, image_id):
    """Находим файл с суперпикселем-чб и отдаем на сервер"""

    logger.warning(f'Находим файл с суперпикселем-чб проекта {proj_name} и изображения {image_id}')
    image = ImageModel.objects.get(imageId=image_id)
    image_name = image.image.name.split('/')[-1].split('.')[0]
    superpixel_boundary_path = os.path.join(MEDIA_ROOT,
                    f'images/{proj_name}/superpixels_boundary/{image_name}_superpixel_boundary.png').replace('\\', '/')
    logger.warning(f'Путь до суперпикселя-чб {superpixel_boundary_path}')
    if os.path.exists(superpixel_boundary_path):
        logger.warning('Передаем суперпиксель-чб')
        superpixel_bound = open(superpixel_boundary_path, 'rb')
        return FileResponse(superpixel_bound, status=200)
    else:
        logger.warning('суперпикселя-чб нет')
        message = 'File boundary doesnt exist'
        return JsonResponse(message, safe=False, status=404)


def get_model(request, proj_name):
    """Передаем на страницу модель ae"""
    logger.warning(f'Загружаем на страницу модель')
    modelid = int(request.GET.get('model', get_latest_modelId(proj_name)))
    logger.warning(f'Номер модели = {modelid}')
    model_path = os.path.join(MEDIA_ROOT, f'images/{proj_name}/models/{modelid}/best_model.pth').replace('\\', '/')
    if os.path.exists(model_path):
        return FileResponse(model_path, status=200)
    else:
        message = 'Данной модели не существует'
        logger.warning(message)
        return JsonResponse(data=message, status=400, safe=False)


def get_prediction(request, proj_name, image_id):
    """Передаем на страницу предсказания модели"""
    logger.warning(f'Получаем предсказание для изображения {image_id} из проекта {proj_name}')
    image = ImageModel.objects.get(imageId=image_id)

    modelid = int(request.GET.get('model', get_latest_modelId(proj_name)))
    logger.warning(f'Модель для создания предсказания = {modelid}')

    if modelid < 0:
        logger.warning('У данного изображения(проекта) нет существующей модели так что предсказание недоступно')
        return JsonResponse(data='Model doesnt exist for this image', status=400, safe=False)

    pred_root = os.path.join(MEDIA_ROOT, f'images/{proj_name}/pred{modelid}')
    image_name = image.image.name.split('/')[-1]
    pred_file = os.path.join(pred_root, f'{image_name}'.replace('.png', '_pred.png')).replace('\\', '/')
    model_path = os.path.join(MEDIA_ROOT, f'images/{proj_name}/models/{modelid}/best_model.pth').replace('\\', '/')

    logger.warning(f'Полный путь до файла предсказаний: {pred_file}')

    config_pred = {
        'batchsize': int(os.getenv('prediction_batchsize', 256)),
        'patchsize': int(os.getenv('prediction_patchsize', 8)),
        'image_path': image.image.path,
        'pred_path': pred_file,
        'model': model_path,
        'resize': 1
    }

    make_prediction_worker(**config_pred)

    return FileResponse(pred_file, status=200)


def create_roi(request, proj_name, image_id):
    """Ф-я для создания Rois изображений"""
    """ post_roimask
        add_roi_to_traintest
        remove_image_from_traintest
        get_rois_for_image"""
    if request.method == 'POST':
        logger.warning(f'Начинаем создание ройса для изображения {image_id}')

        force = bool(request.POST.get('force', False))
        roimask_url = request.POST.get('roimask', None)
        image = ImageModel.objects.get(imageId=image_id)

        if not roimask_url:
            return JsonResponse(data='no mask provided', status=400, safe=False)

        roimask_data = re.search(r'data:image/png;base64,(.*)', roimask_url).group(1)
        roimask_decoded = base64.b64decode(roimask_data)
        roimask = cv2.imdecode(np.frombuffer(roimask_decoded, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        roimask = cv2.cvtColor(roimask, cv2.COLOR_BGR2RGB)

        if not np.all(np.isin(roimask, [0, 255])):
            return JsonResponse(data='Non [0,255] incorrect values are saved in the roimask mask, please check',
                                status=400, safe=False)

        if roimask.shape[2] > 3:
            return JsonResponse(data='Roi Mask has 4 dimensions? Possible Alpha Channel Issue?', status=400, safe=False)

        h = roimask.shape[0]
        w = roimask.shape[1]

        x = int(request.POST.get('pointx', -1))
        y = int(request.POST.get('pointy', -1))

        if -1 == x or -1 == y:
            return JsonResponse(data='no x , y location provided', status=402, safe=False)

        img = cv2.imread(image.image.path)
        if y + h > img.shape[0] or x + w > img.shape[1] or y < 0 or x < 0:
            return JsonResponse(f"ROI not within image, roi xy ({x} ,{y}) vs image size ({img.shape[0]}, {img.shape[1]})",
                                status=402, safe=False)

        image_name = image.image.name.split('/')[-1]
        mask_name = image_name.replace('.png', '_mask.png')
        mask_name = os.path.join(MEDIA_ROOT, f'images/{proj_name}/mask/{mask_name}').replace('\\', '/')
        if not os.path.isfile(mask_name):
            mask = np.zeros(img.shape, dtype=np.uint8)
        else:
            mask = cv2.cvtColor(cv2.imread(mask_name), cv2.COLOR_BGR2RGB)

        roimaskold = mask[y:y + h, x:x + w, :]

        if np.any(roimaskold != 0) and not force:
            logger.warning('ROI exists at this position.')
            return JsonResponse(data="ROI at this position already exists, enable force to overide", status=402, safe=False),

        mask[y:y + h, x:x + w, :] = roimask
        cv2.imwrite(mask_name, cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))

        roi_base_name = f'{image_name.replace(".png", "_")}{x}_{y}_roi.png'
        roi_name = os.path.join(MEDIA_ROOT, f'images/{proj_name}/roi/{roi_base_name}').replace('\\', '/')
        logger.warning(f'Сохраняем изображение как {roi_name}')

        roi = img[y:y + h, x:x + w, :]
        cv2.imwrite(roi_name, roi)

        # --- update positive / negative stats

        image.ppixel = np.count_nonzero(mask[:, :, 1] == 255)
        image.npixel = np.count_nonzero(mask[:, :, 0] == 255)

        # -- determine number of new objects from this roi, will need for statistics later
        nobjects_roi = get_number_of_objects(roimask)
        image.nobjects = get_number_of_objects(mask)

        # ----
        logger.warning('Сохраняем ройсы в бд:')

        newRoi = RoisModel(imageId=image, roi_name=roi_base_name, roi_path=roi_name, width=w, height=h,
                           x=x, y=y, nobjects=nobjects_roi, date=datetime.now())
        newRoi.save()
        newRoi_id = newRoi.roiId
        roi = RoisModel.objects.filter(roiId=newRoi_id).values()[0]
        image.save()

        return JsonResponse(data={'roi': roi, 'success': True}, status=201)
    else:
        return JsonResponse(data='No POST method', status=400, safe=False)


def add_roi_to_traintest(request, proj_name, roiid, traintype):
    logger.warning(request.GET)
    logger.warning(
        f'Добавляем новые аннотации к изображению. Project = {proj_name} Training type = {traintype} Name = {roiid}')

    roi = RoisModel.objects.get(roiId=roiid)
    if roi is None:
        return JsonResponse(data=f"{roiid} not found in project {proj_name}", status=404, safe=False)
    logger.warning('Найден ройс = ' + str(roi.roiId))

    if traintype == "train":
        roi.testingROI = 0
        logger.warning('Cохраняем тип ройса как тренировочный')
    if traintype == "test":
        roi.testingROI = 1
        logger.warning('Cохраняем тип ройса как тестовый')

    roi.save()
    roi = RoisModel.objects.filter(roiId=roiid).values()[0]
    logger.warning(f'Создали roi с параметрами: {roi}')
    return JsonResponse(data={'roi': roi}, status=200)

