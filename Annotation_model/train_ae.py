import os
import time
import math
import cv2
import logging
import glob
import numpy as np
from datetime import datetime
from dotenv import load_dotenv


from albumentations import *
from albumentations.pytorch import ToTensor

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from unet import UNet, get_torch_device


load_dotenv()
logger = logging.getLogger(__name__)


def asMinutes(s):
    """Узнаем в минутах выполнение"""
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    """Узнаем в время выполнения и проценты"""
    now = time.time()
    s = now - since
    es = s / (percent + .00001)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


class LayerActivations():
    """Слой активации модели"""
    features = None

    def __init__(self, layer):
        self.hook = layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def remove(self):
        self.hook.remove()


class Dataset(object):
    """Датасет с изображениями"""
    def __init__(self, fnames, transform=None, maximgs=-1):
        self.fnames = fnames
        self.transform = transform
        self.maximgs = min(maximgs, len(self.fnames))

    def __getitem__(self, index):
        index = index if self.maximgs == -1 else np.random.randint(0, self.maximgs)

        fname = self.fnames[index]
        image = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
        patch = image
        if self.transform is not None:
            patch1 = self.transform(image=image)['image']
            patch2 = self.transform(image=image)['image']

        return patch1, patch2, image

    def __len__(self):
        return len(self.fnames) if self.maximgs == -1 else self.maximgs


def run_train(**kwargs):
    """Запускаем обучение модели на датасете"""
    logger.warning('Начинаем обучение базовой модели')

    logger.warning(f'Создаем директорию для модели {kwargs["model_root"]}')
    os.makedirs(kwargs['model_root'], exist_ok=True)

    input_pattern = kwargs['patches_files']
    patch_size = kwargs['patchsize']
    num_imgs = kwargs['num_images']
    batch_size = kwargs['batchsize']
    num_epochs = kwargs['num_epochs']
    num_epochs_earlystop = kwargs['num_epochs_earlystop'] if kwargs['num_epochs_earlystop'] > 0 else float("inf")
    num_min_epochs = kwargs['num_min_epochs']

    logger.warning('Проверяем ос выполняющего устройства')
    if os.name =="nt":
        numworkers = 0
    else:
        numworkers = kwargs['num_workers'] if kwargs['num_workers']!=-1 else os.cpu_count()

    # параметры модели
    n_classes = 3
    in_channels = 3
    padding = True
    depth = 5
    wf = 2
    up_mode = 'upsample'
    batch_norm = True

    logger.warning('Берем устройство torch')
    device = get_torch_device()
    logger.warning(f'Устройство --> {device}')

    logger.warning('Инициализуруем модель:')
    model = UNet(n_classes=n_classes, in_channels=in_channels, padding=padding,
                 depth=depth, wf=wf, up_mode=up_mode, batch_norm=batch_norm, concat=True).to(device)
    logger.warning(f'Все параметры: \t{sum([np.prod(p.size()) for p in model.parameters()])}')

    dr = LayerActivations(model.down_path[-1].block[5])

    img_transform = Compose([
        RandomScale(scale_limit=0.1, p=.9),
        PadIfNeeded(min_height=patch_size, min_width=patch_size),
        VerticalFlip(p=.5),
        HorizontalFlip(p=.5),
        Blur(p=.5),
        # Downscale(p=.25, scale_min=0.64, scale_max=0.99),
        GaussNoise(p=.5, var_limit=(10.0, 50.0)),
        GridDistortion(p=.5, num_steps=5, distort_limit=(-0.3, 0.3),
                       border_mode=cv2.BORDER_REFLECT),
        ISONoise(p=.5, intensity=(0.1, 0.5), color_shift=(0.01, 0.05)),
        RandomBrightness(p=.5, limit=(-0.2, 0.2)),
        RandomContrast(p=.5, limit=(-0.2, 0.2)),
        RandomGamma(p=.5, gamma_limit=(80, 120), eps=1e-07),
        MultiplicativeNoise(p=.5, multiplier=(0.9, 1.1), per_channel=True, elementwise=True),
        HueSaturationValue(hue_shift_limit=20, sat_shift_limit=10, val_shift_limit=10, p=.9),
        Rotate(p=1, border_mode=cv2.BORDER_REFLECT),
        RandomCrop(patch_size, patch_size),
        ToTensor()
    ])

    logger.warning(f'Получаем тренировочные данные из наших путей {input_pattern}:')
    data_train = Dataset(glob.glob(input_pattern), transform=img_transform, maximgs=num_imgs)  # img_transform)

    print(f'Создаем загрузчик для тренировочных данных')
    data_train_loader = DataLoader(data_train, batch_size=batch_size,
                                   shuffle=True, num_workers=numworkers, pin_memory=True)

    # Метрика и оптимизатор
    optim = torch.optim.Adam(model.parameters(), lr=.1)
    best_loss = np.infty

    color_trans = Compose([
        HueSaturationValue(hue_shift_limit=50, sat_shift_limit=0, val_shift_limit=0, p=1),
        ToTensor()
    ])

    # +
    start_time = time.time()
    writer = SummaryWriter(log_dir=f"{kwargs['model_root']}/{datetime.now().strftime('%b%d_%H-%M-%S')}")
    criterion = nn.MSELoss()
    best_loss = np.infty
    best_epoch = -1

    for epoch in range(num_epochs):
        if (epoch > num_min_epochs and epoch - best_epoch > num_epochs_earlystop):
            logger.warning(f'USER: Обучение модели DL останавливается из-за отсутствия прогресса.'
                           f' Эпоха: {epoch} Последнее улучшение: {best_epoch}')
            break
        all_loss = torch.zeros(0).to(device)
        for X1, X2, X_orig in data_train_loader:
            X = torch.cat((X1, X2), 0)
            X = X.to(device)

            halfX = int(X.shape[0] / 2)

            prediction = model(X)  # [N, 2, H, W]
            Xfeatures = dr.features

            loss1 = criterion(prediction, X)
            loss2 = criterion(Xfeatures[0:halfX, ::], Xfeatures[halfX:, ::])

            loss = loss1 + loss2

            optim.zero_grad()
            loss.backward()
            optim.step()

            all_loss = torch.cat((all_loss, loss.detach().view(1, -1)))

        writer.add_scalar(f'train/loss', loss, epoch)

        print(f'PROGRESS: {epoch + 1}/{num_epochs} | {timeSince(start_time, (epoch + 1) / num_epochs)} | {loss.data}',
              flush=True)

        logger.warning('%s ([%d/%d] %d%%), полная потеря: %.4f \t loss1: %.4f \t loss2: %.4f ' % (
        timeSince(start_time, (epoch + 1) / num_epochs),
        epoch + 1, num_epochs, (epoch + 1) / num_epochs * 100, loss.data, loss1.data, loss2.data))

        all_loss = all_loss.cpu().numpy().mean()
        if all_loss < best_loss:
            best_loss = all_loss
            best_epoch = epoch
            logger.warning("  **")

            state = {'epoch': epoch + 1,
                     'model_dict': model.state_dict(),
                     'optim_dict': optim.state_dict(),
                     'best_loss_on_test': all_loss,
                     'n_classes': n_classes,
                     'in_channels': in_channels,
                     'padding': padding,
                     'depth': depth,
                     'wf': wf,
                     'up_mode': up_mode, 'batch_norm': batch_norm}

            torch.save(state, f"{kwargs['model_root']}/best_model.pth")
        else:
            logger.warning('')

    logger.warning(f'USER: Обучение базовой модели завершено!')

