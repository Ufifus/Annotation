import os
import logging
import cv2
import numpy as np
from PIL import Image

from scipy import ndimage
from skimage.color import rgb2gray
from skimage.filters import rank
from skimage.morphology import disk
from skimage.restoration import denoise_tv_chambolle
from skimage.segmentation import watershed, find_boundaries

import sklearn.feature_extraction.image
import torch
from skimage.segmentation import find_boundaries
from skimage.segmentation import slic
from unet import UNet, get_torch_device


logger = logging.getLogger(__name__)


def make_superpixel(**kwargs):
    """Создаем обычный суперпиксель"""
    base_path = kwargs['base_path']
    image_name = kwargs['image_name']
    image_path = kwargs['image_path']

    superpix_root = os.path.join(base_path, 'superpixels').replace('\\', '/')
    superpix_path = os.path.join(superpix_root, image_name.replace('.png', '_superpixel.png')).replace('\\', '/')
    superpixbound_root = os.path.join(base_path, 'superpixels_boundary').replace('\\', '/')
    superpixbound_path = os.path.join(superpixbound_root, image_name.replace('.png', '_superpixel_boundary.png')).replace('\\', '/')

    logger.warning(f'Используем изображение {image_path}')
    logger.warning(f'Сохраняем суперпиксель как {superpix_path}')
    logger.warning(f'Сохраняем суперпиксель чб как {superpixbound_path}')

    if (not kwargs['force']) and os.path.exists(superpix_path):
        logger.warning('Пропускаем т.к Суперпиксель уже существует')
    else:

        io = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        # denoise
        denoised = denoise_tv_chambolle(io, weight=0.15, multichannel=True)
        denoised = rgb2gray(denoised)

        # find continuous region (low gradient) --> markers
        markers = rank.gradient(denoised, disk(1)) < 5
        markers = ndimage.label(markers)[0]

        # local gradient
        gradient = rank.gradient(denoised, disk(1))

        # watershed
        segments = watershed(gradient, markers, connectivity=1, compactness=0.0001, watershed_line=False)
        colors = np.array(  # TODO: replace with colormap approach
            [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for i in
             range(segments.max() + 1)])

        cv2.imwrite(superpix_path, colors[segments])

        boundary = find_boundaries(segments, connectivity=1, mode='outer', background=0)
        boundary = boundary.astype(np.uint8) * 255

        cv2.imwrite(superpixbound_path, boundary)


class LayerActivations():
    features = None

    def __init__(self, layer):
        self.hook = layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def remove(self):
        self.hook.remove()


# -----helper function to split data into batches
def divide_batch(l, n):
    for i in range(0, l.shape[0], n):
        yield l[i:i + n, ::]


def make_superpixel_dl(**kwargs):
    """Создаем суперпиксель на основе модели"""

    logger.warning(f'Создаем суперпиксели изображения {kwargs["image_name"]} на основе модели {kwargs["model"]}')

    batch_size = kwargs['batchsize']
    patch_size = kwargs['patchsize']
    stride_size = patch_size // 2

    base_path = kwargs['base_path']
    image_name = kwargs['image_name']
    image_path = kwargs['image_path']

    superpix_root = os.path.join(base_path, 'superpixels').replace('\\', '/')
    superpix_path = os.path.join(superpix_root, image_name.replace('.png', '_superpixel.png')).replace('\\', '/')
    superpixbound_root = os.path.join(base_path, 'superpixels_boundary').replace('\\', '/')
    superpixbound_path = os.path.join(superpixbound_root, image_name.replace('.png', '_superpixel_boundary.png')).replace('\\', '/')

    logger.warning(f'Используем изображение {image_path}')
    logger.warning(f'Сохраняем суперпиксель как {superpix_path}')
    logger.warning(f'Сохраняем суперпиксель чб как {superpixbound_path}')

    device = get_torch_device()

    checkpoint = torch.load(kwargs['model'], map_location=lambda storage,
                            loc: storage)  # load checkpoint to CPU and then put to device https://discuss.pytorch.org/t/saving-and-loading-torch-models-on-2-machines-with-different-number-of-gpu-devices/6666
    model = UNet(n_classes=checkpoint["n_classes"], in_channels=checkpoint["in_channels"],
                 padding=checkpoint["padding"], depth=checkpoint["depth"], wf=checkpoint["wf"],
                 up_mode=checkpoint["up_mode"], batch_norm=checkpoint["batch_norm"]).to(device)
    model.load_state_dict(checkpoint["model_dict"])
    model.eval()

    dr = LayerActivations(model.up_path[-1].conv_block.block[-1])

    logger.warning(f"Всего параметров: \t{sum([np.prod(p.size()) for p in model.parameters()])}")

    # ----- get file list

    if not kwargs['force'] and os.path.exists(superpix_path):
        logger.warning('Пропускаем т.к Суперпиксель уже существует')
    else:
        io = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        io_shape_orig = np.array(io.shape)

        # add half the stride as padding around the image, so that we can crop it away later
        io = np.pad(io, [(stride_size // 2, stride_size // 2), (stride_size // 2, stride_size // 2), (0, 0)],
                    mode="reflect")

        io_shape_wpad = np.array(io.shape)

        # pad to match an exact multiple of unet patch size, otherwise last row/column are lost
        npad0 = int(np.ceil(io_shape_wpad[0] / patch_size) * patch_size - io_shape_wpad[0])
        npad1 = int(np.ceil(io_shape_wpad[1] / patch_size) * patch_size - io_shape_wpad[1])

        io = np.pad(io, [(0, npad0), (0, npad1), (0, 0)], mode="constant")

        arr_out = sklearn.feature_extraction.image._extract_patches(io, (patch_size, patch_size, 3), stride_size)
        arr_out_shape = arr_out.shape
        arr_out = arr_out.reshape(-1, patch_size, patch_size, 3)

        # in case we have a large network, lets cut the list of tiles into batches
        output = np.zeros((0, 4, patch_size, patch_size))
        for batch_arr in divide_batch(arr_out, batch_size):
            print(f"PROGRESS: Superpixel Chunk {output.shape[0]}/{arr_out.shape[0]}")

            arr_out_gpu = torch.from_numpy(batch_arr.transpose(0, 3, 1, 2) / 255).type('torch.FloatTensor').to(device)

            # ---- get results
            output_batch = model(arr_out_gpu)

            # --- pull from GPU and append to rest of output
            output_batch = dr.features.detach().cpu().numpy().astype(np.double)

            output = np.append(output, output_batch, axis=0)

        output = output.transpose((0, 2, 3, 1))

        # turn from a single list into a matrix of tiles
        output = output.reshape(arr_out_shape[0], arr_out_shape[1], patch_size, patch_size, output.shape[3])

        # remove the padding from each tile, we only keep the center
        output = output[:, :, stride_size // 2:-stride_size // 2, stride_size // 2:-stride_size // 2, :]

        # turn all the tiles into an image
        output = np.concatenate(np.concatenate(output, 1), 1)

        # incase there was extra padding to get a multiple of patch size, remove that as well
        output = output[0:io_shape_orig[0], 0:io_shape_orig[1], :]  # remove paddind, crop back

        # --- super pixel work
        number_segments = (output.shape[0] // kwargs['approxcellsize']) ** 2
        logger.warning(f"Используем {number_segments} суперпикселя")
        segs_dl = slic(output, n_segments=number_segments, compactness=kwargs['compactness'], multichannel=True,
                       slic_zero=True)  # <--- slic_zero?

        colors = np.array(
            # make random colors. its okay if some are the same, just as long as they're not touching which is unlikely
            [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for i in
             range(segs_dl.max() + 1)])

        cv2.imwrite(superpix_path, colors[segs_dl])

        boundary = find_boundaries(segs_dl, connectivity=1, mode='outer', background=0)
        boundary = boundary.astype(np.uint8) * 255

        cv2.imwrite(superpixbound_path, boundary)