import os
import numpy as np
import cv2
import torch
import sklearn.feature_extraction.image
import ttach as tta
import logging

from unet import UNet
from unet import UNet, get_torch_device


logger = logging.getLogger(__name__)


# -----helper function to split data into batches
def divide_batch(l, n):
    for i in range(0, l.shape[0], n):
        yield l[i:i + n, ::]


def make_prediction(**kwargs):
    """Создаем предсказания для изображения"""

    resize = kwargs['resize']
    batch_size = kwargs['batchsize']
    patch_size = kwargs['patchsize']
    stride_size = patch_size // 2

    # ----- load network
    device = get_torch_device()

    checkpoint = torch.load(kwargs['model'], map_location=lambda storage,
                                                            loc: storage)  # load checkpoint to CPU and then put to device https://discuss.pytorch.org/t/saving-and-loading-torch-models-on-2-machines-with-different-number-of-gpu-devices/6666
    model = UNet(n_classes=checkpoint["n_classes"], in_channels=checkpoint["in_channels"],
                 padding=checkpoint["padding"], depth=checkpoint["depth"], wf=checkpoint["wf"],
                 up_mode=checkpoint["up_mode"], batch_norm=checkpoint["batch_norm"]).to(device)
    model.load_state_dict(checkpoint["model_dict"])
    model.eval()
    tta_model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')

    logger.warning(f"Всего параметров: \t{sum([np.prod(p.size()) for p in model.parameters()])}")

    # ----- get image_file

    fname = kwargs['image_path']

    newfname_class = kwargs['pred_path']

    logger.warning(f"Работаем с файлом: \t {fname}")
    logger.warning(f"Сохраняем: \t {newfname_class}")

    if not kwargs['force'] and os.path.exists(newfname_class):
        print("Skipping as output file exists", flush=True)
    else:
        io = cv2.cvtColor(cv2.imread(fname), cv2.COLOR_BGR2RGB)
        io = cv2.resize(io, (0, 0), fx=resize, fy=resize)

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
        output = np.zeros((0, checkpoint["n_classes"], patch_size, patch_size))
        batch_index = 0
        for batch_arr in divide_batch(arr_out, batch_size):
            batch_index += 1
            print(f'PROGRESS: Generating Prediction - File , Batch {batch_size * batch_index}/{arr_out.shape[0]}',
                  flush=True)
            arr_out_gpu = torch.from_numpy(batch_arr.transpose(0, 3, 1, 2) / 255).type('torch.FloatTensor').to(device)

            # ---- get results
            output_batch = tta_model(arr_out_gpu)

            # --- pull from GPU and append to rest of output
            output_batch = output_batch.detach().cpu().numpy()

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

        # --- save output

        # cv2.imwrite(newfname_class, (output.argmax(axis=2) * (256 / (output.shape[-1] - 1) - 1)).astype(np.uint8))
        cv2.imwrite(newfname_class, output.argmax(axis=2) * (256 / (output.shape[-1] - 1) - 1))