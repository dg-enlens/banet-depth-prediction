# Copyright Enlens LLC, 2021. All rights reserved.
#
# This software is licensed under the terms of the "BANet Monocular Depth Prediction" licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# Author: Denis Girard (denis.girard@enlens.net)

import os
from typing import Any

from numpy import ndarray
from torch import nn, Tensor
from torch.utils.data import Dataset

from torch.utils.mobile_optimizer import optimize_for_mobile

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import argparse

from torchvision.transforms import transforms
import torchvision.transforms.functional as tf

from parameters_config import ParametersConfig
from system_config import SystemConfig


def str2bool(v) -> bool:
    """
    Parse string to bool

    Args:
        v (str): String to be parsed to bool

    Raises:
        ArgumentTypeError: Boolean value expected.

    Returns:
        bool: Boolean.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_extension(filepath: str) -> str:
    """
    Returns file extension.

    Args:
        filepath (str): Filename or full path to file.

    Returns:
        str: File extension.
    """
    _, ext = os.path.splitext(filepath)
    return ext.lower()


def get_params_count(model: nn.Module) -> int:
    """
    Returns the number of parameters in the model.

    Args:
        model (nn.Module): Model architecture as a nn.Module.

    Returns:
        bool: Number of parameters.
    """
    return sum(p.numel() for p in model.parameters())


def apply_image_transform(img: ndarray) -> Tensor:
    """
    Prepare an image for prediction.

    Args:
        img (ndarray): Input image as a numpy array.

    Returns:
        Tensor: Output Tensor ready for inference.
    """

    # convert BGR (from OpenCV) to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # set image size
    out_w = ParametersConfig.input_size[0]
    out_h = ParametersConfig.input_size[1]

    # define resize transformation
    resize_img_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(out_h, tf.InterpolationMode.BILINEAR),
    ])

    # apply resize
    img = resize_img_transform(img)

    # center crop
    crop_transform = transforms.CenterCrop((out_h, out_w))
    img = crop_transform(img)

    # transform image to tensor
    img = transforms.ToTensor()(np.array(img))

    # normalize image
    dataset_mean = (0.4279, 0.4450, 0.4426)
    dataset_std = (0.3055, 0.3134, 0.3170)
    img_tensor = transforms.Normalize(dataset_mean, dataset_std)(img)

    return img_tensor


def complete_depthmap(depthmap: ndarray, kernel_size: int = 5, use_inpaint: bool = False) -> ndarray:
    """
    Experimental. Fill holes in the depth-map.

    Args:
        depthmap (ndarray): Input depth-map.
        kernel_size (int): Kernel size.
        use_inpaint (bool): If true, the InPaint algorithm is used (accurate but slow), otherwise fast
        morphological transform is used.

    Returns:
        ndarray: Filled depth-map.
    """
    if use_inpaint:

        # get pixels to reconstruct (null pixels in our depthmap)
        inpaint_mask = depthmap == 0

        # exclude parts of the depthmap which have not enough values for a proper reconstruction
        # (areas with a lot of zeros)
        depthmap_non_zeros = depthmap.astype(np.int16) > 0
        zero_areas = cv2.boxFilter(depthmap_non_zeros.astype(np.int16), -1, (kernel_size, kernel_size), normalize=False)
        zero_areas = zero_areas < kernel_size * kernel_size / 5
        inpaint_mask[zero_areas] = 0

        # reconstruct depthmap
        depthmap = cv2.inpaint(depthmap, inpaint_mask.astype(np.uint8), kernel_size, cv2.INPAINT_NS)

        return depthmap

    else:

        # 2D kernel sizer
        kernel_size = (kernel_size, kernel_size)

        # increase resolution of numpy matrix
        depthmap = depthmap.astype(np.int32)

        # get the non-zero mask (values of the depthmap which are non-null)
        non_zero_mask = depthmap > 0

        # apply a 2D filter. An output pixel is sum the value of all the surrounding 
        # pixels inside a (kernel_size, kernel_size) area of the input image
        filtered_dephtmap = cv2.boxFilter(depthmap, -1, kernel_size, normalize=False)

        # count the number of elements that we need to devide by (we want to exclude zero values of the mean
        # calculation)
        non_zero_count = cv2.boxFilter(non_zero_mask.astype(np.int32), -1, kernel_size, normalize=False)
        filtered_dephtmap[non_zero_count > 0] = filtered_dephtmap[non_zero_count > 0] / non_zero_count[
            non_zero_count > 0]
        filtered_dephtmap[non_zero_count == 0] = 0

        return filtered_dephtmap.astype(np.uint8)


def depthmap_to_rgb(depthmap: ndarray, mask: ndarray = None) -> ndarray:
    """
    Convert a depth-map to a fancy RGB picture.

    Args:
        depthmap (ndarray): Input depth-map.
        mask (ndarray): Optional mask, non-zero values are put to zero in the RGB depth-map.

    Returns:
        ndarray: Fancy RGB depth-map.
    """
    if depthmap.dtype == np.float32:
        depthmap = 255 - 5 * depthmap
        depthmap[depthmap < 0] = 0
        depthmap = depthmap.astype(np.uint8)
    else:
        depthmap = depthmap.astype(np.int16)
        depthmap = 255 - 5 * depthmap
        depthmap[depthmap < 0] = 5
        depthmap[depthmap == 255] = 0
        depthmap = depthmap.astype(np.uint8)

    depthmap_rgb = cv2.applyColorMap(depthmap, cv2.COLORMAP_INFERNO)

    if mask is not None:
        depthmap_rgb[mask] = (0, 0, 0)

    depthmap_rgb = cv2.cvtColor(depthmap_rgb, cv2.COLOR_BGR2RGB)

    return depthmap_rgb


def display_img_and_target(
        img: ndarray,
        target_depth: ndarray,
        predicted_depth: ndarray = None
) -> None:
    """
    Plot input RGB image, target depth-map and predicted depth-map.

    Args:
        img (ndarray): Input image.
        target_depth (ndarray): Target depth-map.
        predicted_depth (ndarray): Optional. Predicted depth-map.
    """

    # compute mask (i.e. pixels to keep to 0)
    mask = target_depth < 0

    if predicted_depth is None:

        target_rgb = depthmap_to_rgb(target_depth, mask)

        f, axarr = plt.subplots(1, 2, figsize=(12, 6))
        axarr[0].imshow(img)
        axarr[0].set_title('RGB image')
        axarr[1].imshow(target_rgb)
        axarr[1].set_title('Target depth')

        plt.show()

    else:

        target_rgb = depthmap_to_rgb(target_depth, mask)
        pred_rgb = depthmap_to_rgb(predicted_depth, mask)

        f, axarr = plt.subplots(1, 3, figsize=(18, 6))
        axarr[0].imshow(img)
        axarr[0].set_title('RGB image')
        axarr[1].imshow(target_rgb)
        axarr[1].set_title('Target depth')
        axarr[2].imshow(pred_rgb)
        axarr[2].set_title('Predicted depth')

        plt.show()


def compute_mean_std(dataset: Dataset, num_workers: int = 4) -> Any:
    """
    Compute the mean and standard deviation of a dataset.

    Args:
        dataset (Dataset): Input dataset.
        num_workers (int): Number of workers for the dataset loader.

    Returns:
        (Any): Mean and Std tuples (3 elements each).
    """

    dataset_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=num_workers
    )

    # accumulators for mean and std
    mean = 0.
    std = 0.

    # iterate over dataset
    for _, img_batch, depth_batch in dataset_loader:
        # batch size (the last batch can have smaller size!)
        batch_samples = img_batch.size(0)
        img_batch = img_batch.view(batch_samples, img_batch.size(1), -1)
        mean += img_batch.mean(2).sum(0)
        std += img_batch.std(2).sum(0)

    mean /= len(dataset_loader.dataset)
    std /= len(dataset_loader.dataset)

    print('Mean: {}, Std: {}'.format(mean, std))

    return mean, std


def save_model(
        model: nn.Module,
        device: str,
        model_dir: str = 'models',
        model_file_name: str = 'model.pt'
) -> None:
    """
    Save model weights to file.

    Args:
        model (nn.Module): Model architecture as a nn.Module.
        device (str): cpu or cuda.
        model_dir (str): Output directory path.
        model_file_name (str): Output filename.
    """

    # create models directory
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, model_file_name)

    # save the state_dict
    torch.save(model.state_dict(), model_path)
    print("Model saved to {}".format(model_path))

    if device == 'cuda':
        model.to('cuda')


def save_model_quantized(
        model: nn.Module,
        device: str,
        model_dir: str = 'models',
        model_file_name: str = 'model_quantized.pt'
) -> None:
    """
    Save a quantized model weights to file. Note that this model should have been prepared to be quantized
    (Quantization Aware Training)

    Args:
        model (nn.Module): Model prepared for Quantization Aware Training.
        device (str): cpu or cuda.
        model_dir (str): Output directory path.
        model_file_name (str): Output filename.
    """

    # create models directory
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # convert to quantized
    with torch.no_grad():

        # make sure you transfer the model to cpu.
        if device == 'cuda':
            model.to('cpu')

        model.eval()
        model_quantized = torch.quantization.convert(model, inplace=False)

        # save path
        model_path_jit = os.path.join(model_dir, model_file_name)

        # dummy input
        # x = torch.randn(1, 3, ParametersConfig.input_size[0], ParametersConfig.input_size[1])

        # export mobile optimized JIT
        traced_script_module = torch.jit.script(model_quantized)
        traced_script_module = optimize_for_mobile(traced_script_module)
        traced_script_module.save(model_path_jit)

        print("Quantized model saved to {}".format(model_path_jit))

        if device == 'cuda':
            model.to('cuda')


def load_model(
        model: nn.Module,
        model_path: str = 'model.pt'
) -> nn.Module:
    """
    Load model weights from file.

    Args:
        model (nn.Module): Model architecture as a nn.Module.
        model_path (str): Path to the model file.

    Returns:
        nn.Module: Model with updated weights.
    """

    # loading the model and getting model parameters by using load_state_dict
    model.load_state_dict(torch.load(model_path, map_location=SystemConfig.device))

    return model
