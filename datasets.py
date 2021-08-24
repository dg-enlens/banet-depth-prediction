# Copyright Enlens LLC, 2021. All rights reserved.
#
# This software is licensed under the terms of the "BANet Monocular Depth Prediction" licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# Author: Denis Girard (denis.girard@enlens.net)

import random
from typing import Tuple

import cv2
import numpy as np
import pandas as pd

import torchvision.transforms.functional as tf
from numpy import ndarray
from torch import Tensor

from torch.utils.data import Dataset
from torchvision import transforms

from parameters_config import ParametersConfig
from utils import complete_depthmap, get_extension


class MyDataset(Dataset):
    """
    Class allowing to conveniently access dataset images, applying transformations required to perform model inference
    as well as optional data augmentation.

    Args:
        input_csv_path (str): Path to the input CSV file listing training/validation data files.
        image_size (Tuple[int, int]): Image size (width, height).
        data_augmentation (bool): Apply data augmentation (random crop, random rotation and flip, random color jitter).
        complete_depth (bool): Apply an algorithm to fill holes in the depth-maps (experimental).

    Attributes:
        complete_depth (bool): Apply an algorithm to fill holes in the depth-maps (experimental).
        image_size (tuple): Image size (width, height).
        data_augmentation (bool): Apply data augmentation (random crop, random rotation and flip, random color jitter).
        df_raw_and_depths (DataFrame): Panda frame resulting in reading the CSV file.
        data_dict (dict): Dictionary holding image and depth-map paths.
        transform_to_tensor (transforms): transform PIL Image to tensor.
        dataset_mean (tuple): Mean of the dataset across all 3 channels.
        dataset_std (tuple): Std of the dataset across all 3 channels.
    """

    def __init__(
            self,
            input_csv_path: str,
            image_size: Tuple[int, int] = (256, 192),
            data_augmentation: bool = False,
            complete_depth: bool = False
    ) -> None:
        # init class members
        self.complete_depthmap = complete_depth
        self.image_size = image_size
        self.data_augmentation = data_augmentation

        # read csv (raw image paths and corresponding depth-map paths)
        self.df_raw_and_depths = pd.read_csv(input_csv_path, delimiter=' *, *', engine='python')

        # initialize the data dictionary
        self.data_dict = {
            'raw_path': [],
            'depth_path': []
        }

        # fill dictionary
        for index, row in self.df_raw_and_depths.iterrows():
            # get paths from CSV
            raw_path = row['raw_path']
            depth_path = row['depth_path']

            # add entry in dictionary
            self.data_dict['raw_path'].append(raw_path)
            self.data_dict['depth_path'].append(depth_path)

        # set to tensor transformation
        self.transform_to_tensor = transforms.ToTensor()

        # set mean and std of the dataset (determined with utils.compute_mean_std())
        self.dataset_mean = (0.4279, 0.4450, 0.4426)
        self.dataset_std = (0.3055, 0.3134, 0.3170)

    def __len__(self) -> int:
        """
        Return size of the dataset.

        Returns:
            int: Size of the dataset.
        """
        return len(self.data_dict['raw_path'])

    def __getitem__(self, idx) -> (Tensor, Tensor):
        """
        Return the image and depth-map for the provided dataset index. Required and augmentation
        transformation are applied.

        Args:
            idx (int): Index of image and depth-map in the dataset.

        Raises:
            Exception: Unsupported depth-map file. Image files (jpg, jpeg, png, bmp) amd .npz are supported.

        Returns:
            (Tensor, Tensor): image and corresponding depth-map as formatted tensors.
        """

        # open raw image and depthmap
        img = cv2.imread(self.data_dict['raw_path'][idx])

        # check depth-map extension and open the file accordingly
        file_ext = get_extension(self.data_dict['depth_path'][idx])

        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            depthmap = cv2.imread(self.data_dict['depth_path'][idx], cv2.IMREAD_GRAYSCALE)
        elif file_ext == '.npz':
            depthmap = next(iter(np.load(self.data_dict['depth_path'][idx]).items()))[1]
            depthmap[np.isnan(depthmap)] = 0
            depthmap[depthmap > ParametersConfig.max_depth] = ParametersConfig.max_depth
        elif file_ext == '.npy':
            depthmap = np.load(self.data_dict['depth_path'][idx])
            depthmap[np.isnan(depthmap)] = 0
            depthmap[depthmap > ParametersConfig.max_depth] = ParametersConfig.max_depth
        else:
            raise Exception('Unsupported depth-map file. Image files (jpg, jpeg, png, bmp) amd .npz are supported')

        # Convert depth to float32 and normalize
        depthmap = depthmap.astype(np.float32)
        depthmap = depthmap / ParametersConfig.max_depth

        # convert BGR (from OpenCV) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # complete depthmap if required
        if self.complete_depthmap:
            depthmap = complete_depthmap(depthmap)

        out_w = img.shape[1]
        out_h = img.shape[0]

        if self.image_size is not None:
            out_w = self.image_size[0]
            out_h = self.image_size[1]

        # define resize transformation
        resize_img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(out_h, tf.InterpolationMode.BILINEAR),
        ])
        resize_depth_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(out_h, tf.InterpolationMode.NEAREST),
        ])

        # apply resize
        img = resize_img_transform(img)
        depthmap = resize_depth_transform(depthmap)

        # perform data augmentation
        if self.data_augmentation:

            # select a random crop
            i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(out_h, out_w))

            # apply the same random crop to img and depthmap
            img = tf.crop(img, i, j, h, w)
            depthmap = tf.crop(depthmap, i, j, h, w)

            # random flip
            if random.random() > 0.5:
                img = tf.hflip(img)
                depthmap = tf.hflip(depthmap)
            if random.random() > 0.5:
                img = tf.vflip(img)
                depthmap = tf.vflip(depthmap)

            # select a random rotation
            angle = random.randrange(-45, 45)
            img = tf.rotate(img, angle, tf.InterpolationMode.BILINEAR)
            depthmap = tf.rotate(depthmap, angle, tf.InterpolationMode.NEAREST)

            # apply some color noise on image
            color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
            img = color_jitter(img)

        else:

            # if no data augmentation, just center crop
            crop_transform = transforms.CenterCrop((out_h, out_w))
            img = crop_transform(img)
            depthmap = crop_transform(depthmap)

        # transform image to tensor
        img = self.transform_to_tensor(np.array(img))
        depthmap = self.transform_to_tensor(np.array(depthmap))

        # normalize image
        img = transforms.Normalize(self.dataset_mean, self.dataset_std)(img)

        # return idx, processed image, processed depthmap, and corresponding originals
        return img, depthmap

    def get_originals(self, idx) -> (ndarray, ndarray):
        """
        Return the original image and depth-map for the provided dataset index.

        Args:
            idx (int): Index of image and depth-map in the dataset.

        Returns:
            (ndarray, ndarray): image and corresponding depth-map as numpy arrays.
        """

        # open raw image and depthmap
        img = cv2.imread(self.data_dict['raw_path'][idx])

        # check depth-map extension and open the file accordingly
        file_ext = get_extension(self.data_dict['depth_path'][idx])

        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            depthmap = cv2.imread(self.data_dict['depth_path'][idx], cv2.IMREAD_GRAYSCALE)
        elif file_ext == '.npz':
            depthmap = next(iter(np.load(self.data_dict['depth_path'][idx]).items()))[1]
            depthmap[np.isnan(depthmap)] = 0
            depthmap[depthmap > ParametersConfig.max_depth] = ParametersConfig.max_depth
        elif file_ext == '.npy':
            depthmap = np.load(self.data_dict['depth_path'][idx])
            depthmap[np.isnan(depthmap)] = 0
            depthmap[depthmap > ParametersConfig.max_depth] = ParametersConfig.max_depth
        else:
            raise Exception('Unsupported depth-map file. Image files (jpg, jpeg, png, bmp) amd .npz are supported')

        return img, depthmap

    def get_not_augmented(self, idx, complete_depth=False) -> (ndarray, ndarray):
        """
        Return the image and depth-map for the provided dataset index. Only the transformations strictly required for
        model inference are applied (no augmentation).

        Args:
            idx (int): Index of image and depth-map in the dataset.
            complete_depth (bool): Complete holes in the depth-map (experimental).

        Returns:
            (ndarray, ndarray): Image and corresponding depth-map as ndarray.
        """

        # open raw image and depthmap
        img = cv2.imread(self.data_dict['raw_path'][idx])

        # check depth-map extension and open the file accordingly
        file_ext = get_extension(self.data_dict['depth_path'][idx])

        if file_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            depthmap = cv2.imread(self.data_dict['depth_path'][idx], cv2.IMREAD_GRAYSCALE)
        elif file_ext == '.npz':
            depthmap = next(iter(np.load(self.data_dict['depth_path'][idx]).items()))[1]
            depthmap[np.isnan(depthmap)] = 0
            depthmap[depthmap > ParametersConfig.max_depth] = ParametersConfig.max_depth
        elif file_ext == '.npy':
            depthmap = np.load(self.data_dict['depth_path'][idx])
            depthmap[np.isnan(depthmap)] = 0
            depthmap[depthmap > ParametersConfig.max_depth] = ParametersConfig.max_depth
        else:
            raise Exception('Unsupported depth-map file. Image files (jpg, jpeg, png, bmp) amd .npz are supported')

        # convert BGR (from OpenCV) to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # complete depthmap if required
        if complete_depth:
            depthmap = complete_depthmap(depthmap)

        out_w = img.shape[1]
        out_h = img.shape[0]

        if self.image_size is not None:
            out_w = self.image_size[0]
            out_h = self.image_size[1]

        # define resize transformation
        resize_img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(out_h, tf.InterpolationMode.BILINEAR),
        ])
        resize_depth_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(out_h, tf.InterpolationMode.NEAREST),
        ])

        # apply resize
        img = resize_img_transform(img)
        depthmap = resize_depth_transform(depthmap)

        # just center crop
        crop_transform = transforms.CenterCrop((out_h, out_w))
        img = crop_transform(img)
        depthmap = crop_transform(depthmap)

        img = np.copy(np.asarray(img))
        depthmap = np.copy(np.asarray(depthmap))

        return img, depthmap
