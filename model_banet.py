# Copyright Enlens LLC, 2021. All rights reserved.
#
# This software is licensed under the terms of the "BANet Monocular Depth Prediction" licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# Author: Denis Girard (denis.girard@enlens.net)

import torch
import torch.nn as nn
from torch import Tensor
from torchvision import models

import torch.nn.functional as functional


class DepthToSpace(nn.Module):
    """
    Perform the depth-to-space transformation from tensorflow:
    https://www.tensorflow.org/api_docs/python/tf/nn/depth_to_space

    Args:
        block_size (int): Block size.

    Attributes:
        bs (int): Block size.
    """

    def __init__(self, block_size: int) -> None:
        super().__init__()
        self.bs = block_size

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x (Tensor): input Tensor.

        Returns:
            Tensor: output Tensor.
        """
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, int(C // (self.bs ** 2)), H, W)  # (N, bs, bs, C//bs^2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
        x = x.view(N, int(C // (self.bs ** 2)), H * self.bs, W * self.bs)  # (N, C//bs^2, H * bs, W * bs)
        return x


class D2SBlock(nn.Module):
    """
    Depth-to-space block as described in BANet paper.

    Args:
        channels_in (int): Number of input channels.
        block_size (int): Block size.
        image_size (tuple): Image size as a tuple (height, width)
    """

    def __init__(self, channels_in: int, block_size: int, image_size: tuple = (192, 256)):
        super().__init__()

        # keep image size
        self.image_size = image_size

        # 1x1 projection module adapting channels count before depth-to-space operation
        self.proj = nn.Sequential(
            # batch normalization
            nn.BatchNorm2d(channels_in),
            # relu activation
            nn.ReLU(),
            # convolution
            nn.Conv2d(channels_in, block_size ** 2, kernel_size=(1, 1), bias=False)
        )

        # depth-to-space operation
        self.depth_to_space = DepthToSpace(block_size)

        # average pooling
        self.avg2d = nn.AvgPool2d(kernel_size=32)

        # fully connected layer
        fc_channels = (image_size[0] // 32) * (image_size[1] // 32)
        self.fc = nn.Sequential(
            nn.Linear(fc_channels, fc_channels),
            nn.ReLU()
        )

        # 3x3 convolution
        self.conv = nn.Conv2d(2, 2, kernel_size=(3, 3), padding=(1, 1))

        # final BN+ReLU sequence
        self.bn_relu = nn.Sequential(
            # batch normalization
            nn.BatchNorm2d(1),
            # relu
            nn.ReLU()
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): input Tensor.

        Returns:
            Tensor: output Tensor.
        """

        # init process (conv 1x1)
        init = self.proj(x)

        # depth-to-space operation
        depth_to_space = self.depth_to_space(init)

        # avg pooling
        avg_pool = self.avg2d(depth_to_space)

        # fully connected layer
        avg_pool = avg_pool.view(avg_pool.size()[0], -1)
        fc = self.fc(avg_pool)

        # resize tensor after fully connected layer
        fc = fc.view(fc.size(0), 1, self.image_size[0] // 32, self.image_size[1] // 32)

        # upsampling
        up = functional.interpolate(input=fc, size=self.image_size, mode='bilinear', align_corners=False)

        # concat init and sequence 1
        concat = torch.cat([depth_to_space, up], dim=1)

        # 3x3 conv and softmax
        soft = functional.softmax(self.conv(concat), dim=1)

        # element-wise product concat and and softmax, then sum
        mul = torch.mul(concat, soft)
        output = torch.sum(mul, dim=1, keepdim=True)

        # final bn-relu
        output = self.bn_relu(output)

        return output


class D2SBlock_Truncated(nn.Module):
    """
    Depth-to-space block truncated of its last BN+ReLu layer, as described in BANet paper.

    Args:
        channels_in (int): Number of input channels.
        block_size (int): Block size.
        image_size (tuple): Image size as a tuple (height, width)
    """

    def __init__(self, channels_in: int, block_size: int, image_size: tuple = (192, 256)):
        super().__init__()

        # keep image size
        self.image_size = image_size

        # 1x1 projection module adapting channels count before depth-to-space operation
        self.proj = nn.Sequential(
            # batch normalization
            nn.BatchNorm2d(channels_in),
            # relu activation
            nn.ReLU(),
            # convolution
            nn.Conv2d(channels_in, block_size ** 2, kernel_size=(1, 1), bias=False)
        )

        # depth-to-space operation
        self.depth_to_space = DepthToSpace(block_size)

        # average pooling
        self.avg2d = nn.AvgPool2d(kernel_size=32)

        # fully connected layer
        fc_channels = (image_size[0] // 32) * (image_size[1] // 32)
        self.fc = nn.Sequential(
            nn.Linear(fc_channels, fc_channels),
            nn.ReLU()
        )

        # 3x3 convolution
        self.conv = nn.Conv2d(2, 2, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): input Tensor.

        Returns:
            Tensor: output Tensor.
        """

        # init process (conv 1x1)
        init = self.proj(x)

        # depth-to-space operation
        depth_to_space = self.depth_to_space(init)

        # avg pooling
        avg_pool = self.avg2d(depth_to_space)

        # fully connected layer
        avg_pool = avg_pool.view(avg_pool.size()[0], -1)
        fc = self.fc(avg_pool)

        # resize tensor after fully connected layer
        fc = fc.view(fc.size(0), 1, self.image_size[0] // 32, self.image_size[1] // 32)

        # upsampling
        up = functional.interpolate(input=fc, size=self.image_size, mode='bilinear', align_corners=False)

        # concat init and sequence 1
        concat = torch.cat([depth_to_space, up], dim=1)

        # 3x3 conv and softmax
        soft = functional.softmax(self.conv(concat), dim=1)

        # element-wise product concat and and softmax, then sum
        mul = torch.mul(concat, soft)
        output = torch.sum(mul, dim=1, keepdim=True)

        return output


class BANet_DenseNet121(nn.Module):
    """
    BANet network with a DenseNet121 backbone.

    Args:
        image_size (tuple): Image size as a tuple (height, width)
    """

    def __init__(self, image_size=(192, 256)):
        super().__init__()

        # keep image size
        self.image_size = image_size

        # prepare feature extractor from `torchvision` DenseNet model
        feature_extractor = models.densenet121(pretrained=True)

        self.S1 = nn.Sequential(
            feature_extractor.features.conv0,
            feature_extractor.features.norm0,
            feature_extractor.features.relu0,
            feature_extractor.features.pool0
        )
        self.S2 = nn.Sequential(
            feature_extractor.features.denseblock1,
            feature_extractor.features.transition1
        )
        self.S3 = nn.Sequential(
            feature_extractor.features.denseblock2,
            feature_extractor.features.transition2
        )
        self.S4 = nn.Sequential(
            feature_extractor.features.denseblock3,
            feature_extractor.features.transition3
        )
        self.S5 = feature_extractor.features.denseblock4

        # forward D2S layers
        self.d2s_F1 = D2SBlock(channels_in=64, block_size=4, image_size=self.image_size)
        self.d2s_F2 = D2SBlock(channels_in=128, block_size=8, image_size=self.image_size)
        self.d2s_F3 = D2SBlock(channels_in=256, block_size=16, image_size=self.image_size)
        self.d2s_F4 = D2SBlock(channels_in=512, block_size=32, image_size=self.image_size)
        self.d2s_F5 = D2SBlock(channels_in=1024, block_size=32, image_size=self.image_size)

        # forward 9x9 conv layers
        self.conv9x9_F1 = nn.Conv2d(1, 1, kernel_size=(9, 9), padding=(4, 4))
        self.conv9x9_F2 = nn.Conv2d(2, 1, kernel_size=(9, 9), padding=(4, 4))
        self.conv9x9_F3 = nn.Conv2d(3, 1, kernel_size=(9, 9), padding=(4, 4))
        self.conv9x9_F4 = nn.Conv2d(4, 1, kernel_size=(9, 9), padding=(4, 4))
        self.conv9x9_F5 = nn.Conv2d(5, 1, kernel_size=(9, 9), padding=(4, 4))

        # forward 3x3 conv layer
        self.conv3x3_F = nn.Conv2d(5, 5, kernel_size=(3, 3), padding=(1, 1))

        # backward D2S layers
        self.d2s_B1 = D2SBlock(channels_in=64, block_size=4, image_size=self.image_size)
        self.d2s_B2 = D2SBlock(channels_in=128, block_size=8, image_size=self.image_size)
        self.d2s_B3 = D2SBlock(channels_in=256, block_size=16, image_size=self.image_size)
        self.d2s_B4 = D2SBlock(channels_in=512, block_size=32, image_size=self.image_size)
        self.d2s_B5 = D2SBlock(channels_in=1024, block_size=32, image_size=self.image_size)

        # backward 9x9 conv layers
        self.conv9x9_B1 = nn.Conv2d(5, 1, kernel_size=(9, 9), padding=(4, 4))
        self.conv9x9_B2 = nn.Conv2d(4, 1, kernel_size=(9, 9), padding=(4, 4))
        self.conv9x9_B3 = nn.Conv2d(3, 1, kernel_size=(9, 9), padding=(4, 4))
        self.conv9x9_B4 = nn.Conv2d(2, 1, kernel_size=(9, 9), padding=(4, 4))
        self.conv9x9_B5 = nn.Conv2d(1, 1, kernel_size=(9, 9), padding=(4, 4))

        # backward 3x3 conv layer
        self.conv3x3_B = nn.Conv2d(5, 5, kernel_size=(3, 3), padding=(1, 1))

        # features computation D2S_Truncated layers
        self.d2s_Feat1 = D2SBlock_Truncated(channels_in=64, block_size=4, image_size=self.image_size)
        self.d2s_Feat2 = D2SBlock_Truncated(channels_in=128, block_size=8, image_size=self.image_size)
        self.d2s_Feat3 = D2SBlock_Truncated(channels_in=256, block_size=16, image_size=self.image_size)
        self.d2s_Feat4 = D2SBlock_Truncated(channels_in=512, block_size=32, image_size=self.image_size)
        self.d2s_Feat5 = D2SBlock_Truncated(channels_in=1024, block_size=32, image_size=self.image_size)

        # final 3x3 conv for attention maps
        self.conv3x3_A = nn.Sequential(
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 5, kernel_size=(3, 3), padding=(1, 1))
        )

        # sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): input 3-channel normalized image.

        Returns:
            Tensor: output normalized predicted depth-map.
        """

        # last channel memory format
        x.contiguous(memory_format=torch.channels_last)

        # output size = (64, 128, 160)
        s1 = self.S1(x)
        # output size = (64, 64, 80)
        s2 = self.S2(s1)
        # output size = (128, 32, 40)
        s3 = self.S3(s2)
        # output size = (256, 16, 20)
        s4 = self.S4(s3)
        # output size = (512, 8, 10)
        s5 = self.S5(s4)

        # forward depth-to-space
        d2s_F1 = self.d2s_F1(s1)
        d2s_F2 = self.d2s_F2(s2)
        d2s_F3 = self.d2s_F3(s3)
        d2s_F4 = self.d2s_F4(s4)
        d2s_F5 = self.d2s_F5(s5)

        # forward concat d2s before 9x9 conv
        pre9x9_F1 = d2s_F1
        pre9x9_F2 = torch.cat([d2s_F1, d2s_F2], dim=1)
        pre9x9_F3 = torch.cat([d2s_F1, d2s_F2, d2s_F3], dim=1)
        pre9x9_F4 = torch.cat([d2s_F1, d2s_F2, d2s_F3, d2s_F4], dim=1)
        pre9x9_F5 = torch.cat([d2s_F1, d2s_F2, d2s_F3, d2s_F4, d2s_F5], dim=1)

        # forward conv9x9
        conv9x9_F1 = self.conv9x9_F1(pre9x9_F1)
        conv9x9_F2 = self.conv9x9_F2(pre9x9_F2)
        conv9x9_F3 = self.conv9x9_F3(pre9x9_F3)
        conv9x9_F4 = self.conv9x9_F4(pre9x9_F4)
        conv9x9_F5 = self.conv9x9_F5(pre9x9_F5)

        # forward concat of attention maps, then 3x3 convolution
        pre3x3_F = torch.cat([conv9x9_F1, conv9x9_F2, conv9x9_F3, conv9x9_F4, conv9x9_F5], dim=1)
        conv3x3_F = self.conv3x3_F(pre3x3_F)

        # backward depth-to-space
        d2s_B1 = self.d2s_B1(s1)
        d2s_B2 = self.d2s_B2(s2)
        d2s_B3 = self.d2s_B3(s3)
        d2s_B4 = self.d2s_B4(s4)
        d2s_B5 = self.d2s_B5(s5)

        # backward concat d2s before 9x9 conv
        pre9x9_B1 = torch.cat([d2s_B1, d2s_B2, d2s_B3, d2s_B4, d2s_B5], dim=1)
        pre9x9_B2 = torch.cat([d2s_B2, d2s_B3, d2s_B4, d2s_B5], dim=1)
        pre9x9_B3 = torch.cat([d2s_B3, d2s_B4, d2s_B5], dim=1)
        pre9x9_B4 = torch.cat([d2s_B4, d2s_B5], dim=1)
        pre9x9_B5 = d2s_B5

        # backward conv9x9
        conv9x9_B1 = self.conv9x9_B1(pre9x9_B1)
        conv9x9_B2 = self.conv9x9_B2(pre9x9_B2)
        conv9x9_B3 = self.conv9x9_B3(pre9x9_B3)
        conv9x9_B4 = self.conv9x9_B4(pre9x9_B4)
        conv9x9_B5 = self.conv9x9_B5(pre9x9_B5)

        # backward concat of attention maps, then 3x3 convolution
        pre3x3_B = torch.cat([conv9x9_B1, conv9x9_B2, conv9x9_B3, conv9x9_B4, conv9x9_B5], dim=1)
        conv3x3_B = self.conv3x3_B(pre3x3_B)

        # contact and final convolution of attention maps
        concat_A = torch.cat([conv3x3_F, conv3x3_B], dim=1)
        attention_map = self.conv3x3_A(concat_A)
        attention_map = functional.softmax(attention_map, dim=1)

        # features computation
        d2s_Feat1 = self.d2s_Feat1(s1)
        d2s_Feat2 = self.d2s_Feat2(s2)
        d2s_Feat3 = self.d2s_Feat3(s3)
        d2s_Feat4 = self.d2s_Feat4(s4)
        d2s_Feat5 = self.d2s_Feat5(s5)

        # features concatenation
        features = torch.cat([d2s_Feat1, d2s_Feat2, d2s_Feat3, d2s_Feat4, d2s_Feat5], dim=1)

        # multiply features with attention map, sum, sigmoid and return
        output = torch.mul(features, attention_map)
        output = torch.sum(output, dim=1, keepdim=True)
        output = self.sigmoid(output)

        return output
