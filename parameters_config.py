# Copyright Enlens LLC, 2021. All rights reserved.
#
# This software is licensed under the terms of the "BANet Monocular Depth Prediction" licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# Author: Denis Girard (denis.girard@enlens.net)

from dataclasses import dataclass

from typing import Iterable


@dataclass
class ParametersConfig:
    """
    Static class holding hyper-parameters configuration.
    """

    # number of concurrent workers
    num_workers: int = 4

    # batch size
    batch_size: int = 16

    # number of epochs
    num_epoch: int = 50

    # learning rate
    learning_rate: float = 1e-4

    # momentum, used by some optimizers
    momentum: float = 0.9

    # amount of additional regularization on the weights values
    weight_decay: float = 0  # 1e-5

    # at which epochs should we make a "step" in learning rate (i.e. decrease it in some manner)
    lr_step_milestones: Iterable = (25, 40)

    # multiplier applied to current learning rate at each of lr_step_milestones
    lr_gamma: float = 0.1

    # input size (w, h)
    input_size: tuple = (256, 192)

    # max depth
    max_depth: float = 100.0
