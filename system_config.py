# Copyright Enlens LLC, 2021. All rights reserved.
#
# This software is licensed under the terms of the "BANet Monocular Depth Prediction" licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# Author: Denis Girard (denis.girard@enlens.net)

from dataclasses import dataclass
from typing import Type

import random

import torch
import numpy


@dataclass
class SystemConfig:
    """
    Static class holding system configuration.
    """

    # seed number to set the state of all random number generators
    seed: int = 42

    # enable CuDNN benchmark for the sake of performance
    cudnn_benchmark_enabled: bool = True

    # make cudnn deterministic (reproducible training)
    cudnn_deterministic: bool = True

    # set device to CPU or CUDA
    device = 'cuda'


def setup_system(system_config: Type[SystemConfig]) -> None:
    """
    Prepare system.

    Args:
        system_config (Type[SystemConfig]): SystemConfig parameters.
    """
    torch.manual_seed(system_config.seed)
    numpy.random.seed(system_config.seed)
    random.seed(system_config.seed)
    torch.set_printoptions(precision=10)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(system_config.seed)
        torch.backends.cudnn_benchmark_enabled = system_config.cudnn_benchmark_enabled
        torch.backends.cudnn.deterministic = system_config.cudnn_deterministic
