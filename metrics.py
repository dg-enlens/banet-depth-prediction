# Copyright Enlens LLC, 2021. All rights reserved.
#
# This software is licensed under the terms of the "BANet Monocular Depth Prediction" licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# Author: Denis Girard (denis.girard@enlens.net)

import abc
from typing import Any

import numpy as np
import torch
from torch import Tensor


class AccuracyMetrics(abc.ABC):
    """
    Abstract class providing an interface to evaluate custom metrics.

    Attributes:
        accumulator: Accumulate metrics, used to compute the mean metrics later on.
    """

    def __init__(self) -> None:
        self.accumulator = []

    def update(self, prediction: Tensor, target: Tensor) -> None:
        """
        Compute a new metrics for the provided prediction / target pair.

        Args:
            prediction (Tensor): Prediction.
            target (Tensor): Target.
        """
        silog = self.compute_loss(prediction=prediction, target=target)
        self.accumulator.append(silog)

    def value(self) -> Any:
        """
        Returns the average metrics.

        Returns:
            Any: Average metrics
        """
        return np.mean(self.accumulator)

    def reset(self) -> None:
        """
        Reset all metrics and clear the accumulator.
        """
        self.accumulator = []

    @abc.abstractmethod
    def compute_loss(self, prediction: Tensor, target: Tensor) -> Tensor:
        """
        Method to implement, should compute the loss metrics.

        Args:
            prediction (Tensor): Prediction.
            target (Tensor): Target.

        Returns:
            Tensor: Loss metrics.
        """
        pass


class SILogMetrics(AccuracyMetrics):
    """
    Class implementing AccuracyMetrics. Used to compute SILog metrics (which is actually a loss function).
    """

    def compute_loss(self, prediction, target) -> Tensor:
        """
        Compute the SILog metrics.

        Args:
            prediction (Tensor): Prediction.
            target (Tensor): Target.

        Returns:
            Tensor: SILog metrics.
        """

        # rescale depth-map values to their proper scale
        prediction = prediction * 256
        target = target * 256

        # let's only compute the loss on non-null pixels from the ground-truth depth-map
        non_zero_mask = (target > 0) & (prediction > 0)
        masked_input = prediction[non_zero_mask]
        masked_target = target[non_zero_mask]

        # see https://papers.nips.cc/paper/2014/file/7bccfde7714a1ebadf06c5f4cea752c1-Paper.pdf for SILog formula
        d = torch.log(masked_input) - torch.log(masked_target)

        return torch.sqrt(torch.mean(d ** 2) - torch.mean(d) ** 2) * 100
