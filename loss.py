# Copyright Enlens LLC, 2021. All rights reserved.
#
# This software is licensed under the terms of the "BANet Monocular Depth Prediction" licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# Author: Denis Girard (denis.girard@enlens.net)

import torch
import torch.nn.functional as functional
from torch import Tensor


def relative_mse_loss(prediction: Tensor, target: Tensor, mask_zero: bool = False) -> float:
    """
    Compute MSE loss.

    Args:
        prediction (Tensor): Prediction.
        target (Tensor): Target.
        mask_zero (bool): Exclude zero values from the computation.

    Returns:
        float: MSE loss.
    """

    # let's only compute the loss on non-null pixels from the ground-truth depth-map
    if mask_zero:
        non_zero_mask = target > 0
        masked_input = prediction[non_zero_mask]
        masked_target = target[non_zero_mask]
    else:
        masked_input = prediction
        masked_target = target

    # Prediction MSE loss
    pred_mse = functional.mse_loss(masked_input, masked_target)

    # Self MSE loss for mean target
    target_mse = functional.mse_loss(masked_target, torch.ones_like(masked_target) * torch.mean(masked_target))

    return pred_mse / target_mse * 100


def relative_mae_loss(prediction: Tensor, target: Tensor, mask_zero: bool = True):
    """
    Compute MAE loss.

    Args:
        prediction (Tensor): Prediction.
        target (Tensor): Target.
        mask_zero (bool): Exclude zero values from the computation.

    Returns:
        float: MAE loss.
    """

    # let's only compute the loss on non-null pixels from the ground-truth depth-map
    if mask_zero:
        non_zero_mask = target > 0
        masked_input = prediction[non_zero_mask]
        masked_target = target[non_zero_mask]
    else:
        masked_input = prediction
        masked_target = target

    # Prediction MSE loss
    pred_mae = functional.l1_loss(masked_input, masked_target)

    # Self MSE loss for mean target
    target_mae = functional.l1_loss(masked_target, torch.ones_like(masked_target) * torch.mean(masked_target))

    return pred_mae / target_mae * 100


def silog_loss(prediction: Tensor, target: Tensor, variance_focus: float = 0.85) -> float:
    """
    Compute SILog loss. See https://papers.nips.cc/paper/2014/file/7bccfde7714a1ebadf06c5f4cea752c1-Paper.pdf for
    more information about scale-invariant loss.

    Args:
        prediction (Tensor): Prediction.
        target (Tensor): Target.
        variance_focus (float): Variance focus for the SILog computation.

    Returns:
        float: SILog loss.
    """

    # let's only compute the loss on non-null pixels from the ground-truth depth-map
    non_zero_mask = (target > 0) & (prediction > 0)

    # SILog
    d = torch.log(prediction[non_zero_mask]) - torch.log(target[non_zero_mask])
    return torch.sqrt((d ** 2).mean() - variance_focus * (d.mean() ** 2)) * 10.0
