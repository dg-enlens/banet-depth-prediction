# Copyright Enlens LLC, 2021. All rights reserved.
#
# This software is licensed under the terms of the "BANet Monocular Depth Prediction" licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# Author: Denis Girard (denis.girard@enlens.net)

import time
from typing import Callable, Tuple, Any

import progressbar
import numpy as np
import torch
import torch.nn as nn
from numpy import ndarray
from torch.utils.data import Dataset

from parameters_config import ParametersConfig
from metrics import SILogMetrics, AccuracyMetrics
from loss import relative_mae_loss
from utils import load_model


class Inference:
    """
    Helper class allowing to perform inference.

    Args:
        model (nn.Module): Model architecture provided as a torch nn.Module.
        device (str): cpu or cuda.

    Attributes:
        model (nn.Module): Model architecture provided as a torch nn.Module.
        device (str): cpu or cuda.
        last_elapsed_time (float): last inference elapsed time, used for internal time measures.
    """

    def __init__(self, model: nn.Module, device: str = 'cpu') -> None:
        # class members
        self.model = model
        self.device = device
        self.last_elapsed_time = 0.0

        # change model in eval mode
        self.model.eval()

    def predict_single(self, img: torch.Tensor) -> np.ndarray:
        """
        Perform prediction on a single image.

        Args:
            img (torch.Tensor): Image as a normalized tensor.

        Returns:
            np.ndarray: Predicted depth-map.
        """

        # disable grad
        with torch.no_grad():
            # put model to device
            self.model.to(self.device)

            # send data to device (it is mandatory if GPU has to be used)
            img = img.to(self.device)

            # prepare time measurement
            t_before = time.time()

            # forward pass to the model and measure inference time
            prediction = self.model(img.unsqueeze(dim=0))

            # measure inference time
            t_after = time.time()
            self.last_elapsed_time = t_after - t_before

            # perform post-transformation
            prediction = ParametersConfig.max_depth * prediction.squeeze(dim=0).squeeze(dim=0)

            # retransform back to numpy
            prediction = prediction.cpu().numpy().astype(np.float32)

            return prediction

    def get_last_prediction_time(self) -> float:
        """
        Get last prediction duration.

        Returns:
            float: Predicted depth-map.
        """

        duration = self.last_elapsed_time
        self.last_elapsed_time = 0.0

        return duration

    def load_model(self, model_path: str = 'model.pt') -> None:
        """
        Load model weights from file.

        Args:
            model_path (str): Path to the model file.

        Returns:
            nn.Module: Model with updated weights.
        """

        # load model
        load_model(self.model, model_path)

        # change model in eval mode
        self.model.eval()

    def evaluate_on_dataset(
            self,
            dataset: Dataset,
            loss_function: Callable = relative_mae_loss,
            metrics: AccuracyMetrics = SILogMetrics()
    ) -> Tuple[ndarray, Any, ndarray]:
        """
        Evaluate the model on the provided dataset.

        Args:
            dataset (Dataset): Input dataset.
            loss_function (str): Loss function.
            metrics (SILogMetrics): Metrics.

        Returns:
            (float, float, float): Average loss, metric value and average prediction time.
        """

        # disable grad
        with torch.no_grad():
            # dataloader
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=1
            )

            # accumulators
            loss_accumulator = []
            duration_accumulator = []

            # reset metric
            metrics.reset()

            # put model to device
            self.model.to(self.device)

            # display a progressbar with ETA for the epoch
            with progressbar.ProgressBar(
                    widgets=[progressbar.Bar(),
                             progressbar.widgets.SimpleProgress(format=' %(value_s)s of %(max_value_s)s       '),
                             progressbar.ETA()],
                    min_value=0,
                    max_value=len(dataloader.dataset)) as bar:

                # perform inference on the whole dataset
                for batch_idx, (data, target) in enumerate(dataloader):
                    # update progress bar
                    bar.update(batch_idx)

                    # send data to device (it is mandatory if GPU has to be used)
                    data = data.to(self.device)

                    # send target to device
                    target = target.to(self.device)

                    # take timestamp for duration computation
                    t_before = time.time()

                    # forward pass to the model
                    output = self.model(data)

                    # measure inference time
                    t_after = time.time()
                    duration_accumulator.append(t_after - t_before)

                    # compute loss
                    loss = loss_function(output, target)

                    # add loss to accumulator
                    loss_accumulator.append(loss.cpu())

                    # update metrics
                    metrics.update(output.cpu(), target.cpu())

                # get mean loss
                mean_loss = np.mean(loss_accumulator)

                # get metrics
                metric = metrics.value()

                # get mean inference time
                mean_duration = np.mean(duration_accumulator)

                return mean_loss, metric, mean_duration
