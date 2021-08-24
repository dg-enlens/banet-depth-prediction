# Copyright Enlens LLC, 2021. All rights reserved.
#
# This software is licensed under the terms of the "BANet Monocular Depth Prediction" licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# Author: Denis Girard (denis.girard@enlens.net)

import time

from numpy import ndarray
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from metrics import AccuracyMetrics
from parameters_config import ParametersConfig
from system_config import SystemConfig
from utils import save_model, save_model_quantized

import progressbar
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from typing import Callable, Any


class Trainer:
    """
    Helper class allowing to perform training.

    Args:
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        model (nn.Module): Model architecture provided as a torch nn.Module.
        loss_function (Callable): Loss function.
        metrics (AccuracyMetrics): Evaluation metrics.
        optimizer (Optimizer): Optimizer.
        lr_scheduler (Any): LR scheduler.
        tb_writer (SummaryWriter): Tensorboard summary writer.
        mobile_quantization (bool): Enable mobile quantization (only if available with the model).
        sleep_after_epoch (float): Seconds of sleep after every epochs.

    Attributes:
        model (nn.Module): Model architecture provided as a torch nn.Module.
        model (nn.Module): Model architecture provided as a torch nn.Module.
        loss_function (Callable): Loss function.
        metrics (AccuracyMetrics): Evaluation metrics.
        optimizer (Optimizer): Optimizer.
        lr_scheduler (Any): LR scheduler.
        tb_writer (SummaryWriter): Tensorboard summary writer.
        mobile_quantization (bool): Enable mobile quantization (only if available with the model).
        sleep_after_epoch (float): Seconds of sleep after every epochs.
        best_loss (Tensor): Best loss for all epochs.
        best_metrics (Tensor): Best loss for all epochs.
    """

    def __init__(
            self,
            train_dataset: Dataset,
            val_dataset: Dataset,
            model: nn.Module,
            loss_function: Callable,
            metrics: AccuracyMetrics,
            optimizer: Optimizer,
            lr_scheduler: Any,
            tb_writer: SummaryWriter = None,
            mobile_quantization: bool = False,
            sleep_after_epoch: float = 0.0):

        # set class members
        self.model = model
        self.loss_function = loss_function
        self.metrics = metrics
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.tb_writer = tb_writer
        self.best_loss = torch.tensor(np.inf)
        self.best_metrics = torch.tensor(np.inf)
        self.sleep_after_epoch = sleep_after_epoch
        self.mobile_quantization = mobile_quantization

        # set dataloaders
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=ParametersConfig.batch_size,
            shuffle=True,
            num_workers=ParametersConfig.num_workers
        )
        self.val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=ParametersConfig.batch_size,
            shuffle=False,
            num_workers=ParametersConfig.num_workers
        )

    def start_training(self) -> None:

        # upload model to device
        self.model.to(SystemConfig.device)

        # trainig time measurement
        time_measure = TimeMeasure(ParametersConfig.num_epoch)

        # train loop
        for epoch in range(ParametersConfig.num_epoch):

            # print current epoch
            print("\n------------------- Epoch {} -------------------".format(epoch))

            # train model
            epoch_train_loss, epoch_train_metrics = self.train()

            # compute elapsed time and ETA
            time_measure.tick_epoch(len(self.train_dataloader))
            time_measure.print_epoch_stats()

            # test updated model on validation set
            epoch_val_loss, epoch_val_metrics = self.validate()

            # add data to tensorboard
            if self.tb_writer is not None:
                time_measure.log_epoch_stats_to_tensorboard(self.tb_writer)

                # add scalar (training loss) to tensorboard
                self.tb_writer.add_scalar('Loss/Train', epoch_train_loss, epoch)
                self.tb_writer.add_scalar('SILog/Train', epoch_train_metrics, epoch)

                # add scalar (validation loss) to tensorboard
                self.tb_writer.add_scalar('Loss/Validation', epoch_val_loss, epoch)
                self.tb_writer.add_scalar('SILog/Validation', epoch_val_metrics, epoch)

                # add scalars (training vs. validation loss) to tensorboard
                self.tb_writer.add_scalars('Loss/Train-Val', {'train': epoch_train_loss, 'validation': epoch_val_loss},
                                           epoch)
                self.tb_writer.add_scalars('SILog/Train-Val',
                                           {'train': epoch_train_metrics, 'validation': epoch_val_metrics}, epoch)

            # update best loss
            if epoch_val_loss < self.best_loss:
                self.best_loss = epoch_val_loss

                # save model if a new best loss is reached
                save_model(self.model, device=SystemConfig.device)

                # save a quantized optimized version of the model if required
                if self.mobile_quantization:
                    save_model_quantized(self.model, device=SystemConfig.device)

            # save a model every 5 epochs
            if epoch % 5 == 0:

                # name model after epoch
                model_file_name = "epoch_{}.pt".format(epoch)

                # save model if a new best loss is reached
                save_model(self.model, device=SystemConfig.device, model_file_name=model_file_name)

                # save a quantized optimized version of the model if required
                if self.mobile_quantization:
                    # name model after epoch
                    model_file_name = "epoch_{}_quantized.pt".format(epoch)

                    # save quantized version
                    save_model_quantized(self.model, device=SystemConfig.device, model_file_name=model_file_name)

            # update best metrics
            if epoch_val_metrics < self.best_metrics:
                self.best_metrics = epoch_val_metrics

            # scheduler step / update learning rate
            self.update_scheduler(epoch_val_loss)

            # add scalar (training loss) to tensorboard
            if self.tb_writer is not None:
                self.tb_writer.add_scalar('Learning_rate/Learning_rate', self.lr_scheduler.get_last_lr()[0], epoch)

            # sleep if required
            time.sleep(self.sleep_after_epoch)

    def train(self) -> (ndarray, Any):
        """
        Train model on the training set. Returns loss and metrics.

        Returns:
            (ndarray, Any): Epoch loss and epoch metrics.
        """

        # change model in training mode
        self.model.train()

        # loss accumulator
        loss_accumulator = []

        # reset metric
        self.metrics.reset()

        # display a progressbar with ETA for the epoch
        with progressbar.ProgressBar(widgets=[progressbar.Bar(), progressbar.widgets.SimpleProgress(
                format=' %(value_s)s of %(max_value_s)s       '), progressbar.ETA()], min_value=0,
                                     max_value=len(self.train_dataloader.dataset)) as bar:
            # epoch training
            for batch_idx, (data, target) in enumerate(self.train_dataloader):
                # update progress bar
                bar.update(batch_idx * ParametersConfig.batch_size)

                # send data to device (it is mandatory if GPU has to be used)
                data = data.to(SystemConfig.device)

                # send target to device
                target = target.to(SystemConfig.device)

                # reset parameters gradient to zero
                self.optimizer.zero_grad()

                # forward pass to the model
                output = self.model(data)

                # compute loss
                loss = self.loss_function(output, target)

                # find gradients w.r.t training parameters
                loss.backward()

                # update parameters using gardients
                self.optimizer.step()

                # disable grad
                with torch.no_grad():
                    # add loss to accumulator
                    loss_accumulator.append(loss.cpu())

                    # update metrics
                    self.metrics.update(output.cpu(), target.cpu())

        # disable grad
        with torch.no_grad():
            # get mean loss
            epoch_loss = np.mean(loss_accumulator)

            # get metrics
            epoch_metrics = self.metrics.value()

            print('Train Loss: {:.6f}\nTrain SILog: {:.6f}\n'.format(epoch_loss, epoch_metrics))

            return epoch_loss, epoch_metrics

    def validate(self) -> (ndarray, Any):
        """
        Test model on the validation set. Returns loss and metrics.

        Returns:
            (ndarray, Any): Epoch loss and epoch metrics.
        """

        # change model in eval mode
        self.model.eval()

        # disable grad
        with torch.no_grad():
            # loss accumulator
            loss_accumulator = []

            # reset metric
            self.metrics.reset()

            for data, target in self.val_dataloader:
                # send data to device (it is mandatory if GPU has to be used)
                data = data.to(SystemConfig.device)

                # send target to device
                target = target.to(SystemConfig.device)

                # forward pass to the model
                output = self.model(data)

                # compute loss
                loss = self.loss_function(output, target)

                # add loss to accumulator
                loss_accumulator.append(loss.cpu())

                # update metrics
                self.metrics.update(output.cpu(), target.cpu())

            # get mean loss
            epoch_loss = np.mean(loss_accumulator)

            # get metrics
            epoch_metrics = self.metrics.value()

            print("Test Loss : {:.6f}\nTest SILog: {:.6f}\n".format(epoch_loss, epoch_metrics))

            return epoch_loss, epoch_metrics

    def update_scheduler(self, current_loss=None) -> None:
        """
        Update scheduler.
        """
        if self.lr_scheduler is not None:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(current_loss)
            else:
                self.lr_scheduler.step()


class TimeMeasure:
    """
    Helper class allowing to measure elapsed time.

    Args:
        epoch_count (int): Total number of epochs, used to estimate ETA.

    Attributes:
        t_begin (float): Start time, used as a reference.
        epochs_count (int): Total number of epochs.
        epoch (int): Current epoch.
        elapsed_time (float): Elapsed time from start.
        speed_epoch (float): Epoch duration.
        speed_batch (float): Mean batch duration for epoch.
        eta (float): ETA to complete training.
    """

    def __init__(self, epoch_count: int) -> None:
        self.t_begin = time.time()
        self.epochs_count = epoch_count
        self.epoch = 0
        self.elapsed_time = 0.0
        self.speed_epoch = 0.0
        self.speed_batch = 0.0
        self.eta = 0.0

    def tick_epoch(self, batch_count) -> (float, float, float, float):
        """
        Measure times for the epoch.

        Args:
            batch_count (int): Total number of batches for the epoch.

        Returns:
            (float, float, float, float): Elapsed time, epoch speed, batch speed, ETA.
        """
        self.elapsed_time = time.time() - self.t_begin
        self.speed_epoch = self.elapsed_time / (self.epoch + 1)
        self.speed_batch = self.speed_epoch / batch_count
        self.eta = self.speed_epoch * self.epochs_count - self.elapsed_time

        # increase epoch count
        self.epoch += 1

        return self.elapsed_time, self.speed_epoch, self.speed_batch, self.eta

    def print_epoch_stats(self) -> None:
        """
        Print time info.
        """
        print("Elapsed {:.2f}s, {:.2f} s/epoch, {:.2f} s/batch, eta {:.2f}s\n".format(
            self.elapsed_time, self.speed_epoch, self.speed_batch, self.eta))

    def log_epoch_stats_to_tensorboard(self, tb_writer) -> None:
        """
        Log times to tensorboard.

        Args:
            tb_writer (SummaryWriter): Tensorboard summary writer.
        """
        tb_writer.add_scalar('Time/elapsed_time',
                             self.elapsed_time, self.epoch)
        tb_writer.add_scalar('Time/speed_epoch', self.speed_epoch, self.epoch)
        tb_writer.add_scalar('Time/speed_batch', self.speed_batch, self.epoch)
        tb_writer.add_scalar('Time/eta', self.eta, self.epoch)
