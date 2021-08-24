# Copyright Enlens LLC, 2021. All rights reserved.
#
# This software is licensed under the terms of the "BANet Monocular Depth Prediction" licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# Author: Denis Girard (denis.girard@enlens.net)

import os
import random
import argparse

import cv2
import torch
import torch.optim as optim

from torch.utils.mobile_optimizer import optimize_for_mobile
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from system_config import SystemConfig, setup_system
from parameters_config import ParametersConfig
from model_banet import BANet_DenseNet121
from trainer import Trainer
from inference import Inference
from utils import display_img_and_target, get_params_count, load_model, str2bool, apply_image_transform
from datasets import MyDataset
from loss import silog_loss
from metrics import SILogMetrics


# Main
def main():
    """
    Perform training or inference on a BANet network (with PyTorch).
    """

    # parse input arguments
    parser = argparse.ArgumentParser(description='Lightweight depth prediction model')
    parser.add_argument('--height', type=int, default=192,
                        help='Desired height of the input image. Should be a multiple of 32.')
    parser.add_argument('--width', type=int, default=256,
                        help='Desired width of the input image. Should be a multiple of 32.')
    parser.add_argument('--cuda', type=str2bool, default=True, help='Set to 1 to use CUDA')
    parser.add_argument('--train_csv', type=str, default='', help='Path to the training CSV')
    parser.add_argument('--val_csv', type=str, default='', help='Path to the validation CSV')
    parser.add_argument('--train', type=str2bool, default=False, help='Perform training')
    parser.add_argument('--check_dataset', type=str2bool, default=False,
                        help='Set to 1 to perform a visual check of the train dataset')
    parser.add_argument('--augmentation', type=str2bool, default=False,
                        help='Set to 1 to perform data augmentation during training')
    parser.add_argument('--architecture', type=str, default='banet',
                        choices=['banet'],
                        help='The architecture to use. Default is BANet')
    parser.add_argument('--model', type=str, default='',
                        help='PyTorch model file to load. Is used for fine-tuning if specified during training.')
    parser.add_argument('--inference', type=str, default='',
                        help='Perform inference on the selected image, using weights of "--model"')
    parser.add_argument('--tensorboard_logs', type=str, default='',
                        help='Name of tensorboard output logs dir.')
    parser.add_argument('--sleep_after_epoch', type=float, default=0.0,
                        help='Idle time after each epoch, allowing the machine to cool down.')
    parser.add_argument('--onnx', type=str, default='', help='Export model to ONNX.')
    parser.add_argument('--jit', type=str, default='', help='Export model to JIT torchscript file.')
    args = parser.parse_args()
    print(args)

    # update selected device
    if args.cuda:
        SystemConfig.device = 'cuda'
    else:
        SystemConfig.device = 'cpu'

    # update input size (w, h)
    ParametersConfig.input_size = (args.width, args.height)

    # setup system
    setup_system(SystemConfig)

    # load dataset
    train_dataset = None
    val_dataset = None

    # train dataset
    if args.train_csv != '':
        train_dataset = MyDataset(
            input_csv_path=args.train_csv,
            image_size=ParametersConfig.input_size,
            data_augmentation=args.augmentation
        )

    # validation dataset
    if args.val_csv != '':
        val_dataset = MyDataset(
            input_csv_path=args.val_csv,
            image_size=ParametersConfig.input_size,
            data_augmentation=False
        )

    # display random images from train dataset
    if args.check_dataset:

        # check that a train dataset has been selected
        if args.train_csv == '':
            raise Exception("Please select a CSV file pointing to a training dataset (--train_csv).")

        # display random images
        for i in range(50):
            idx = random.randrange(0, len(train_dataset))
            img, depth = train_dataset.get_not_augmented(idx)
            display_img_and_target(img, depth)

    # create model instance
    model = None
    if args.architecture == 'banet':
        model = BANet_DenseNet121(image_size=(ParametersConfig.input_size[1], ParametersConfig.input_size[0]))

    # load model
    if args.model != '':
        load_model(model, args.model)

    # train
    if args.train:
        # optimizer
        optimizer = optim.Adam(
            model.parameters(),
            lr=ParametersConfig.learning_rate,
            weight_decay=ParametersConfig.weight_decay
        )

        # lr scheduler
        scheduler = MultiStepLR(
            optimizer,
            milestones=ParametersConfig.lr_step_milestones,
            gamma=ParametersConfig.lr_gamma
        )

        # loss function
        loss_function = silog_loss

        # metrics
        metrics = SILogMetrics()

        # tensorboard summary writer
        tb_writer = None
        if args.tensorboard_logs != '':
            tb_writer = SummaryWriter(args.tensorboard_logs)

        # sleep after epoch
        sleep_after_epoch = args.sleep_after_epoch

        # check that a train and a validation dataset have been selected
        if args.train_csv == '':
            raise Exception("Please select a CSV file pointing to a training dataset (--train_csv).")
        if args.val_csv == '':
            raise Exception("Please select a CSV file pointing to a validation dataset (--val_csv).")

        # create trainer
        trainer = Trainer(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model=model,
            loss_function=loss_function,
            metrics=metrics,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            tb_writer=tb_writer,
            sleep_after_epoch=sleep_after_epoch)

        # start training
        trainer.start_training()

    # perform random inference on the validation dataset
    if args.inference == 'random':

        # check that a model has been loaded
        if args.model == '':
            raise Exception("Please load model weights (--model) before performing inference.")

        # check that a validation dataset is provided
        if args.val_csv == '':
            raise Exception("Please select a CSV file pointing to a validation dataset (--val_csv).")

        # prepare inference and load model
        inference = Inference(model, SystemConfig.device)

        # inference on random images from val dataset
        for i in range(50):
            # select a random image
            idx = random.randrange(0, len(val_dataset))

            # retrieve normalized image and original rescaled image
            img, depth = val_dataset[idx]
            img_original, depth_original = val_dataset.get_not_augmented(idx)

            # perform prediction
            prediction = inference.predict_single(img)
            duration = inference.get_last_prediction_time()

            # print last elapsed time
            print("Inference time: {:.4f} sec".format(duration))

            # display
            display_img_and_target(img_original, depth_original, prediction)

    # perform inference on the whole validation dataset. Measure loss, accuracy and speed.
    elif args.inference == 'validation':

        # check that a model has been loaded
        if args.model == '':
            raise Exception("Please load model weights (--model) before performing inference.")

        # check that a validation dataset is provided
        if args.val_csv == '':
            raise Exception("Please select a CSV file pointing to a validation dataset (--val_csv).")

        # prepare inference and load model
        inference = Inference(model, SystemConfig.device)

        # evaluate model on validation dataset
        mean_loss, metric, mean_duration = inference.evaluate_on_dataset(val_dataset)

        # print results
        print("\n\n--- Performance evaluation for model {} (backbone {}) on device {} ---".format(args.model,
                                                                                                  args.backbone,
                                                                                                  SystemConfig.device))
        print("Average Loss           = {:.4f}".format(mean_loss))
        print("Average inference time = {:.6f} s".format(mean_duration))
        print("Metric                 = {:.4f}".format(metric))
        print("Model size             = {}".format(get_params_count(model)))

    # perform inference on the whole validation dataset. Measure loss, accuracy and speed.
    elif args.inference != '':

        # check that a model has been loaded
        if args.model == '':
            raise Exception("Please load model weights (--model) before performing inference.")

        # prepare inference and load model
        inference = Inference(model, SystemConfig.device)

        # open image
        img = cv2.imread(args.inference)

        # check that image is not empty
        if img is None:
            raise Exception("Couldn't open image at {}.".format(args.inference))

        # prepare image
        img_tensor = apply_image_transform(img)

        # perform prediction
        pred = inference.predict_single(img_tensor)

        # display
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        display_img_and_target(img, pred)

    if args.onnx:

        # check that a model has been loaded
        if args.model == '':
            raise Exception("Please load model weights (--model) before performing ONNX conversion.")

        # model in eval mode
        model.eval()

        # dummy input
        x = torch.randn(1, 3, ParametersConfig.input_size[1], ParametersConfig.input_size[0])

        # save path
        model_path = os.path.join('models', args.onnx)

        # create models directory
        if not os.path.exists('models'):
            os.makedirs('models')

        # export ONNX
        torch.onnx.export(model, x, model_path, verbose=True, input_names=["input"], output_names=["output"])

    if args.jit:

        # check that a model has been loaded
        if args.model == '':
            raise Exception("Please load model weights (--model) before performing TorchScript conversion.")

        # model in eval mode
        model.eval()

        # save path
        model_path_jit = os.path.join('models', args.jit)

        # create models directory
        if not os.path.exists('models'):
            os.makedirs('models')

        # export ONNX
        traced_script_module = torch.jit.script(model)
        traced_script_module = optimize_for_mobile(traced_script_module)
        traced_script_module.save(model_path_jit)


# program entrypoint
if __name__ == "__main__":
    main()
