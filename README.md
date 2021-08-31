# BANet Monocular Depth Prediction

This project provides a deep-learning based method to perform monocular depth prediction on RGB images. It is a custom 
PyTorch implementation of the "Bidirectional Attention Network for Monocular Depth Estimation" paper described in:

> ### Bidirectional Attention Network for Monocular Depth Estimation
> Shubhra Aich, Jean Marie Uwabeza Vianney, Md Amirul Islam, Mannat Kaur, and Bingbing Liu.
> ([arXiv PDF](https://arxiv.org/abs/2009.00743)).

![BANet Monocular Depth Prediction](res/depth_prediction.gif "BANet Monocular Depth Prediction")

## License

This code is for non-commercial use; please see the [license file](LICENSE) for terms.

## Setup

### Dependencies

This software depends on the following Python packages:

```
opencv-python
opencv-contrib-python
tensorboard
matplotlib
progressbar2
pandas

torch==1.8.0
torchvision==0.9.0
```

All can be installed using `pip install`. The PyTorch version used for development was 1.8; it can be installed 
following instructions [here](https://pytorch.org/get-started/previous-versions/).

CUDA is recommended for best performances. Version 11.2 was used during development.

### Pretrained model

An evaluation model has been trained on the [KITTI dataset](http://www.cvlibs.net/datasets/kitti/). To quickly test the 
software or to fine-tune this model, feel free to download it from [this link](https://drive.google.com/file/d/15AelkISUJM5tmwhTQjqpoW_vDVSzLgqn/view?usp=sharing).

## Usage

### Quick test

To quickly try out the code:
- Download the PyTorch model from [this link](https://drive.google.com/file/d/15AelkISUJM5tmwhTQjqpoW_vDVSzLgqn/view?usp=sharing) 
and place it under the root of the cloned repository. 
- Execute the command below:
  ```bash
  python3 main.py --inference samples/test.png --height 128 --width 416 --model kitty_banet.pt
  ```

### Training

The code has been conveniently designed such that performing a training on a custom dataset is relatively 
easy. The instructions are:

- Create two CSV files `train.csv` and `val.csv`. Each CSV must have two columns `raw_path, depth_path` and each row
respectively provides the full paths to RGB images and depth-map images / NumPy arrays. Examples of CSV files can be found in the 
`samples` directory.
- The depth-maps should be formatted either as grayscale images (higher intensity pixels corresponds to higher depths, 
common image formats are supported) or as [saved NumPy arrays](https://numpy.org/doc/stable/reference/generated/numpy.save.html?highlight=save#numpy.save)
with float32 values (`.npy` or `.npz`).
- Training is launched with the following command (note that `--width` and `--height` values must be multiple of **32**):
  ```bash
  python3 main.py --train 1 --height 192 --width 256 --train_csv train.csv --val_csv val.csv
  ```
- Optionally, the `--augmentation 1` argument can be added to perform data augmentation on the training set
(random rotation, random flip, random crop and random color jitter).
- Optionally, a pre-trained model can be passed with `--model model.pt` to fine-tune an existing model.
- Optionally, the `--tensorboard_logs tb_logs` argument can be used to log training statistics to tensorboard (a 
`tb_logs` directory will be created).

Hyper-parameters can be adjusted directly from the `parameters_config.py` file.

### Argument list

The full list of options can also be retrieved with `python3 main.py -h`.

| Argument            | Type    | Default | Description |
|---------------------|---------|---------|-------------|
| `height`            | int     | `192`   | Input height of the image to be passed to the network. Images and depth-maps are resized to this height. Must be a multiple of **32**. |
| `width`         | int     | `256`   | Input width of the image to be passed to the network. Images and depth-maps are resized to this width. Must be a multiple of **32**. |
| `cuda`              | bool    | `1`     | Activate CUDA. Set to `0` to run the code on CPU only. |
| `train_csv`         | str     | Empty   | Path to a CSV file pointing to training data. See the _Training_ section above. |
| `val_csv`           | str     | Empty   | Path to a CSV file pointing to validation data. See the _Training_ section above. |
| `train`             | bool    | `0`     | Enable training if set to `1`. |
| `inference`         | str     | Empty   | Path of an image to perform inference on. Alternatively, set to `random` to perform random inferences on the `val_csv` dataset. |
| `check_dataset`     | bool    | `0`     | Check your training set (see random RGB images and depth-maps from the set). |
| `augmentation`      | bool    | `0`     | Apply data augmentation to the training set (random rotation, random flip, random crop and random color jitter). |
| `model`             | str     | Empty   | Path to an existing PyTorch model file. Useful for fine-tuning, performing inference, or ONNX / TorchScript conversion. |
| `tensorboard_logs`  | str     | Empty   | Name of a directory used to write tensorboard logs. Tensorboard logging is disabled if empty. |
| `sleep_after_epoch` | double  | `0.0`   | Number of seconds of sleep after each epoch. |
| `onnx`              | str     | Empty   | Export `model` to ONNX. Provide desired output filename. |
| `jit`               | str     | Empty   | Export `model` to TorchScript file. Provide desired output filename. |

## Contact

Please contact us at [denis.girard@enlens.net](mailto:denis.girard@enlens.net) for any request.