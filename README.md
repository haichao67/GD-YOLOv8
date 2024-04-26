# GD-YOLOv8
This repository contains the code implementation for the experiments described in the paper "Lightweight Rail Surface Defect Detection Algorithm Based on an Improved YOLOv8".

## Model
The project files is modified based on the YOLOv8 project. The model configuration file [gd-yolov8.yaml](https://github.com/haichao67/GD-YOLOv8/blob/main/ultralytics/cfg/models/v8/gd-yolov8.yaml) is located in the directory [./ultralytics/cfg/models/v8](https://github.com/haichao67/GD-YOLOv8/tree/main/ultralytics/cfg/models/v8).

## Recommended Configuration
- Python: 3.8
- Torch: 1.13.1
- Torchvision: 0.14.1

## Additional Packages Required
Install packages by yourself if they are not already installed
Recommended dependencies:
pip install timm thop efficientnet_pytorch einops grad-cam dill

## Training
Run the [train.py](https://github.com/haichao67/GD-YOLOv8/blob/main/train.py) file. The default hyperparameters are located in the [./ultralytics/cfg/default.yaml](https://github.com/haichao67/GD-YOLOv8/blob/main/ultralytics/cfg/default.yaml) file.
**Note**: Please modify the model path and [data.yaml](https://github.com/haichao67/GD-YOLOv8/blob/main/dataset/data.yaml) file path in [train.py](https://github.com/haichao67/GD-YOLOv8/blob/main/train.py) as needed. The default location for [data.yaml](https://github.com/haichao67/GD-YOLOv8/blob/main/dataset/data.yaml) is [./dataset](https://github.com/haichao67/GD-YOLOv8/tree/main/dataset). The dataset format is the same as that used in the YOLOv8 project, so be sure to modify [data.yaml](https://github.com/haichao67/GD-YOLOv8/blob/main/dataset/data.yaml) in [./dataset](https://github.com/haichao67/GD-YOLOv8/tree/main/dataset) to point to your dataset location.

## Validation
Run the [val.py](https://github.com/haichao67/GD-YOLOv8/blob/main/val.py) file. The modification method is consistent with training.

## Inference on Images
Run the [detect.py](https://github.com/haichao67/GD-YOLOv8/blob/main/detect.py) file. The modification method is consistent with training.

## Evaluation
We have trained and evaluated GD-YOLOv8 on the RSDDs dataset, which is included in the project. The trained model weights file is located at [./weight/gd-yolo](https://github.com/haichao67/GD-YOLOv8/tree/main/weight/gd-yolo).

## Ablation Experiments
For the ablation experiments described in the paper, the model weights files are located in the [./weight/ablation](https://github.com/haichao67/GD-YOLOv8/tree/main/weight/ablation) folder.
