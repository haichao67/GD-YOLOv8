import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import torch.nn as nn
import torch

if __name__ == '__main__':
    model = YOLO('runs/train/exp/weights/best.pt') # select your model.pt path
    model.val(data='dataset/data.yaml', # select your data.yaml path
        split='val',
        imgsz=640,
        device='0',
        batch=16,
        # rect=False,
        # save_json=True, # if you need to cal coco metrice 
        project='runs/val',
        name='exp',
        )

