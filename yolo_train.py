import torch
from data import RaidiumDataset
from model import YOLODetectModel
import argparse


# argparse
parser = argparse.ArgumentParser(description='Train YOLO model')
parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device')
parser.add_argument('--weights', type=str, default='yolo11x.pt', help='weights')

args = parser.parse_args()

dataset = RaidiumDataset(train=True, keep_rgb=True, bounding_boxes=True)
dataset.only_labeled()
dataset.only_coherent_label()
dataset.save_yolo_format('data/yolo_boxes')

model = YOLODetectModel(args.weights)

model.yolo_model.train(
    data="data/yolo_boxes/raidium.yaml",
    epochs=args.epochs,
    imgsz=256,
    device=args.device,
    lr0=0.0002,
    optimizer='adamw',
)
