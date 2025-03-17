from model import SegmentationModel
from data import RaidiumDataset
from PIL import Image
from utils import imask_to_bmask, mask_to_yolo_annotation
from plot import save_masked_image, save_boxed_image
from ultralytics import YOLO
import torchvision.transforms as T
import torch

model = SegmentationModel('runs/detect/train11/weights/best.pt')
# model.to('cuda')

# model = SegmentationModel(1)

dataset = RaidiumDataset(train=True, keep_rgb=True, bounding_boxes=True)
dataset.only_labeled()
# dataset.save_yolo_format('data/yolo_masks')

# print(mask_to_yolo_annotation(dataset[0][1]))
# print(dataset.labels[0])

# model = YOLO('yolo11n-seg.pt')
#
# img = T.Resize((1024, 1024))(dataset[14][0])
#

img1 = dataset[0][0]
box1 = dataset[0][1]

#imgs = torch.stack([img1, img2])
#boxes = [box1, box2]

save_boxed_image(img1, box1, 'img1.png')
#save_boxed_image(img2, box2, 'img2.png')

#print(img.shape)
output = model(img1, boxes=[box1])

save_masked_image(img1, output, 'output1.png')
#save_masked_image(img2, output[1], 'output2.png')

#
# print(output[0].masks)
# bmask = imask_to_bmask(dataset.labels[0])
#save_masked_image(img, output.squeeze())
# print(output[0].shape)

# save_boxed_image(dataset.images[0], dataset.labels[0])

# model.sam_model.train(data="data/yolo_masks/raidium.yaml", epochs=5, imgsz=256)
