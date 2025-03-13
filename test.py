from model import SegmentationModel, YOLODetectCoherenceModel , WatershedCoherenceModel
from data import RaidiumDataset
from PIL import Image
from utils import imask_to_bmask, mask_to_yolo_annotation
from plot import save_masked_image, save_boxed_image
from ultralytics import YOLO
import torchvision.transforms as T

model = SegmentationModel()
# model.to('cuda')

# model = SegmentationModel(1)

dataset = RaidiumDataset(train=True, keep_rgb=True, bounding_boxes=False)
dataset.only_labeled()
# dataset.save_yolo_format('data/yolo_masks')

# print(mask_to_yolo_annotation(dataset[0][1]))
# print(dataset.labels[0])

# model = YOLO('yolo11n-seg.pt')
#
# img = T.Resize((1024, 1024))(dataset[14][0])
#
output = model(dataset[14][0])
#
# print(output[0].masks)

# bmask = imask_to_bmask(dataset.labels[0])
save_masked_image(dataset[0][0], output)
# print(output[0].shape)

# save_boxed_image(dataset.images[0], dataset.labels[0])

# model.sam_model.train(data="data/yolo_masks/raidium.yaml", epochs=5, imgsz=256)
