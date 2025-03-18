from model import SegmentationModel
from data import RaidiumDataset, save_predictions
from PIL import Image
from utils import imask_to_bmask, mask_to_yolo_annotation
from plot import save_masked_image, save_boxed_image
from ultralytics import YOLO
import torchvision.transforms as T
import torch


# model = SegmentationModel('best.pt')
# model.to('cuda')

# model = SegmentationModel(1)

dataset = RaidiumDataset(train=True, keep_rgb=True, bounding_boxes=True)
dataset.only_labeled()
dataset.only_coherent_label()
dataset.save_yolo_format('data/yolo_bboxes')
print(dataset.max_label)

print(dataset[0][1].shape)

# dataset.only_labeled()
# dataset.save_yolo_format('data/yolo_masks')

# print(mask_to_yolo_annotation(dataset[0][1]))
# print(dataset.labels[0])

# model = YOLO('yolo11n-seg.pt')
#
# img = T.Resize((1024, 1024))(dataset[14][0])
#


preds = []
for i in range(500):
    img = dataset[i]
    pred = model(img)
    preds.append(pred)
    if i % 100 == 1:
        save_masked_image(img, pred, f'output_{i}.png')
save_predictions(preds, 'output.csv')

#imgs = torch.stack([img1, img2])
#boxes = [box1, box2]

# save_boxed_image(img1, box1, 'img1.png')
#save_boxed_image(img2, box2, 'img2.png')

#print(img.shape)
output1 = model(img1)
output2 = model(img2)

save_predictions([output1, output2], 'output.csv')

save_masked_image(img2, output2, 'output1.png')
#save_masked_image(img2, output[1], 'output2.png')

#
# print(output[0].masks)
# bmask = imask_to_bmask(dataset.labels[0])
#save_masked_image(img, output.squeeze())
# print(output[0].shape)

# save_boxed_image(dataset.images[0], dataset.labels[0])

# model.sam_model.train(data="data/yolo_masks/raidium.yaml", epochs=5, imgsz=256)
