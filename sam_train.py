import torch
from data import RaidiumDataset, collate_fn
from model import SegmentationModel
from criterion import DiceLoss

import argparse


# argparse
parser = argparse.ArgumentParser(description='Train SAM model')
parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device')
parser.add_argument('--yolo_weights', type=str, default='yolo11x.pt', help='weights')

args = parser.parse_args()

dataset = RaidiumDataset(train=True, keep_rgb=True, bounding_boxes=True)
# dataset.only_labeled()

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

model = SegmentationModel(args.yolo_weights)
model.to(args.device)

criterion = DiceLoss()
optimizer = torch.optim.Adam(model.sam_model.parameters(), lr=0.001)

for epoch in range(args.epochs):
    for i, (images, labels) in enumerate(dataloader):
        # print(images.shape, len(labels), labels[0].shape)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch}, Iteration {i}, Loss {loss.item()}')
