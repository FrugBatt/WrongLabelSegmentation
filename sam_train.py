import torch
from data import RaidiumDataset, collate_fn
from model import SegmentationModel
from coherence import YOLODetectCoherenceModel
from criterion import DiceLoss

import argparse

import matplotlib.pyplot as plt


# argparse
parser = argparse.ArgumentParser(description='Train SAM model')
parser.add_argument('--epochs', type=int, default=5, help='number of epochs')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='device')
parser.add_argument('--yolo_weights', type=str, default='yolo11x.pt', help='weights')
parser.add_argument('--output', type=str, default='sam_weights.pt', help='output')

args = parser.parse_args()

train_dataset = RaidiumDataset(train=True, keep_rgb=True, bounding_boxes=False)
train_dataset.only_labeled()
train_dataset.train_set()

val_dataset = RaidiumDataset(train=True, keep_rgb=True, bounding_boxes=False)
val_dataset.only_labeled()
val_dataset.validation_set()

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

model = SegmentationModel(args.yolo_weights)
model.to(args.device)

coherence = YOLODetectCoherenceModel(args.yolo_weights)
#coherence.to(args.device)

model.sam_model.model.sam_mask_decoder.train(True)
model.sam_model.model.sam_prompt_encoder.train(True)

criterion = DiceLoss()
optimizer = torch.optim.Adam(list(model.sam_model.model.parameters()) + list(model.coherence_model.parameters()), lr=0.001)
train_losses = []
val_losses = []

best_val_loss = float('inf')

for epoch in range(args.epochs):
    c_loss = 0
    for i, (images, labels) in enumerate(train_dataloader):
        coherence_score = coherence(images[0], labels)

        optimizer.zero_grad()
        outputs = model(images[0], coherence=coherence_score, return_logits=True)
        outputs_sg = torch.sigmoid(outputs)
        
        if outputs_sg.ndim == 2:
            outputs_sg = outputs_sg.unsqueeze(0)

        loss = criterion(outputs_sg, labels[0])
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch}, Iteration {i}, Loss {loss.item()}')
        c_loss += loss.item()
    print('***************************************')
    print(f'End of Epoch {epoch}, Training Loss {c_loss / len(train_dataloader)}')
    train_losses.append(c_loss / len(train_dataloader))

    val_loss = 0
    for i, (images, labels) in enumerate(val_dataloader):
        outputs = model(images[0], return_logits=True)
        outputs_sg = torch.sigmoid(outputs)
        
        if outputs_sg.ndim == 2:
            outputs_sg = outputs_sg.unsqueeze(0)

        loss = criterion(outputs_sg, labels[0])
        val_loss += loss.item()
    val_losses.append(val_loss / len(val_dataloader))
    print(f'End of Epoch {epoch}, Validation Loss {val_loss / len(val_dataloader)}')
    print('***************************************')
    print('***************************************')

    if val_loss < best_val_loss:
        torch.save(model.state_dict(), args.output)
        best_val_loss = val_loss
        print('Model saved')


plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()

plt.savefig(f'loss_{args.output}.png')