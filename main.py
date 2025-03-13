import torch
from data import RaidiumDataset, save_predictions
from model import SegmentationModel 
from criterion import DiceLoss
import argparse

parser = argparse.ArgumentParser(description='Raidium Challenge')
parser.add_argument('--test', action='store_true', help='Test the model')
parser.add_argument('--model', type=str, default=None, help='Model file to test')
parser.add_argument('--output', type=str, default='output.csv', help='Output file')
parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')


args = parser.parse_args()

dataset = RaidiumDataset(train=not args.test)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

model = SegmentationModel(args.model)

output = model(dataset.images)
save_predictions(output, args.output)
