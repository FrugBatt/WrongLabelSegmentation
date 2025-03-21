import torch
import os
from PIL import Image
import torchvision.transforms as T
import pandas as pd
from tqdm import tqdm
from einops import rearrange
import shutil
from utils import alphanum_sort, imask_to_bmask, mask_to_yolo_annotation, bmask_to_imask
from torchvision.ops import masks_to_boxes

train_labels = 'data/label_Hnl61pT.csv'

def save_predictions(predictions, path = 'predictions.csv') :
    ipred = torch.stack([bmask_to_imask(pred) for pred in predictions])
    pred_np = ipred.detach().numpy()
    pd.DataFrame(pred_np.reshape((pred_np.shape[0], -1))).T.to_csv(path)
    print(f'Predictions saved to {path}')

class RaidiumDataset(torch.utils.data.Dataset):
    def __init__(self, train = True, keep_rgb = False, bounding_boxes = False):
        self.train = train
        self.keep_rgb = keep_rgb
        self.bounding_boxes = bounding_boxes

        self.data_dir = 'data/train-images' if train else 'data/test-images'
        self.data_files = alphanum_sort([f for f in os.listdir(self.data_dir) if self.is_valid_file(f)])
        self.data_width, self.data_height = 256, 256

        self.transform_loader = T.Compose([
            T.Resize((self.data_width, self.data_height)),
            T.ToTensor()
        ])

        self.images, self.labels = self.load_data()

    def is_valid_file(self, file):
        return file.endswith('.png')

    def validation_set(self):
        self.data_files = self.data_files[:len(self.data_files) // 10]
        self.images = self.images[:len(self.images) // 10]
        self.labels = self.labels[:len(self.labels) // 10]
        print('Validation set created')
    
    def train_set(self):
        self.data_files = self.data_files[len(self.data_files) // 10:]
        self.images = self.images[len(self.images) // 10:]
        self.labels = self.labels[len(self.labels) // 10:]
        print('Train set created')

    def load_data(self):
        imgs = torch.empty((len(self.data_files), 1 if not self.keep_rgb else 3, self.data_width, self.data_height)) # Grayscale images

        for i, file in tqdm(enumerate(self.data_files), desc='Loading images'):
            with open(os.path.join(self.data_dir, file), 'rb') as f:
                img = Image.open(f).convert('L' if not self.keep_rgb else 'RGB')
                img = self.transform_loader(img)
                imgs[i] = img

        if self.train: # We load the labels only if we are in the training phase
            # labels = torch.empty((len(self.data_files), self.data_width, self.data_height), dtype=torch.uint8) # Labeled images
            raw_labels = pd.read_csv(train_labels, index_col=0, header=0).T.values
            labels = torch.tensor(rearrange(raw_labels, 'b (w h) -> b w h', w=self.data_width, h=self.data_height), dtype=torch.uint8)

            if self.bounding_boxes:
                # bounding_boxes = [masks_to_boxes(imask_to_bmask(label)) ]
                bounding_boxes = []
                boxes_label = []
                max_label = 0
                for label in tqdm(labels, desc='Computing bounding boxes'):
                    bmask = imask_to_bmask(label)
                    empty = bmask.any(dim=-1).any(dim=-1)
                    indexes = empty.nonzero().squeeze()
                    if indexes.ndim == 0:
                        indexes = indexes.unsqueeze(0)
                    if indexes.numel() == 0:
                        bounding_boxes.append(indexes)
                        boxes_label.append(indexes)
                        continue
                    boxes = masks_to_boxes(bmask[indexes])
                    bounding_boxes.append(boxes)
                    boxes_label.append(indexes)
                
                self.boxes_label = boxes_label
                for l in boxes_label:
                    if l.numel() == 0:
                        continue
                    max_label = max(max_label, l.max().item())
                self.max_label = max_label
                return imgs, bounding_boxes

            return imgs, labels

        return imgs, None

    def only_labeled(self) :
        if self.bounding_boxes :
            idx = [i for i in range(len(self.data_files)) if self.labels[i].numel() != 0]
            self.boxes_label = [self.boxes_label[i] for i in idx]
        else :
            idx = [i for i in range(len(self.data_files)) if self.labels[i].sum() > 0]

        self.images = self.images[idx]
        self.labels = [self.labels[i] for i in idx]
        self.data_files = [self.data_files[i] for i in idx]
        print('Dataset cleaned')

    def only_coherent_label(self) :
        with open('data/annotated_labels.json', 'r') as f:
            labels = eval(f.read())
            self.coherent_labels = labels[:len(self.data_files)]
    
    def pixel_to_yolo_bbox(self, box) :
        x_min = box[0].item()
        y_min = box[1].item()
        x_max = box[2].item()
        y_max = box[3].item()

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        return (x_center / self.data_width, y_center / self.data_height, width / self.data_width, height / self.data_height)

    def save_yolo_format(self, path, val_split = 0.1) :
        if os.path.exists(path) :
            print('Path already exists')
            return
        train_dir = os.path.join(path, 'train')
        val_dir = os.path.join(path, 'val')
        os.makedirs(os.path.join(train_dir, 'images'))
        os.makedirs(os.path.join(train_dir, 'labels'))
        os.makedirs(os.path.join(val_dir, 'images'))
        os.makedirs(os.path.join(val_dir, 'labels'))

        if self.bounding_boxes :
            id_labels = [ [[self.boxes_label[i][j].item()] + list(self.pixel_to_yolo_bbox(x)) for j,x in enumerate(b)] for i,b in enumerate(self.labels)]
        else :
            id_labels = [ mask_to_yolo_annotation(imask_to_bmask(b)) for b in self.labels]

        for i,f in tqdm(enumerate(self.data_files), desc='Saving trainset') :
            if i >= len(self.data_files) * val_split :
                shutil.copy(os.path.join(self.data_dir, f), os.path.join(train_dir, 'images', f))
                if self.bounding_boxes :
                    df = pd.DataFrame(id_labels[i])
                    df.to_csv(os.path.join(train_dir, 'labels', f.replace('.png', '.txt')), header=False, index=False, sep=' ')
                else:
                    with open(os.path.join(train_dir, 'labels', f.replace('.png', '.txt')), 'w') as f :
                        f.write('\n'.join(id_labels[i]))
                

        for i,f in tqdm(enumerate(self.data_files), desc='Saving valset') :
            if i < len(self.data_files) * val_split :
                shutil.copy(os.path.join(self.data_dir, f), os.path.join(val_dir, 'images', f))
                if self.bounding_boxes :
                    df = pd.DataFrame(id_labels[i])
                    df.to_csv(os.path.join(val_dir, 'labels', f.replace('.png', '.txt')), header=False, index=False, sep=' ')
                else:
                    with open(os.path.join(val_dir, 'labels', f.replace('.png', '.txt')), 'w') as f :
                        f.write('\n'.join(id_labels[i]))

        names = 'names:\n'
        for i in range(self.max_label+1) :
            names += f'  {i}: struct_{i}\n'
        with open(os.path.join(path, 'raidium.yaml'), 'w') as f :
            f.write(f'path: ../{path}\ntrain: train\nval: val\n{names}')

        print('Yolo format saved')
        
    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        if self.train:

            if self.bounding_boxes:
                return self.images[idx], self.labels[idx]

            return self.images[idx], imask_to_bmask(self.labels[idx], annotated_labels=self.coherent_labels[idx])

        return self.images[idx]

def collate_fn(batch):
    if batch == []:
        return []
    if len(batch[0]) == 2:
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = list(labels) # lists of binary mask or list of bounding boxes
        return images, labels

    return torch.stack(batch) # images
