import torch
import torch.nn as nn

#from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2 import load_model

from einops import rearrange

import numpy as np

from ultralytics import YOLO


class SegmentationModel(nn.Module) :

    def __init__(self, yolo_path):
        super(SegmentationModel, self).__init__()

        self.yolo_model = YOLODetectModel(yolo_path=yolo_path)
        self.sam_model = self.load_sam() # Load the SAM model here
        self.coherence_model = nn.Linear(1, self.sam_model.model.sam_prompt_encoder.embed_dim)

    def load_sam(self):
        print('Loading SAM model')

        #need to download weights before running
        checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        #predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))
        predictor = SAM2ImagePredictor(load_model(variant='large', ckpt_path='checkpoints/sam2_hiera_large.pt', device='cpu'))

        return predictor

    def forward(self, imgs, coherence = None, return_logits=False):
        _, boxes = self.yolo_model(imgs.unsqueeze(0))

        #img = self.preprocess(img)
        if coherence is None :
            coherence = torch.ones(1)

        
        c_emb = self.coherence_model(coherence)

        #c_emb = torch.zeros(256)
        c_emb = c_emb.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(boxes[0].shape[0], -1, -1, -1)

        #np_imgs = [np.array(rearrange(img, 'c h w -> h w c')) for img in imgs]

        #self.sam_model.set_image_batch(np_imgs)

        #masks, _, _ = self.sam_model.predict_batch(
        #    box_batch=boxes,
        #    coherence_emb_batch=c_embs,
        #    multimask_output=False
        #)

        #print(boxes[0].shape)
        #print(len(boxes))

        img = np.array(rearrange(imgs, 'c h w -> h w c'))
        self.sam_model.set_image(img)
        masks, _, _ = self.sam_model.predict(box=boxes[0], coherence_emb=c_emb, multimask_output=False, return_logits=return_logits)
        #print(boxes[0])
        #print(self.sam_model._prep_prompts(None, None, boxes[0], None, True))
        #mask_input, unnorm_coords, labels, unnorm_box = self.sam_model._prep_prompts(None, None, boxes[0], None, True)
        #if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0 or labels.shape[0] == 0:
        #    exit()

        #sparse_embeddings, dense_embeddings = self.sam_model.model.sam_prompt_encoder(
        #    points=(unnorm_coords, labels), boxes=None, masks=None,
        #)
        #print(sparse_embeddings.shape, dense_embeddings.shape)

        #print(masks, masks.requires_grad)
        #masks = [torch.tensor(mask).squeeze() for mask in masks]
        #print(masks.shape)
        return masks

class YOLODetectModel(nn.Module) :

    def __init__(self, yolo_path='yolo11n.pt'):
        super(YOLODetectModel, self).__init__()

        self.yolo_model = YOLO(yolo_path)

    def forward(self, img):
        output = self.yolo_model.predict(source=img, verbose=False)

        boxes = [o.boxes.data[:,0:4] for o in output]
        probs = [o.boxes.conf for o in output]

        return probs, boxes
