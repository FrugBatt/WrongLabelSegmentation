import torch
import torch.nn as nn

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from ultralytics import YOLO


class SegmentationModel(nn.Module) :

    def __init__(self, yolo_path):
        super(SegmentationModel, self).__init__()

        self.yolo_model = YOLODetectModel(yolo_path=yolo_path)
        self.sam_model = self.load_sam() # Load the SAM model here
        self.coherence_model = nn.Linear(1, self.sam_model.sam_prompt_emb_dim)

    def load_sam(self):
        print('Loading SAM model')

        #need to download weights before running
        checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

        return predictor

    def forward(self, img, coherence = None):
        _, boxes = self.yolo_model(img)

        img = self.preprocess(img)

        if coherence is None :
            coherence = torch.ones(img.shape[0])

        c_emb = self.coherence_model(coherence)

        self.sam_model.set_image_batch(img)

        masks, _, _ = self.sam_model.predict_batch(
            box_batch=boxes,
            coherence_batch=c_emb,
            multimask_output=False
        )

        return masks

class YOLODetectModel(nn.Module) :

    def __init__(self, yolo_path='yolo11n.pt'):
        super(YOLODetectModel, self).__init__()

        self.yolo_model = YOLO(yolo_path)

    def forward(self, img):
        output = self.yolo_model.predict(source=img)

        boxes = [o.boxes.data[:,0:4] for o in output]
        probs = [o.boxes.conf for o in output]

        return probs, boxes
