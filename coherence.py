import torch
from abc import ABC, abstractmethod
import numpy as np
import cv2

from model import YOLODetectModel
from criterion import DiceLoss

class CoherenceModel(ABC) :
    """
    Coherence model
    It converts an input (an image) to a coherence value (a scalar or a coherence map)
    """

    def __init__(self, c_d):
        self.c_d = c_d

    @abstractmethod
    def __call__(self, img, seg_map):
        pass

class NoCoherenceModel(CoherenceModel) :

    def __init__(self):
        super(NoCoherenceModel, self).__init__(1)

    def __call__(self, x, _):
        return torch.ones((1))

class YOLODetectCoherenceModel(CoherenceModel) :

    def __init__(self, yolo_path):
        super(YOLODetectCoherenceModel, self).__init__(1)

        self.yolo_model = YOLODetectModel(yolo_path=yolo_path)

    def __call__(self, img, seg_map):
        probs, _ = self.yolo_model(img.unsqueeze(0))

        n_yolo = torch.tensor([p.shape[0] for p in probs])
        n_seg = torch.tensor([s.shape[0] for s in seg_map])

        return torch.clamp(torch.tensor([n_seg / n_yolo]), 0, 1)

class WatershedCoherenceModel(CoherenceModel) :

    def __init__(self):
        super(WatershedCoherenceModel, self).__init__(256*256)
        self.dice = DiceLoss()

    def watershed_feature_segmentation(self, image_tensor):
        """
        Applies the Watershed segmentation algorithm to detect multiple features within objects.
        
        Args:
            image_tensor (torch.Tensor): Image tensor of shape (C, H, W) with values in [0, 1] or [0, 255].
        
        Returns:
            torch.Tensor: Segmentation mask tensor of shape (num_classes, H, W), where each class is a separate feature.
        """
        # Convert PyTorch tensor to NumPy
        image_np = image_tensor.cpu().numpy()
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)

        # Convert from (C, H, W) to (H, W, C) for OpenCV
        image_np = np.transpose(image_np, (1, 2, 0))

        # Convert to grayscale
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

        # Use adaptive thresholding to highlight internal object structures
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 11, 2)

        # Use Canny edge detection to refine object boundaries
        edges = cv2.Canny(gray, 50, 150)

        # Combine edge and threshold information
        combined = cv2.bitwise_or(adaptive_thresh, edges)

        # Use morphological operations to enhance segmentation markers
        kernel = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Compute distance transform to separate different internal structures
        dist_transform = cv2.distanceTransform(morph, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)

        # Convert to binary format
        sure_fg = np.uint8(sure_fg)

        # Find connected components for marker labeling
        _, markers = cv2.connectedComponents(sure_fg)

        # Add 1 to markers so the background is labeled as 1
        markers += 1

        # Apply the Watershed algorithm
        markers = cv2.watershed(image_np, markers)

        # Determine the number of segmented classes (excluding background)
        num_classes = markers.max()

        # Create a binary mask for each detected feature
        masks = np.zeros((num_classes, markers.shape[0], markers.shape[1]), dtype=np.uint8)
        for i in range(1, num_classes + 1):
            masks[i - 1] = (markers == i).astype(np.uint8)

        # Convert masks to a PyTorch tensor
        masks_tensor = torch.tensor(masks, dtype=torch.bool)

        id_background = torch.argmax(masks_tensor.sum(dim=(1, 2)))

        #delete background mask
        masks_tensor = torch.cat((masks_tensor[:id_background], masks_tensor[id_background+1:]), dim=0)
        return masks_tensor

    def __call__(self, img, seg_map):
        w_bmask = self.watershed_feature_segmentation(img)

        dice = self.dice(w_bmask, seg_map[0])
        return dice.unsqueeze(0)
