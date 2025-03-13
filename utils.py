import torch
import re
import cv2
import numpy as np

def alphanum_sort(l) :
    """
    Sorts a list of strings in a natural order
    """
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def imask_to_bmask(imask, dtype = torch.bool) :
    """
    Converts an int mask to a binary mask
    imask : (W, H) tensor
    """
    cl = imask.unique()[1:] # We exclude the background class
    bmask = torch.zeros((cl.shape[0], *imask.shape), dtype = dtype)
    for i,c in enumerate(cl) :
        bmask[i] = imask == c
    return bmask

def bmask_to_imask(bmask) :
    """
    Converts a binary mask to an int mask
    bmask : (C, W, H) tensor
    """
    c, w, h = bmask.shape
    imask = torch.zeros((w, h))
    for i in range(c) :
        imask += (i+1) * bmask[i]
    return imask

def mask_to_yolo_annotation(mask_tensor):
    """
    Converts a binary segmentation mask tensor (C, H, W) to YOLO format.
    
    Args:
        mask_tensor (torch.Tensor): Binary segmentation mask of shape (C, H, W)
    
    Returns:
        list: A list of strings in YOLO format, where each string represents one object instance.
    """
    C, H, W = mask_tensor.shape
    yolo_annotations = []
    
    for class_idx in range(C):
        mask = mask_tensor[class_idx].cpu().numpy().astype(np.uint8)  # Convert to NumPy array
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if len(contour) < 3:
                continue  # Ignore invalid contours (need at least 3 points for a polygon)
            
            # Normalize contour coordinates to be between 0 and 1
            normalized_contour = [(x / W, y / H) for [x, y] in contour[:, 0, :]]
            
            # Flatten the contour and format it for YOLO
            annotation = f"0 " + " ".join([f"{x:.6f} {y:.6f}" for x, y in normalized_contour])
            yolo_annotations.append(annotation)
    
    return yolo_annotations
