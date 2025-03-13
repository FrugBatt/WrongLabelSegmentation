from torchvision.utils import draw_segmentation_masks, save_image, draw_bounding_boxes, make_grid
from einops import rearrange

def save_masked_image(img, bmask, path='masked.png', both=True):
    output = draw_segmentation_masks(img, bmask)
    if both :
        output = make_grid([img, output])
    save_image(output, path)

def save_boxed_image(img, boxes, path='boxed.png', both=True):
    output = draw_bounding_boxes(img, boxes)
    if both :
        output = make_grid([img, output])
    save_image(output, path)
