import torch

class DiceLoss():

    def __init__(self, smooth = 1e-5):
        self.smooth = smooth

    def __call__(self, seg1: torch.Tensor, seg2: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
        """
        Compute the Dice loss between two binary segmentation maps.
        The segmentation maps have shape (C, H, W), and the i-th map in seg1
        does not necessarily correspond to the i-th map in seg2.
        
        Args:
            seg1 (torch.Tensor): First segmentation map (C, H, W), binary values.
            seg2 (torch.Tensor): Second segmentation map (C, H, W), binary values.
            epsilon (float): Small constant to avoid division by zero.
        
        Returns:
            torch.Tensor: Dice loss.
        """
        C1, H1, W1 = seg1.shape
        C2, H2, W2 = seg2.shape
        assert (H1, W1) == (H2, W2), "Input segmentation maps must have the same spatial dimensions."
        
        # Reshape to (C, -1) for easier computation
        seg1_flat = seg1.view(C1, -1).float()  # Shape: (C1, H*W)
        seg2_flat = seg2.view(C2, -1).float()  # Shape: (C2, H*W)
        
        # Compute intersection and union for all channel pairings (C1, C2)
        intersection = torch.matmul(seg1_flat, seg2_flat.T)  # Shape: (C1, C2)
        union = seg1_flat.sum(dim=1, keepdim=True) + seg2_flat.sum(dim=1, keepdim=True).T  # Shape: (C1, C2)
        
        dice_scores = (2.0 * intersection + epsilon) / (union + epsilon)  # Dice coefficient (C1, C2)
        
        # Find best matching for each channel
        best_dice_scores, _ = dice_scores.max(dim=1)  # shape : (C1,)
        
        # Compute Dice loss
        dice_loss = 1.0 - best_dice_scores.mean()  # Average over all channels in seg1
        
        return dice_loss
