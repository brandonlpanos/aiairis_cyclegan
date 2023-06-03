import torch
import numpy as np
from skimage import exposure

# Define custom transforms

def min_max_normalization(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor

class ClipOutliers(torch.nn.Module):
        def __init__(self, outlier_cut=2):
                super().__init__()
                self.outlier_cut = outlier_cut
            
        def forward(self, tensor):
                "Forward pass"
                mean = torch.mean(tensor)
                std = torch.std(tensor)
                lower_bound = mean - self.outlier_cut * std
                upper_bound = mean + self.outlier_cut * std
                clipped_image = torch.clamp(tensor, lower_bound, upper_bound)
                return clipped_image