import torch
import numpy as np
from skimage import exposure

# Define custom transforms

def min_max_normalization(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor

def z_score_normalization(tensor):
    mean_val = torch.mean(tensor)
    std_val = torch.std(tensor)
    normalized_tensor = (tensor - mean_val) / std_val
    return normalized_tensor

def gamma_correction(tensor, gamma=1.0):
    image = tensor.numpy()
    corrected_image = np.power(image, gamma)
    return torch.from_numpy(corrected_image)

def contrast_stretching(tensor):
    image = tensor.numpy()
    stretched_image = exposure.rescale_intensity(image)
    return torch.from_numpy(stretched_image)


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
    
class GammaCorrection(torch.nn.Module):
    def __init__(self, gamma=0.3):
        super(GammaCorrection, self).__init__()
        self.gamma = gamma

    def forward(self, image):
        '''Forward pass'''
        # Apply gamma correction using OpenCV
        image = torch.pow(image, self.gamma)
        return image  
    
class InstanceNormNorm(torch.nn.Module):
     def __init__(self):
         super(InstanceNormNorm, self).__init__()
         self.norm = torch.nn.InstanceNorm2d(1)

     def forward(self, image):
            """Forward pass"""
            return self.norm(image)
     
class CutHigh(torch.nn.Module):
     def __init__(self, quart):
         super(CutHigh, self).__init__()
         self.quart = quart

     def forward(self, image):
            """Forward pass"""
            # Use torch.quantile() to find the 99.9th percentile of the image and replace with mean
            image[image > torch.quantile(image, self.quart)] = torch.mean(image)
            return image