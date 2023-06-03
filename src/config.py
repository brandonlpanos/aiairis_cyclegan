from normalizations import *
from torchvision import transforms
from torchvision.transforms import functional as F

BATCH_SIZE = 1
NUM_EPOCHS = 200
LEARING_RATE = 0.0002 # keep for first 100 epochs then linearly decay for 100
LAMBDA_IDENTITY = 0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
ALPHA = 0.3 # times image factor

# List of transformations to apply to the input images
TRANSFORM_LIST = [
    ClipOutliers(outlier_cut=1), # Clip outliers (standard devisations)
    transforms.Lambda(min_max_normalization),  # Min-max normalization
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    ]

# List of all transformations to apply to the input images
# ------------------------------------------------------------------------------------
# CutHigh(0.95), # Cut the top 5% of pixels
# InstanceNormNorm(), # Instance normalization
# transforms.RandomCrop(1000), # Crop the image randomly to 500 x 500 pixels
# transforms.Resize((500, 500)),  # Resize the image while maintaining aspect ratio
# transforms.ToTensor(),  # Convert the numpy array to a PyTorch tensor
# transforms.Lambda(min_max_normalization),  # Min-max normalization
# transforms.Lambda(z_score_normalization),  # Z-score normalization
# transforms.Lambda(gamma_correction),  # Gamma correction
# transforms.Lambda(contrast_stretching),  # Contrast stretching
# transforms.Lambda(clip_outliers),  # Clip outliers
# transforms.Normalize((0.5,), (0.5,)),  # Normalize pixel values to the range [-1, 1]
# HistogramEqualization(), # Smooths out pix values over entire range 
# ------------------------------------------------------------------------------------