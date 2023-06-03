from normalizations import *
from torchvision import transforms
from torchvision.transforms import functional as F

BATCH_SIZE = 1 # Batch size during training
NUM_EPOCHS = 200 # Number of epochs to train for
LEARING_RATE = 0.0002  # keep for first 100 epochs then linearly decay for 100
LAMBDA_IDENTITY = 0 # Weight for identity loss
LAMBDA_CYCLE = 10 # Weight for cycle loss
NUM_WORKERS = 4 # Number of workers for dataloader
ALPHA = 0.3  # times image factor

# List of transformations to apply to the input images
TRANSFORM_LIST = [
    transforms.Resize((256, 256)), # Resize the image while maintaining aspect ratio
    ClipOutliers(outlier_cut=1),  # Clip outliers (standard devisations)
    transforms.Lambda(min_max_normalization),  # Min-max normalization
    transforms.RandomHorizontalFlip(p=0.5), # Randomly flip the image horizontally
    transforms.RandomVerticalFlip(p=0.5), # Randomly flip the image vertically
]