import os
import torch
import config
import numpy as np
from PIL import Image
from normalizations import *
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

def create_loaders(test_percent=0.2, batch_size=64, sdo_channels=['171'], iris_channel='1400'):
    """
    Creates data loaders for training and testing based on the provided parameters.
    Args:
        test_percent (float, optional): The percentage of data to be used for testing.
            Defaults to 0.8 (80% for training and 20% for testing).
        batch_size (int, optional): The batch size for the data loaders.
            Defaults to 64.
    Returns:
        tuple: A tuple containing the training and testing data loaders.
    Raises:
        AssertionError: If the observation directories for SDO and IRIS data do not match.
    """
    assert set(os.listdir("../data/sdo/")) == set(os.listdir("../data/iris/")), 'obs not in both directories'
    if batch_size > 1:
        print('Warning: Center cropping images to [463, 463] to allow for batch sizes larger than 1')
    # Get the list of observations for SDO data
    obs_list = os.listdir("../data/sdo/")
    # Split the observations into training and testing sets
    train_set, test_set = train_test_split(obs_list, test_size=test_percent, random_state=42)
    train_obs_list = train_set
    test_obs_list = test_set
    # Create the training and testing datasets
    train_dataset = AIAIRISDataset(
        sdo_channels=sdo_channels, iris_channel=iris_channel, obs_list=train_obs_list, bs=batch_size)
    test_dataset = AIAIRISDataset(
        sdo_channels=sdo_channels, iris_channel=iris_channel, obs_list=test_obs_list, bs=batch_size)
    # Create the training and testing data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader

class AIAIRISDataset(Dataset):
    """
    Custom dataset class for AIA and IRIS images.
    Args:
        sdo_channels (list, optional): List of SDO channels to include. Defaults to ['304'].
        iris_channel (str, optional): IRIS channel to include. Defaults to '1400'.
        obs_list (list, optional): List of observations to include. Defaults to None.
    Attributes:
        iris_paths (ndarray): Array of paths to IRIS images.
        sdo_paths (ndarray): Array of paths to SDO images.
        transform (Compose): Composition of image transformations.
    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns the preprocessed images at the given index.
    """
    def __init__(self, sdo_channels=['304'], iris_channel='1400', obs_list=None, bs=64):
        super(AIAIRISDataset, self).__init__()
        self.bs = bs
        # Collect paths to IRIS images
        iris_paths = []
        for obs in obs_list:
            if obs == '20160212_135911_3690113103': continue
            path_to_iris_obs = f'../data/iris/{obs}/{iris_channel}/'
            for file in os.listdir(path_to_iris_obs):
                iris_paths.append(f'{path_to_iris_obs}/{file}')
        # Collect paths to SDO images
        sdo_paths = []
        for obs in obs_list:
            if obs == '20160212_135911_3690113103': continue
            path_to_sdo_obs = f'../data/sdo/{obs}/'
            for sdo_channel in sdo_channels:
                path_to_sdo_obs_channel = f'{path_to_sdo_obs}/{sdo_channel}/'
                for file in os.listdir(path_to_sdo_obs_channel):
                    sdo_paths.append(f'{path_to_sdo_obs_channel}/{file}')
        # Randomly shuffle the indices for both IRIS and SDO images. Note that results could imporve if images alligned in time
        iris_rand_ints = np.random.choice(
            len(iris_paths), size=len(iris_paths), replace=False)
        sdo_rand_ints = np.random.choice(
            len(sdo_paths), size=len(sdo_paths), replace=False)
        # Store the shuffled paths
        self.iris_paths = np.array(iris_paths)[iris_rand_ints]
        self.sdo_paths = np.array(sdo_paths)[sdo_rand_ints]
        self.transform= transforms.Compose(config.TRANSFORM_LIST)
        # if self.bs > 1: self.transform = transforms.Compose([transforms.CenterCrop((463, 463))] + config.TRANSFORM_LIST)
    def __len__(self):
        return min(len(self.iris_paths), len(self.sdo_paths))
    def __getitem__(self, idx): 
        iris_im = np.load(self.iris_paths[idx], allow_pickle=True).astype(np.float32)
        iris_im = Image.fromarray(iris_im)
        iris_im = np.array(iris_im).astype(np.float32)
        iris_im = torch.from_numpy(iris_im)
        iris_im = torch.unsqueeze(iris_im, 0)
        sdo_im = np.load(self.sdo_paths[idx], allow_pickle=True).astype(np.float32)
        sdo_im = torch.from_numpy(sdo_im)
        sdo_im = torch.unsqueeze(sdo_im, 0)
        iris_im = self.transform(iris_im)
        sdo_im = self.transform(sdo_im)
        return iris_im, sdo_im