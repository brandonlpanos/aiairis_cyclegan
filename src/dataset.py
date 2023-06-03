import torch
import numpy as np
from normalizations import *
from torch.utils.data import Dataset, DataLoader


def create_loaders():
    train_dataset = AIAIRISDataset('train')
    test_dataset = AIAIRISDataset('test')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    return train_loader, test_loader


class AIAIRISDataset(Dataset):
    def __init__(self, mode='train'):
        super(AIAIRISDataset, self).__init__()

        self.iris_data = np.load(f"../data/iris_{mode}.npy", allow_pickle=True)
        self.aia_data = np.load(f"../data/aia_{mode}.npy", allow_pickle=True)

    def __len__(self):
        return min(self.iris_data.shape[0], self.aia_data.shape[0])

    def __getitem__(self, idx):

        iris_im = self.iris_data[idx]
        iris_im = torch.from_numpy(iris_im)
        sdo_im = self.aia_data[idx]
        sdo_im = torch.from_numpy(sdo_im)

        return iris_im, sdo_im
