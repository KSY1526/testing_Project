from torchvision import datasets, transforms
from base import BaseDataLoader
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torch


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class MovieDataLoader(BaseDataLoader):
    """
    MovieLens 100k Dataset modified data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):

        self.data_dir = data_dir
        self.data = pd.read_csv('data/movie.csv')
        self.dataset = TensorDataset(torch.LongTensor(np.array(self.data[['user_id', 'movie_id']])), torch.FloatTensor(np.array(self.data[['implicit_feedback']])))
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)