import numpy as np
import torch
from torch.utils.data import Dataset


class ImagePoolDataset(Dataset):
    def __init__(self, pool_path, target_path, nsamples=500):
        self.toa_pool = np.load(pool_path, mmap_mode='r')
        self.atm_targets = np.load(target_path)
        self.nsamples = nsamples
        self.num_pixels_in_pool = self.toa_pool.shape[1]

    def __len__(self):
        return self.toa_pool.shape[0]

    def __getitem__(self, idx):
        indices = np.random.randint(0, self.num_pixels_in_pool, self.nsamples)
        toa_batch = self.toa_pool[idx, indices, :]
        atm_batch = self.atm_targets[idx][indices]

        return {
            "toa": torch.from_numpy(toa_batch),
            "atmosphere": torch.from_numpy(self.atm_targets[idx])
        }
