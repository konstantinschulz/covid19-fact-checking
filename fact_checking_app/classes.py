# the custom dataset class
import torch
from torch.utils.data import Dataset

from config import Config


class CovidDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).to(Config.device) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).to(Config.device)
        return item

    def __len__(self):
        return len(self.labels)
