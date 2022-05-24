# the custom dataset class
import json
import os

import torch
from torch.utils.data import Dataset
from transformers import TensorType, BatchEncoding
from transformers.file_utils import PaddingStrategy

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


class FangCovidDataset(Dataset):
    def __init__(self, indices: list[int] = 0):
        if not indices:
            indices = [i for i in range(len(os.listdir(Config.fang_covid_dir)))]
        self.indices: list[int] = indices

    def __getitem__(self, idx: int):
        with open(os.path.join(Config.fang_covid_dir, f"{idx}.json")) as f:
            json_dict: dict = json.load(f)
            encodings: BatchEncoding = Config.tokenizer(
                json_dict["article"], truncation=True, padding=PaddingStrategy.MAX_LENGTH, max_length=Config.max_length,
                return_tensors=TensorType.PYTORCH)
            item = {key: val.squeeze().to(Config.device) for key, val in encodings.data.items()}
            label: int = 1 if json_dict["label"] == "fake" else 0
            item['labels'] = torch.tensor(label).to(Config.device)
            return item

    def __len__(self):
        return len(self.indices)
