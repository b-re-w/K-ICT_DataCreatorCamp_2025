from .kompsat import KompsatDataset, KompsatDatasetForObjectDetection, KompsatDatasetForHeightRegression
from .sentinel import SentinelDataset

from dataclasses import dataclass
from torch.utils.data import Dataset


@dataclass
class DatasetHolder:
    train: Dataset = None
    valid: Dataset = None
    test: Dataset = None

    def __post_init__(self):
        print(f"INFO: Dataset loaded successfully. Number of samples - ", end='')
        if self.train:
            print(f"Train: {len(self.train)}", end='')
        if self.valid:
            if self.train: print(', ', end='')
            print(f"Valid: {len(self.valid)}", end='')
        if self.test:
            if self.train: print(', ', end='')
            print(f"Test: {len(self.test)}", end='')
        print('\n')
