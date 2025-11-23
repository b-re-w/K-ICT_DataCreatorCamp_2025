

from .index import LandsatIndex
from .sentinel import SentinelDataset


class LandsatDatasetForSegmentation(SentinelDataset):
    dataset_name = "Landsat"
