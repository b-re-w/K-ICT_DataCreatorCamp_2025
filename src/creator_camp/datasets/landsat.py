from typing import Union, Optional, Callable
from pathlib import Path

from .index import LandsatIndex
from .sentinel import SentinelDataset, DatasetModals


class LandsatDataset(SentinelDataset):
    dataset_name = "Landsat"

    CLASSES = "background", "urban_area"  # Class definitions: 0=background, 1=urban_area
    PALETTE = [0], [1]  # Grayscale palette | Background: black, Urban area: white
    ORIGINAL_PALETTE = [90, 90, 90], [10, 10, 10]  # Background: gray, Urban area: black

    DIRECTORIES = ["images", "masks"]
    DATA_LIST = [
        LandsatIndex.TRAIN, LandsatIndex.VALID,
        LandsatIndex.TRAIN_MASK, LandsatIndex.VALID_MASK
    ]
    TRAIN_LIST = [LandsatIndex.TRAIN, LandsatIndex.TRAIN_MASK]  # should be matched order with extract_dirs and valid_list
    VALID_LIST = [LandsatIndex.VALID, LandsatIndex.VALID_MASK]

    def __init__(
        self,
        root: Union[str, Path] = None,
        train: bool = True,
        data_type: DatasetModals | list[DatasetModals] = DatasetModals.RGB,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        types = [dt for dt in data_type if dt in (DatasetModals.RGB, DatasetModals.NIR)]
        super().__init__(root, train, types, transforms, transform, target_transform)


class LandsatDatasetForSegmentation(LandsatDataset):
    pass
