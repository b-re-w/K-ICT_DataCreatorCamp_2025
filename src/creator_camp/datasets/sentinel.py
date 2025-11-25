import traceback
from os import path
from glob import glob
from enum import Enum
from pathlib import Path
from typing import Union, Optional, Callable

import numpy as np
import torch

from torchvision.datasets import VisionDataset, utils
import rasterio

from tqdm.asyncio import tqdm
import concurrent.futures
import asyncio

import nest_asyncio
nest_asyncio.apply()

from .index import SentinelIndex


class DatasetModals(Enum):
    RGB = "rgb"
    NIR = "nir"
    GEMS = "gems"
    AIR = "air"


class SentinelDataset(VisionDataset):
    dataset_name = "Sentinel"

    CLASSES = "background", "industrial_area"  # Class definitions: 0=background, 1=industrial_area
    PALETTE = [0], [1]  # Grayscale palette | Background: black, Industrial area: white
    ORIGINAL_PALETTE = [90, 90, 90], [10, 10, 10]  # Background: gray, Industrial area: black

    DIRECTORIES = ["images", "masks", "gems", "air"]
    DATA_LIST = [
        SentinelIndex.TRAIN, SentinelIndex.VALID,
        SentinelIndex.TRAIN_MASK, SentinelIndex.VALID_MASK,
        SentinelIndex.TRAIN_GEMS, SentinelIndex.VALID_GEMS,
        SentinelIndex.TRAIN_AIR, SentinelIndex.VALID_AIR,
    ]
    TRAIN_LIST = [SentinelIndex.TRAIN, SentinelIndex.TRAIN_MASK, SentinelIndex.TRAIN_GEMS, SentinelIndex.TRAIN_AIR]  # should be matched order with extract_dirs and valid_list
    VALID_LIST = [SentinelIndex.VALID, SentinelIndex.VALID_MASK, SentinelIndex.VALID_GEMS, SentinelIndex.VALID_AIR]

    @classmethod
    async def download_method(cls, url, root, filename):
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            await loop.run_in_executor(executor, utils.download_url, url, root, filename)

    @classmethod
    async def extract_method(cls, from_path, to_path):
        try:
            loop = asyncio.get_event_loop()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                await loop.run_in_executor(executor, utils.extract_archive, from_path, to_path)
        except FileExistsError as e:
            traceback.print_exc()
            raise FileExistsError(str(e) + "\nPlease use Python 3.13 or later. 3.12 or earlier versions not support unzip over existing directory.")

    def __init__(
        self,
        root: Union[str, Path] = None,
        train: bool = True,
        data_type: DatasetModals | list[DatasetModals] = DatasetModals.RGB,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        """
        Sentinel-2 dataset for semantic segmentation.
        
        Args:
            root: Dataset root directory
            train: True for training set, False for validation set
            transforms: Joint transforms for image+mask
            transform: Image transforms
            target_transform: Mask transforms
        """
        super().__init__(root, transforms=transforms, transform=transform, target_transform=target_transform)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.download(root))

        self.types = data_type if isinstance(data_type, list) else [data_type]
        self.root = path.join(root, self.dataset_name)
        self.train = train
        split = "train" if train else "val"
        self.images, self.masks, self.gems, self.air = lists = [], [], [], []
        extract_dirs = [path.join(self.root, anno, split) for anno in self.DIRECTORIES]
        for lst, anno in zip(lists, self.DIRECTORIES):
            lst.extend(sorted(glob(path.join(extract_dirs[self.DIRECTORIES.index(anno)], "*.tif"))))

        assert len(self.images) == len(self.masks), \
            f"Number of images ({len(self.images)}) and masks ({len(self.masks)}) do not match."
        if DatasetModals.GEMS in self.types:
            assert len(self.images) == len(self.gems), \
                f"Number of images ({len(self.images)}) and GEMS data ({len(self.gems)}) do not match."
        if DatasetModals.AIR in self.types:
            assert len(self.images) == len(self.air), \
                f"Number of images ({len(self.images)}) and AIR data ({len(self.air)}) do not match."

        self.cached_data = {}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if idx in self.cached_data:
            image, mask, gems, air = self.cached_data[idx]
        else:
            # Load image/mask using default_loader
            image = self.load_raster(self.images[idx], channels=(1, 2, 3, 4) if DatasetModals.NIR in self.types else (1, 2, 3), normalize=True)
            mask = self.load_raster(self.masks[idx])

            # Convert mask to grayscale
            mask = torch.where(mask == 10, 1, 0).to(torch.uint8)

            # Load additional data if specified
            gems, air = None, None
            if DatasetModals.GEMS in self.types:
                gems = self.load_raster(self.gems[idx], channels=range(1, 11), normalize=True)
            if DatasetModals.AIR in self.types:
                air = self.load_raster(self.air[idx], channels=range(1, 7), normalize=True)
    
            # Cache the loaded data
            self.cached_data[idx] = (image, mask, gems, air)

        # Apply transforms
        if self.transforms:
            # Joint transforms (e.g., albumentations)
            try:
                image, mask, gems, air = self.transforms(image, mask, gems, air)
            except Exception:
                image, mask = self.transforms(image, mask)
        else:
            # Individual transforms
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                try:
                    mask, gems, air = self.target_transform(mask, gems, air)
                except Exception:
                    mask = self.target_transform(mask)

        return image, mask, gems, air

    def load_raster(self, path: Path, channels=(1,), normalize=False) -> torch.Tensor:
        """
        Load TIF image using rasterio.
    
        Args:
            path: Path to TIF file
            channels: Channels to read (1-based indexing)
            normalize: Whether to normalize the image to 0-1 range
    
        Returns:
            Normalized image array in (C, H, W) format
        """
        with rasterio.open(path) as src:
            data = src.read(channels)

        # Normalize to 0-1
        if normalize:
            data = data.astype(np.float32)
            # Channel-wise Z-score normalization
            for i in range(data.shape[0]):
                channel_min = data[i].min()
                channel_max = data[i].max()
                diff = channel_max - channel_min
                if diff != 0:
                    data[i] = (data[i] - channel_min) / diff
                else:
                    data[i] = 0
        return torch.from_numpy(data)

    @classmethod
    async def download(cls, root: str):
        dataset_root = path.join(root, cls.dataset_name)
        if path.exists(dataset_root):  # If the dataset directory already exists, skip download
            return

        print(f"INFO: Downloading '{cls.dataset_name}' from server to {root}...")
        routines = []
        for data in cls.DATA_LIST:
            if path.isfile(path.join(root, data.value)):
                print(f"INFO: Dataset archive {data.value} found in the root directory. Skipping download.")
                continue

            routines.extend(cls.download_method(url, root=root, filename=file) for url, file in zip(data.urls, data.names))
        await tqdm.gather(*routines, desc=f"Downloading {len(routines)} files")

        print(f"INFO: Extracting '{cls.dataset_name}' dataset...")
        routines = []
        as_train, as_valid = lambda d: path.join(d, "train"), lambda d: path.join(d, "val")
        extract_dirs = [path.join(dataset_root, anno) for anno in cls.DIRECTORIES]
        for trains, dirs in zip(cls.TRAIN_LIST, extract_dirs):
            routines.extend(cls.extract_method(path.join(root, file), to_path=as_train(dirs)) for file in trains.names)
        for valids, dirs in zip(cls.VALID_LIST, extract_dirs):
            routines.extend(cls.extract_method(path.join(root, file), to_path=as_valid(dirs)) for file in valids.names)

        await tqdm.gather(*routines, desc=f"Extracting {len(routines)} files")


class SentinelDatasetForSegmentation(SentinelDataset):
    pass
