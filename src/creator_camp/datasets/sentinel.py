from .index import SentinelIndex

from torchvision.datasets import VisionDataset, utils, folder

import traceback
from os import path
from glob import glob
from pathlib import Path
from typing import Union, Optional, Callable

from tqdm.asyncio import tqdm
import concurrent.futures
import asyncio

import nest_asyncio
nest_asyncio.apply()


class SentinelDataset(VisionDataset):
    dataset_name = "Sentinel"

    CLASSES = "background", "industrial_area"  # Class definitions: 0=background, 1=industrial_area
    PALETTE = [90, 90, 90], [0, 0, 0]  # Background: gray, Industrial area: black

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

        self.root = path.join(root, self.dataset_name)
        self.train = train
        split = "train" if train else "val"
        img_dir = path.join(self.root, "images", split)
        ann_dir = path.join(self.root, "annotations", split)

        self.images = sorted(glob(path.join(img_dir, "*.tif")))
        self.masks = sorted(glob(path.join(ann_dir, "*.tif")))

        assert len(self.images) == len(self.masks), \
            f"Number of images ({len(self.images)}) and masks ({len(self.masks)}) do not match."

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        # Load image/mask using default_loader
        image = folder.default_loader(self.images[idx])
        mask = folder.default_loader(self.masks[idx])

        # Convert mask to grayscale
        if mask.mode != 'L':
            mask = mask.convert('L')

        # Apply transforms
        if self.transforms:
            # Joint transforms (e.g., albumentations)
            image, mask = self.transforms(image, mask)
        else:
            # Individual transforms
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                mask = self.target_transform(mask)

        return image, mask

    @classmethod
    async def download(cls, root: str):
        dataset_root = path.join(root, cls.dataset_name)
        if path.exists(dataset_root):  # If the dataset directory already exists, skip download
            return

        data_list = [SentinelIndex.TRAIN, SentinelIndex.VALID, SentinelIndex.TRAIN_MASK, SentinelIndex.VALID_MASK]

        print(f"INFO: Downloading '{cls.dataset_name}' from server to {root}...")
        routines = []
        for data in data_list:
            if path.isfile(path.join(root, data.value)):
                print(f"INFO: Dataset archive {data.value} found in the root directory. Skipping download.")
                continue

            if data == SentinelIndex.TRAIN:
                routines.extend(
                    cls.download_method(url, root=root, filename=file)
                    for url, file in zip(
                        SentinelIndex.get_partial_train_urls(),
                        SentinelIndex.get_partial_trains(),
                    )
                )
            else:
                routines.append(cls.download_method(data.url, root=root, filename=data.value))
        await tqdm.gather(*routines, desc="Downloading files")

        print(f"INFO: Extracting '{cls.dataset_name}' dataset...")
        routines = []
        img_dir, anno_dir = path.join(dataset_root, "images"), path.join(dataset_root, "annotations")
        as_train, as_valid = lambda d: path.join(d, "train"), lambda d: path.join(d, "val")
        if path.isfile(path.join(root, SentinelIndex.TRAIN.value)):
            routines.append(cls.extract_method(path.join(root, SentinelIndex.TRAIN.value), to_path=as_train(img_dir)))
        else:
            routines.extend(
                cls.extract_method(
                    path.join(root, file), to_path=as_train(img_dir)
                ) for file in SentinelIndex.get_partial_trains()
            )
        routines.extend((
            cls.extract_method(path.join(root, SentinelIndex.VALID.value), to_path=as_valid(img_dir)),
            cls.extract_method(path.join(root, SentinelIndex.TRAIN_MASK.value), to_path=as_train(anno_dir)),
            cls.extract_method(path.join(root, SentinelIndex.VALID_MASK.value), to_path=as_valid(anno_dir)),
        ))
        await tqdm.gather(*routines, desc="Extracting files")
