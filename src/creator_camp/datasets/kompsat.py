from .index import KompsatIndex

from torchvision.datasets import VisionDataset, utils, folder
from torchvision.ops import box_convert
import torch

import traceback
from os import path
from glob import glob
from enum import Enum
from pathlib import Path
from typing import Union, Optional, Callable

from json import load as json_load

from tqdm.asyncio import tqdm
import concurrent.futures
import asyncio

import nest_asyncio
nest_asyncio.apply()


class KompsatType(Enum):
    BBOX = "bbox"
    LINE = "line"


class KompsatDataset(VisionDataset):
    dataset_name = "Kompsat"

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
        data_type: KompsatType = KompsatType.BBOX,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        """
        Kompsat-3/3A dataset for object detection and height estimation.

        Args:
            root: Dataset root directory
            train: True for training set, False for validation set
            data_type: BBOX or LINE
            transform: Image transforms
            target_transform: Mask transforms
        """
        super().__init__(root, transforms=transform, transform=transform, target_transform=target_transform)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.download(root))

        self.root = path.join(root, self.dataset_name)
        self.train = train
        self.data_type = data_type
        split = "train" if train else "val"
        img_dir = path.join(self.root, "images", split)
        ann_dir = path.join(self.root, "annotations", split)
        line_dir = path.join(self.root, "lines", split)

        self.images = sorted(glob(path.join(img_dir, "*.jpg")))
        self.labels = []
        for pth in self.images:
            if data_type is KompsatType.BBOX:
                annotation_path = path.join(ann_dir, Path(pth).stem + ".json")
                if not path.exists(annotation_path):
                    raise FileNotFoundError(f"Annotation file {annotation_path} does not exist.")
                annotation = list(json_load(open(annotation_path, "r", encoding="utf-8")).values())[0]
                regions = []
                for anno in sorted(annotation['regions'], key=lambda x: int(x['region_attributes']['chi_id'])):
                    bbox = anno["shape_attributes"]
                    bbox = [bbox["x"], bbox["y"], bbox["width"], bbox["height"]]
                    regions.append(dict(
                        chi_id=int(anno["region_attributes"]["chi_id"]),
                        xywh=bbox,
                        xyxy=box_convert(torch.tensor(bbox), "xywh", "xyxy").tolist(),
                        cxcywh=box_convert(torch.tensor(bbox), "xywh", "cxcywh").tolist()
                    ))
                annotation['regions'] = regions
                annotation['file_attributes']['img_width'] = int(annotation['file_attributes']['img_width'])
                annotation['file_attributes']['img_height'] = int(annotation['file_attributes']['img_height'])
                self.labels.append(annotation)
            else:
                label_path = path.join(line_dir, Path(pth).stem + ".json")
                if not path.exists(label_path):
                    raise FileNotFoundError(f"Label file {label_path} does not exist.")
                label = list(json_load(open(label_path, "r", encoding="utf-8")).values())[0]
                regions = []
                for ln in sorted(label['regions'], key=lambda x: int(x['region_attributes']['chi_id'])):
                    poly = ln["shape_attributes"]
                    xyxy = [poly["all_points_x"][0], poly["all_points_y"][0], poly["all_points_x"][1], poly["all_points_y"][1]]
                    x1 = min(xyxy[0], xyxy[2])
                    y1 = min(xyxy[1], xyxy[3])
                    w = max(abs(xyxy[2] - xyxy[0]), 1)  # w must be at least 1
                    h = max(abs(xyxy[3] - xyxy[1]), 1)  # h must be at least 1
                    regions.append(dict(
                        polyline=xyxy,
                        polyline_xywh=[x1, y1, w, h],
                        chi_height=float(ln["region_attributes"]['chi_height_m']),
                    ))
                label['regions'] = regions
                label['file_attributes']['img_width'] = int(label['file_attributes']['img_width'])
                label['file_attributes']['img_height'] = int(label['file_attributes']['img_height'])
                self.labels.append(label)

        assert len(self.images) == len(self.labels), \
            f"Number of images ({len(self.images)}) and labels ({len(self.labels)}) do not match."

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        # Load image/label using default_loader
        image = folder.default_loader(self.images[idx])
        label = self.labels[idx]

        # Apply transforms
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    @classmethod
    async def download(cls, root: str):
        dataset_root = path.join(root, cls.dataset_name)
        if path.exists(dataset_root):  # If the dataset directory already exists, skip download
            return

        data_list = [
            KompsatIndex.TRAIN, KompsatIndex.VALID,
            KompsatIndex.TRAIN_BBOX, KompsatIndex.VALID_BBOX,
            KompsatIndex.TRAIN_LINE, KompsatIndex.VALID_LINE
        ]

        print(f"INFO: Downloading '{cls.dataset_name}' from server to {root}...")
        routines = []
        for data in data_list:
            if path.isfile(path.join(root, data.value)):
                print(f"INFO: Dataset archive {data.value} found in the root directory. Skipping download.")
                continue

            routines.append(cls.download_method(data.url, root=root, filename=data.value))
        await tqdm.gather(*routines, desc="Downloading files")

        print(f"INFO: Extracting '{cls.dataset_name}' dataset...")
        routines = []
        img_dir, anno_dir, line_dir = path.join(dataset_root, "images"), path.join(dataset_root, "annotations"), path.join(dataset_root, "lines")
        as_train, as_valid = lambda d: path.join(d, "train"), lambda d: path.join(d, "val")
        routines.extend((
            cls.extract_method(path.join(root, KompsatIndex.TRAIN.value), to_path=as_train(img_dir)),
            cls.extract_method(path.join(root, KompsatIndex.VALID.value), to_path=as_valid(img_dir)),
            cls.extract_method(path.join(root, KompsatIndex.TRAIN_BBOX.value), to_path=as_train(anno_dir)),
            cls.extract_method(path.join(root, KompsatIndex.VALID_BBOX.value), to_path=as_valid(anno_dir)),
            cls.extract_method(path.join(root, KompsatIndex.TRAIN_LINE.value), to_path=as_train(line_dir)),
            cls.extract_method(path.join(root, KompsatIndex.VALID_LINE.value), to_path=as_valid(line_dir)),
        ))
        await tqdm.gather(*routines, desc="Extracting files")


class KompsatDatasetForObjectDetection(KompsatDataset):
    def __init__(
        self,
        root: Union[str, Path] = None,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        super().__init__(root, train, KompsatType.BBOX, transform, target_transform)


class KompsatDatasetForHeightRegression(KompsatDataset):
    def __init__(
        self,
        root: Union[str, Path] = None,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None
    ):
        super().__init__(root, train, KompsatType.LINE, transform, target_transform)
