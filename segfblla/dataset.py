#
# Copyright 2015 Benjamin Kiessling
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
"""
Utility functions for data loading and training of VGSL networks.
"""
import traceback
from collections import defaultdict
from itertools import groupby
from typing import TYPE_CHECKING, Any, Callable, Dict, Sequence, Tuple, Union, Optional

import cv2
cv2.setNumThreads(0)

import numpy as np
import pytorch_lightning as pl
import shapely.geometry as geom
import torch
import torch.nn.functional as F
import albumentations as A

from PIL import Image
from shapely.ops import split
from skimage.draw import polygon
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.transforms import v2

from kraken.lib.xml import XMLPage

if TYPE_CHECKING:
    from kraken.containers import Segmentation
    from os import PathLike

__all__ = ['BaselineSet']

import logging

logger = logging.getLogger(__name__)


class BaselineDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_files: Sequence[Union[str, 'PathLike']],
                 val_files: Optional[Sequence[Union[str, 'PathLike']]] = None,
                 line_width: int = 4,
                 augmentation: bool = False,
                 valid_baselines: Sequence[str] = None,
                 merge_baselines: Dict[str, Sequence[str]] = None,
                 valid_regions: Sequence[str] = None,
                 merge_regions: Dict[str, Sequence[str]] = None,
                 batch_size: int = 16,
                 num_workers: int = 8,
                 partition: Optional[float] = 0.95,
                 topline: Union[bool, None] = False,
                 patch_size: Tuple[int, int] = (512, 512)):
        super().__init__()

        self.save_hyperparameters()


        logger.info(f'Parsing {len(train_files)} XML files for training data')
        train_data = [XMLPage(file).to_container() for file in self.hparams.train_files]

        val_data = None

        if val_files:
            logger.info(f'Parsing {len(val_files)} XML files for validation data')
            val_data = [XMLPage(file).to_container() for file in self.hparams.val_files]

        if not train_data:
            raise ValueError('No training data provided. Please add some.')

        train_set = BaselineSet(line_width=self.hparams.line_width,
                                augmentation=self.hparams.augmentation,
                                valid_baselines=self.hparams.valid_baselines,
                                merge_baselines=self.hparams.merge_baselines,
                                valid_regions=self.hparams.valid_regions,
                                merge_regions=self.hparams.merge_regions)

        for page in train_data:
            train_set.add(page)

        if val_data:
            val_set = BaselineSet(line_width=self.hparams.line_width,
                                  augmentation=False,
                                  valid_baselines=self.hparams.valid_baselines,
                                  merge_baselines=self.hparams.merge_baselines,
                                  valid_regions=self.hparams.valid_regions,
                                  merge_regions=self.hparams.merge_regions)

            for page in val_data:
                val_set.add(page)

            train_set = Subset(train_set, range(len(train_set)))
            val_set = Subset(val_set, range(len(val_set)))
        else:
            train_len = int(len(train_set)*self.hparams.partition)
            val_len = len(train_set) - train_len
            logger.info(f'No explicit validation data provided. Splitting off '
                        f'{val_len} (of {len(train_set)}) samples to validation '
                        'set.')
            train_set, val_set = random_split(train_set, (train_len, val_len))

        if len(train_set) == 0:
            raise ValueError('No valid training data provided. Please add some.')

        if len(val_set) == 0:
            raise ValueError('No valid validation data provided. Please add some.')

        # overwrite class mapping in validation set
        val_set.dataset.num_classes = train_set.dataset.num_classes
        val_set.dataset.class_mapping = train_set.dataset.class_mapping

        self.num_classes = train_set.dataset.num_classes
        self.class_mapping = train_set.dataset.class_mapping
        self.train_class_stats = train_set.dataset.class_stats
        self.val_class_stats = val_set.dataset.class_stats

        self.bl_train = train_set
        self.bl_val = val_set

        self.save_hyperparameters()

    def train_dataloader(self):
        return DataLoader(self.bl_train,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.bl_val,
                          batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)


    def state_dict(self):
        # track whatever you want here
        return {"class_mapping": self.class_mapping}

    def load_state_dict(self, state_dict):
        # restore the state based on what you tracked in (def state_dict)
        self.class_mapping = state_dict['class_mapping']


class BaselineSet(Dataset):
    """
    Dataset for training a baseline/region segmentation model.
    """
    def __init__(self,
                 line_width: int = 4,
                 augmentation: bool = False,
                 valid_baselines: Sequence[str] = None,
                 merge_baselines: Dict[str, Sequence[str]] = None,
                 valid_regions: Sequence[str] = None,
                 merge_regions: Dict[str, Sequence[str]] = None):
        """
        Creates a dataset for a text-line and region segmentation model.

        Args:
            line_width: Height of the baseline in the scaled input.
            target_size: Target size of the image as a (height, width) tuple.
            augmentation: Enable/disable augmentation.
            valid_baselines: Sequence of valid baseline identifiers. If `None`
                             all are valid.
            merge_baselines: Sequence of baseline identifiers to merge.  Note
                             that merging occurs after entities not in valid_*
                             have been discarded.
            valid_regions: Sequence of valid region identifiers. If `None` all
                           are valid.
            merge_regions: Sequence of region identifiers to merge. Note that
                           merging occurs after entities not in valid_* have
                           been discarded.
        """
        super().__init__()
        self.imgs = []
        self.im_mode = '1'
        self.targets = []
        # n-th entry contains semantic of n-th class
        self.class_mapping = {'aux': {'_start_separator': 0, '_end_separator': 1}, 'baselines': {}, 'regions': {}}
        # keep track of samples that failed to load
        self.failed_samples = set()
        self.class_stats = {'baselines': defaultdict(int), 'regions': defaultdict(int)}
        self.num_classes = 2
        self.mbl_dict = merge_baselines if merge_baselines is not None else {}
        self.mreg_dict = merge_regions if merge_regions is not None else {}
        self.valid_baselines = valid_baselines
        self.valid_regions = valid_regions

        self.aug = None
        if augmentation:
            self.aug = A.Compose([A.ToFloat(),
                                  A.OneOf([
                                      A.MotionBlur(p=0.2),
                                      A.MedianBlur(blur_limit=3, p=0.1),
                                      A.Blur(blur_limit=3, p=0.1),
                                  ], p=0.2),
                                  A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
                                  A.OneOf([
                                      A.OpticalDistortion(p=0.3),
                                      A.ElasticTransform(p=0.1),
                                  ], p=0.2),
                                  A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=0.1, val_shift_limit=0.1, p=0.3),
                                  A.ToGray(p=0.5)
                                 ], p=0.5)
        self.line_width = line_width
        self.transforms = v2.Compose([v2.PILToTensor(),
                                      v2.ConvertDtype(torch.float32),
                                      v2.Normalize(mean=(0.485, 0.456, 0.406),
                                                   std=(0.229, 0.224, 0.225))
                                     ]
                                    )
        self.patch_crop = v2.RandomCrop((512, 512))
        self.seg_type = None

    def add(self, doc: 'Segmentation'):
        """
        Adds a page to the dataset.

        Args:
            doc: A Segmentation container class.
        """
        if doc.type != 'baselines':
            raise ValueError(f'{doc} is of type {doc.type}. Expected "baselines".')

        baselines_ = defaultdict(list)
        for line in doc.lines:
            if self.valid_baselines is None or set(line.tags.values()).intersection(self.valid_baselines):
                tags = set(line.tags.values()).intersection(self.valid_baselines) if self.valid_baselines else line.tags.values()
                tags = set([self.mbl_dict.get(v, v) for v in tags])
                for tag in tags:
                    baselines_[tag].append(line.baseline)
                    self.class_stats['baselines'][tag] += 1

                    if tag not in self.class_mapping['baselines']:
                        self.num_classes += 1
                        self.class_mapping['baselines'][tag] = self.num_classes - 1

        regions_ = defaultdict(list)
        for k, v in doc.regions.items():
            if self.valid_regions is None or k in self.valid_regions:
                reg_type = self.mreg_dict.get(k, k)
                regions_[reg_type].extend(v)
                self.class_stats['baselines'][reg_type] += len(v)
                if reg_type not in self.class_mapping['regions']:
                    self.num_classes += 1
                    self.class_mapping['regions'][reg_type] = self.num_classes - 1
                self.class_stats['regions'][reg_type] += 1
        self.targets.append({'baselines': baselines_, 'regions': regions_})
        self.imgs.append(doc.imagename)

    def __getitem__(self, idx):
        im = self.imgs[idx]
        target = self.targets[idx]
        if not isinstance(im, Image.Image):
            try:
                logger.debug(f'Attempting to load {im}')
                im = Image.open(im)
                im, target = self.transform(im, target)
                return {'image': im, 'target': target}
            except Exception:
                raise
                self.failed_samples.add(idx)
                idx = np.random.randint(0, len(self.imgs))
                logger.debug(traceback.format_exc())
                logger.info(f'Failed. Replacing with sample {idx}')
                return self[idx]
        im, target = self.transform(im, target)
        return {'image': im, 'target': target}

    @staticmethod
    def _get_ortho_line(lineseg, point, line_width, offset):
        lineseg = np.array(lineseg)
        norm_vec = lineseg[1, ...] - lineseg[0, ...]
        norm_vec_len = np.sqrt(np.sum(norm_vec**2))
        unit_vec = norm_vec / norm_vec_len
        ortho_vec = unit_vec[::-1] * ((1, -1), (-1, 1))
        if offset == 'l':
            point -= unit_vec * line_width
        else:
            point += unit_vec * line_width
        return (ortho_vec * 10 + point).astype('int').tolist()

    def transform(self, image, target):
        orig_size = image.size
        image = self.transforms(image)
        i, j, h, w = self.patch_crop.get_params(image, (512, 512))
        image = v2.functional.crop(image, i, j, h, w)

        t = torch.zeros((self.num_classes,) + image.shape[1:])

        start_sep_cls = self.class_mapping['aux']['_start_separator']
        end_sep_cls = self.class_mapping['aux']['_end_separator']

        for key, lines in target['baselines'].items():
            try:
                cls_idx = self.class_mapping['baselines'][key]
            except KeyError:
                # skip lines of classes not present in the training set
                continue
            for line in lines:
                # buffer out line to desired width
                line = [k for k, g in groupby(line)]
                line = np.array(line)
                shp_line = geom.LineString(line)
                split_offset = min(5, shp_line.length/2)
                line_pol = shp_line.buffer(self.line_width/2, cap_style=2)
                # negative coords get ignored by skimage.draw.polygon
                line_pol = np.array(line_pol.boundary.coords, dtype=int) - (j, i)
                rr, cc = polygon(line_pol[:, 1], line_pol[:, 0], shape=image.shape[1:])
                t[cls_idx, rr, cc] = 1
                split_pt = shp_line.interpolate(split_offset).buffer(0.001)
                # top
                start_sep = split(shp_line, split_pt).geoms[0].buffer(self.line_width, cap_style=3)
                # start_sep in image
                if start_sep:
                    start_sep = np.array(start_sep.boundary.coords, dtype=int) - (j, i)
                    rr_s, cc_s = polygon(start_sep[:, 1], start_sep[:, 0], shape=image.shape[1:])
                    t[start_sep_cls, rr_s, cc_s] = 1
                    t[start_sep_cls, rr, cc] = 0
                split_pt = shp_line.interpolate(-split_offset).buffer(0.001)
                # top
                end_sep = split(shp_line, split_pt).geoms[-1].buffer(self.line_width, cap_style=3)
                # end_sep in image
                if end_sep:
                    end_sep = np.array(end_sep.boundary.coords, dtype=int) - (j, i)
                    rr_s, cc_s = polygon(end_sep[:, 1], end_sep[:, 0], shape=image.shape[1:])
                    t[end_sep_cls, rr_s, cc_s] = 1
                    t[end_sep_cls, rr, cc] = 0
        for key, regions in target['regions'].items():
            try:
                cls_idx = self.class_mapping['regions'][key]
            except KeyError:
                # skip regions of classes not present in the training set
                continue
            for region in regions:
                region = np.array(region.boundary, dtype=int) - (j, i)
                rr, cc = polygon(region[:, 1], region[:, 0], shape=image.shape[1:])
                t[cls_idx, rr, cc] = 1
        target = t
        if self.aug:
            image = image.permute(1, 2, 0).numpy()
            target = target.permute(1, 2, 0).numpy()
            o = self.aug(image=image, mask=target)
            image = torch.tensor(o['image']).permute(2, 0, 1)
            target = torch.tensor(o['mask']).permute(2, 0, 1)
        return image, target

    def __len__(self):
        return len(self.imgs)
