import collections.abc
import json
import os

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def batch_to_device(batch, device):
    batch_dict = {key: batch[key].to(device) for key in batch}
    return batch_dict


class CustomDataset(Dataset):
    def __init__(self, df, cfg, aug, mode="train"):

        self.cfg = cfg
        self.mode = mode
        self.df = df.copy()
        self.images = self.df["image_path"].astype(str).values
        self.aug = aug

        # Set labels
        self.labels = self.df[self.cfg.classes].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.labels[idx]

        img = self._read_image(idx=idx)

        if self.aug:
            img = self.augment(img)

        img = img / 255.
        feature_dict = {
            "input": torch.tensor(img.transpose(2, 0, 1)).float(),
            "target": torch.tensor(label),
        }
        return feature_dict

    def _read_image(self, idx):
        path = self.images[idx]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return  img

    def augment(self, img):
        # img = img.astype(np.float32)
        img = self.aug(image=img)["image"]
        return img


