import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class BehaviorCloneDataset(Dataset):
    """
    Behavioral cloning dataset.
    I referred to https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Read the csv here and leave the reading of images to __getitem__. This is memory efficient because all images
        are not stored in the memory at once but read as required.
        :param csv_file: path to the csv file with relative image paths and corresponding control commands, velocity.
        :param root_dir: directory with all the images.
        :param transform: optional transform to be applied on a sample.
        """
        self.driving_records = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.driving_records)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # take the center image
        img_name = os.path.join(self.root_dir,
                                self.driving_records.iloc[idx, 0])
        image = Image.open(img_name)
        steer = torch.tensor(self.driving_records.iloc[idx, 3], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)
        return image, steer
