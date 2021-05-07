import os
import random
import torch
import pandas as pd
import torchvision.transforms.functional as T_F
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
        self.steer_correction = 0.1

    def __len__(self):
        # center, left, right images are all used
        return 3 * len(self.driving_records)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # use center, left, right image
        row_index = int(idx / 3)
        column_index = idx % 3

        # take the center image
        img_name = os.path.join(self.root_dir,
                                self.driving_records.iloc[row_index, column_index])
        image = Image.open(img_name)
        steer = torch.tensor(self.driving_records.iloc[row_index, 3], dtype=torch.float32)
        if column_index == 1:
            # left image
            steer = steer + self.steer_correction
        elif column_index == 2:
            # right image
            steer = steer - self.steer_correction

        # add random hflip because steer needs to be modified when horizontally flipping image
        if random.random() < 0.5:
            image = T_F.hflip(image)
            steer = -1.0 * steer

        if self.transform:
            image = self.transform(image)
        return image, steer
