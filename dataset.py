import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, images, masks, transform=None, bs=4):
         self.transform = transform

         self.images = images
         self.masks = masks
         self.bs = bs

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = np.array(Image.open(self.images[index]).convert("RGB"))
        mask = np.array(Image.open(self.masks[index]).convert("L"), dtype=np.float32)

        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

    # def __iter__(self):
    #     for i in range(0, len(self.images), self.bs): 
    #         yield self.images[i:i+self.bs], self.masks[i:i+self.bs]