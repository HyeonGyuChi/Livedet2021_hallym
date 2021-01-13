import os
import cv2
import numpy as np
import pandas as pd
import albumentations
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


import matplotlib.pylab as plt

def get_df(source_path):

    print(source_path)
    df_test = pd.read_csv(source_path, delimiter='\n', names=['filepath'])

    target_idx = 0

    return df_test, target_idx

def get_transforms(image_size):
    transforms_template = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])

    transforms_probe = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])
    return transforms_template, transforms_probe


class MMC_FPDataset(Dataset):

    def __init__(self, template_csv, probe_csv, template_transform=None, probe_transform=None):
        self.template_csv = template_csv.reset_index(drop=True)
        self.probe_csv = probe_csv.reset_index(drop=True)

        #Need some events that len(template_csv) == len(probe)

        self.template_transform = template_transform
        self.probe_transform = probe_transform

    def __len__(self):
        return self.template_csv.shape[0]

    def __getitem__(self, index):
        #Get row, same index each dataframe
        template_row = self.template_csv.iloc[index]
        probe_row = self.probe_csv.iloc[index]

        #Get imagefile
        template_img = cv2.imread(template_row.filepath)
        probe_img = cv2.imread(probe_row.filepath)
        
        #Convert image RGB -> BGR
        template_img = cv2.cvtColor(template_img, cv2.COLOR_RGB2BGR)
        probe_img = cv2.cvtColor(probe_img, cv2.COLOR_RGB2BGR)

        #Transform template image
        if self.template_transform is not None:
            res = self.template_transform(image=template_img)
            template_img = res['image'].astype(np.float32)
        else:
            template_img = template_img.astype(np.float32)        

        #Transform probe image
        if self.probe_transform is not None:
            res = self.probe_transform(image=probe_img)
            probe_img = res['image'].astype(np.float32)
        else:
            probe_img = probe_img.astype(np.float32)

        #image = cv2.normalize(image, image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)  ### 이부분!!###
        template_img = template_img.transpose(2, 0, 1)
        probe_img = probe_img.transpose(2, 0, 1)

        #To tensor(float)
        template_data = torch.tensor(template_img).float()
        probe_data = torch.tensor(probe_img).float()

        return template_data, probe_data
