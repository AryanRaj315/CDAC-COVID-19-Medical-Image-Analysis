import numpy as np 
import pandas as pd
import cv2
import albumentations as aug
from albumentations import (HorizontalFlip, RandomResizedCrop, VerticalFlip,OneOf, ShiftScaleRotate, Normalize, Resize, Compose,Cutout, GaussNoise, RandomRotate90, Transpose, RandomBrightnessContrast, RandomCrop)
from albumentations import ElasticTransform, GridDistortion, OpticalDistortion, Blur, RandomGamma
from albumentations.pytorch import ToTensor
from torch.utils.data import DataLoader, Dataset, sampler


class Covid_Dataset(Dataset):
    def __init__(self, df, phase='train', transform =True):
        self.df = df
        self.phase = phase
        self.aug = get_transforms(self.phase)
        self.transform = transform
    def __getitem__(self,idx):
        image = cv2.imread(self.df.loc[idx].Path)
        image = cv2.resize(image, (256, 256), interpolation = cv2.INTER_NEAREST)
        label = self.df.loc[idx].Covid
        label = np.asarray(label).reshape(1,)
        augment = self.aug(image =image)
        image = augment['image']       
        return image,label
    def __len__(self):
        return len(self.df)

def get_transforms(phase):
    """
        This function returns the transformation list.
        These are some commonly used augmentation techniques that 
        I believed would be useful.
    """
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
             HorizontalFlip(p = 0.5),
             VerticalFlip(p = 0.5),
            
            Cutout(num_holes=4, p=0.5),
            ShiftScaleRotate(p=1,border_mode=cv2.BORDER_CONSTANT),
#             OneOf([
#             ElasticTransform(p=0.1, alpha=1, sigma=50, alpha_affine=50,border_mode=cv2.BORDER_CONSTANT),
#             GridDistortion(distort_limit =0.05 ,border_mode=cv2.BORDER_CONSTANT, p=0.1),
#             OpticalDistortion(p=0.1, distort_limit= 0.05, shift_limit=0.2,border_mode=cv2.BORDER_CONSTANT)                  
#             ], p=0.3),
#              OneOf([
#             Blur(blur_limit=7)
#             ], p=0.4),    
#             RandomGamma(p=0.8)
        ]
      )
    list_transforms.extend(
        [
#             RandomResizedCrop(height = 224, width = 224, p = 1),
#             Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225], p=1),
            ToTensor(),
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms

def provider(phase,batch_size=16, num_workers=0):
    """
        This function returns the dataloader according to 
        the phase passed.
    """
    
    if phase == 'train' :
        df = pd.read_csv('df_train.csv')
        image_dataset = Covid_Dataset(df)
    else:
        df = pd.read_csv('df_val.csv')
        image_dataset = Covid_Dataset(df, transform = False)
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        shuffle=True,   
    )
    return dataloader