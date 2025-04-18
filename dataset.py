import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import timm  # A library that contains ViT models
import random
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



# 1. Define CutMix augmentation class
class CutMix:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def __call__(self, image, target):
        # Randomly sample lambda
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size, _, height, width = image.size()
        
        # Get the random location to cutmix
        cx, cy = np.random.randint(width), np.random.randint(height)
        bw, bh = int(width * np.sqrt(1 - lam)), int(height * np.sqrt(1 - lam))
        
        # Get the coordinates for the cutmix patch
        x1 = np.clip(cx - bw // 2, 0, width)
        x2 = np.clip(cx + bw // 2, 0, width)
        y1 = np.clip(cy - bh // 2, 0, height)
        y2 = np.clip(cy + bh // 2, 0, height)
        
        # CutMix and swap the patches between images
        image[:, :, y1:y2, x1:x2] = image.flip(0)[:, :, y1:y2, x1:x2]
        
        return image, target

# 2. Define MixUp augmentation class
class MixUp:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def __call__(self, image, target):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size, _, height, width = image.size()
        
        # Shuffle the batch
        index = torch.randperm(batch_size).cuda()
        
        # Mix images
        mixed_image = lam * image + (1 - lam) * image[index, :]
        return mixed_image, target, lam
    

# 4. Data augmentation pipeline with CutMix and MixUp
def get_transformations(cutmix_alpha=1.0, mixup_alpha=1.0):
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform, CutMix(alpha=cutmix_alpha), MixUp(alpha=mixup_alpha)


# 5. Load ImageNet dataset (assuming the dataset is pre-downloaded)
def load_imagenet_data(batch_size=64):
    train_transform, cutmix, mixup = get_transformations()

    train_dataset = datasets.ImageNet(root='./data', split='train', transform=train_transform)
    val_dataset = datasets.ImageNet(root='./data', split='val', transform=train_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, cutmix, mixup



# Load Pascal VOC or MSCOCO dataset
def load_pascal_voc(batch_size=32):
    dataset = datasets.VOCDetection(root='./data', year='2012', image_set='val', download=True, transform=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return dataloader

def load_coco(batch_size=32):
    dataset = datasets.CocoDetection(root='./data', annFile='./data/annotations/instances_val2017.json', transform=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return dataloader
