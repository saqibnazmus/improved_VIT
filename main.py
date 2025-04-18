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
from model import VisionTransformer
from dataset import load_pascal_voc, load_coco
from dataset import get_transformations
from test import test_model



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VisionTransformer(num_classes=20).to(device)  # Assuming 20 classes in VOC or change for MSCOCO

# Load pretrained weights if available
model.load_state_dict(torch.load('best_vit_model.pth'))

# Load test data (Pascal VOC or MSCOCO)
test_dataloader_pascal = load_pascal_voc(batch_size=32)
test_dataloader_coco = load_coco(batch_size=32)

 # Test the model on Pascal VOC
print("\nTesting on Pascal VOC dataset...")
test_model(model, test_dataloader_pascal, device)

# Test the model on MSCOCO
print("\nTesting on MSCOCO dataset...")

test_model(model, test_dataloader_coco, device)