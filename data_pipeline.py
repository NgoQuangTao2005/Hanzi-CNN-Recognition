import os
import json
import torch
import h5py
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

train_dir = r'train_dataset.h5'
test_dir = r'test_dataset.h5'


train_transforms = transforms.Compose([
    transforms.RandomAffine(
        degrees=8,               
        translate=(0.08, 0.08), 
        scale=(0.92, 1.08)       
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class HanziHDF5Dataset(Dataset):
    def __init__(self, hdf5_path, transform=None):
        self.hdf5_path = hdf5_path
        self.transform = transform
        with h5py.File(self.hdf5_path, 'r') as f:
            self.length = len(f['labels'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if not hasattr(self, 'hdf5_file'):
            self.hdf5_file = h5py.File(self.hdf5_path, 'r')
            
        img_array = self.hdf5_file['images'][idx]
        label = self.hdf5_file['labels'][idx]
        
        img = Image.fromarray(img_array)
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(label, dtype=torch.long)

print("Loading Train dataset from HDF5...")
train_dataset = HanziHDF5Dataset(hdf5_path=train_dir, transform=train_transforms)

print("Loading Test dataset from HDF5...")
test_dataset = HanziHDF5Dataset(hdf5_path=test_dir, transform=test_transforms)

try:
    with open(r'mapping.json', 'r', encoding='utf-8') as f:
        idx_to_class = json.load(f)
    print(f"Loaded mapping.json successfully with {len(idx_to_class)} Chinese characters.")
except FileNotFoundError:
    print("ERROR: mapping.json not found.")

BATCH_SIZE = 256 
NUM_WORKERS = 2

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True, 
    num_workers=NUM_WORKERS, 
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)

NUM_CLASSES = len(idx_to_class)

print(f"Total batches in Train set: {len(train_loader)}")
print(f"Total batches in Test set: {len(test_loader)}")
print("Ready to be used in model.ipynb")