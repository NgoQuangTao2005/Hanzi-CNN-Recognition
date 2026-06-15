import os
import json
import torch
import h5py
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Đã chuyển sang trỏ vào 2 file nén nguyên khối
train_dir = r'D:\DeepLearning\train_dataset.h5'
test_dir = r'D:\DeepLearning\test_dataset.h5'

# ==========================================
# BƯỚC 1: XÂY DỰNG PIPELINE TIỀN XỬ LÝ (TRANSFORMS)
# ==========================================
train_transforms = transforms.Compose([
    transforms.RandomAffine(
        degrees=8,               # Xoay ngẫu nhiên ±8 độ
        translate=(0.08, 0.08),  # Dịch chuyển ngẫu nhiên ảnh lên/xuống/trái/phải 8%
        scale=(0.92, 1.08)       # Phóng to/thu nhỏ ngẫu nhiên nét chữ 8%
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ==========================================
# BƯỚC 1.5: ĐỊNH NGHĨA DATASET ĐỌC HDF5
# ==========================================
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

# ==========================================
# BƯỚC 2: LOAD DATASET TRỰC TIẾP TỪ FILE HDF5
# ==========================================
print("Đang nạp tập Train từ HDF5...")
train_dataset = HanziHDF5Dataset(hdf5_path=train_dir, transform=train_transforms)

print("Đang nạp tập Test từ HDF5...")
test_dataset = HanziHDF5Dataset(hdf5_path=test_dir, transform=test_transforms)

# ==========================================
# BƯỚC 3: ĐỌC LOOKUP TABLE TỪ FILE ĐÃ CÓ
# ==========================================
# Thay vì tạo mới, ta nạp luôn file mapping cũ đã có sẵn để lấy tổng số class
try:
    with open(r'D:\DeepLearning\mapping.json', 'r', encoding='utf-8') as f:
        idx_to_class = json.load(f)
    print(f"Đã load mapping.json thành công với {len(idx_to_class)} chữ Hán.")
except FileNotFoundError:
    print("❌ LỖI: Không tìm thấy mapping.json.")

# ==========================================
# BƯỚC 4: ĐÓNG GÓI VÀO DATALOADER
# ==========================================
BATCH_SIZE = 256 
NUM_WORKERS = 2

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True, # Bắt buộc trộn câu hỏi khi huấn luyện
    num_workers=NUM_WORKERS, # Số worker để đọc dữ liệu nhanh hơn 
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

print(f"Tổng số batch trong tập Train: {len(train_loader)}")
print(f"Tổng số batch trong tập Test: {len(test_loader)}")
print("✅ Sẵn sàng đưa vào file model.ipynb")