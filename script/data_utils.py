
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102
import numpy as np

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)


# Define data augmentation transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),  # Randomly crop and resize
        transforms.RandomHorizontalFlip(),  # Randomly flip horizontally
        transforms.RandomRotation(30),  # Randomly rotate by up to 30 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Adjust brightness, contrast, saturation, and hue
        transforms.RandomGrayscale(p=0.2),  # Convert to grayscale with a 20% probability
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random affine transformation
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),  # Random perspective transformation
        transforms.RandomVerticalFlip(),  # Randomly flip vertically
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),  # Resize for validation and test
        transforms.CenterCrop(224),  # Center crop for validation and test
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),  # Resize for validation and test
        transforms.CenterCrop(224),  # Center crop for validation and test
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

# Apply the data augmentation transformations to your datasets
train_dataset = Flowers102(
    root='./data',
    split='train',
    transform=data_transforms['train'],  # Use data augmentation for training
    download=True
)

val_dataset = Flowers102(
    root='./data',
    split='val',
    transform=data_transforms['val'],  # Use validation data transformations
    download=True
)

test_dataset = Flowers102(
    root='./data',
    split='test',
    transform=data_transforms['test'],  # Use test data transformations
    download=True
)

# Create data loaders for training, validation, and test sets

def load_data(batch_sz=128):
	# batch_size = 128  # You can adjust the batch size
	train_loader = DataLoader(train_dataset, batch_size=batch_sz, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=batch_sz, shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=batch_sz, shuffle=False)
	return train_loader, val_loader, test_loader


