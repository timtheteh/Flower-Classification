
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torchvision.models import resnet152, ResNet152_Weights, resnet18
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102
import os
import numpy as np
import time

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
batch_size = 256  # You can adjust the batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# Define the ResNet18 architecture
model = models.resnet18(pretrained=True)

# Freeze all layers except the final fully connected layer
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad = True

# Modify the output layer to match the number of classes in the dataset (102 for Oxford Flowers)
model.fc = nn.Linear(model.fc.in_features, 102)

# Define loss function and optimizer
#criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

# Training loop
num_epochs = 40
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Print training loss for each epoch
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print validation accuracy for each epoch
    print(f"Validation Accuracy: {100 * correct / total}%")

# Save the trained model
torch.save(model.state_dict(), 'resnet18_pretrained_40_flowers102.pth')



# Load the saved model (if not already loaded)
model.load_state_dict(torch.load('resnet50_flowers102.pth'))
model.to(device)
model.eval()  # Set the model to evaluation mode

test_loader = DataLoader(test_dataset, batch_size=batch_size)

test_loss = 0.0
test_correct = 0
test_total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_accuracy = 100 * test_correct / test_total
test_loss /= len(test_loader)

print(f"Test Accuracy: {test_accuracy}%")
print(f"Test Loss: {test_loss}")
     
