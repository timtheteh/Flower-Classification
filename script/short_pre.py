import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torchvision.models import  resnet18
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102
import os
from torchvision.ops import deform_conv2d, DeformConv2d
import numpy as np
import time

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)


# Define data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet statistics
])

# Download the dataset (training split)
train_dataset = Flowers102(
    root='./data',  # The root directory where the dataset will be saved
    split='train',   # 'train' for the training set, 'test' for the test set
    transform=transform,  # Apply the defined transformation
    download=True  # Download if not already present
)

# Download the dataset (validation split)
val_dataset = Flowers102(
    root='./data',  # The root directory where the dataset will be saved
    split='val',   # 'train' for the training set, 'test' for the test set
    transform=transform,  # Apply the defined transformation
    download=True  # Download if not already present
)

# Download the dataset (test split)
test_dataset = Flowers102(
    root='./data',  # The root directory where the dataset will be saved
    split='test',   # 'train' for the training set, 'test' for the test set
    transform=transform,  # Apply the defined transformation
    download=True  # Download if not already present
)


# Create data loaders
batch_size = 320
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


class MyModel(nn.Module):
    def __init__(self, num_classes=1000):
        super(MyModel, self).__init__()
        pretrained = resnet18(pretrained=True)

        # Define the layers of ResNet-18
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight.data = pretrained.conv1.weight.data
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn1.weight.data = pretrained.bn1.weight.data
        
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        # self.layer2 = self._make_layer(64, 128, 2, stride=2)
        # self.layer3 = self._make_layer(128, 256, 2, stride=2)
        # self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512, num_classes)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        kern1 = 3
        pad1 = 1
        self.offset = nn.Conv2d(in_channels, out_channels=2*kern1*kern1, kernel_size=(kern1,kern1), stride=stride, padding=(pad1, pad1), bias=False)

        self.conv1 = DeformConv2d(in_channels, out_channels, kernel_size=(kern1,kern1), stride=stride, padding=(pad1, pad1), bias=False)
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        a = self.offset(x)
        x = self.conv1(x, a)

        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)

        return x



model = MyModel(num_classes=102)



# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.0025, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

best_model_weights = model.state_dict()
best_loss = float('inf')  # Initialize with a high value
patience = 3  # Number of epochs to wait before early stopping
since = time.time()
num_epochs = 40
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
deformable_groups = 2
kernel_size =3


# Initialize early stopping parameters
early_stopping_patience = 3
best_validation_loss = float("inf")
best_epoch = 0
no_improvement_count = 0

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
    validation_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            validation_loss += criterion(outputs, labels)

    # Print validation accuracy and loss for each epoch
    validation_accuracy = 100 * correct / total
    print(f"Validation Accuracy: {validation_accuracy}%")
    print(f"Validation Loss: {validation_loss / len(val_loader)}")

    # Check for early stopping
    if validation_loss < best_validation_loss:
        best_validation_loss = validation_loss
        best_epoch = epoch
        no_improvement_count = 0
        # Save the model checkpoint
        torch.save(model.state_dict(), 'deform_flowers102.pth')
    else:
        no_improvement_count += 1
        if no_improvement_count >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1} (Best epoch: {best_epoch + 1})")
            break  # Stop training

# Save the final trained model
torch.save(model.state_dict(), 'deform_flowers102_final.pth')


# Load the best model weights
model.load_state_dict(best_model_weights)

# Test the model on the test dataset
model.eval()
test_corrects = 0
test_loss = 0.0  # Initialize the test loss

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        test_corrects += torch.sum(preds == labels.data)
        
        # Compute the test loss
        loss = criterion(outputs, labels)
        test_loss += loss.item() * inputs.size(0)

test_acc = test_corrects.double() / len(test_loader.dataset)
test_loss = test_loss / len(test_loader)  # Calculate the average test loss

print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_acc:.4f}')
