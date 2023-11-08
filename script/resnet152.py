import torch
import time
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102
import os

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
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Define the ResNet-152 model
model = models.resnet152(pretrained=True)

# Freeze all layers except the final fully connected layer
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad = True

# Modify the output layer to match the number of classes in the dataset (102 for Oxford Flowers)
model.fc = nn.Linear(model.fc.in_features, 102)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Initialize early stopping parameters
early_stopping_patience = 3 # Number of consecutive epochs without improvement to wait before stopping
best_validation_loss = float("inf")
best_epoch = 0
no_improvement_count = 0

# Training loop
num_epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
for epoch in range(num_epochs):
    start = time.time()
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
    end = time.time()
    # Print training loss for each epoch
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, Time: {end-start}")

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
        torch.save(model.state_dict(), 'resnet50_flowers102.pth')
    else:
        no_improvement_count += 1
        if no_improvement_count >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1} (Best epoch: {best_epoch + 1})")
            break  # Stop training

# Save the final trained model
torch.save(model.state_dict(), 'resnet50_flowers102_final.pth')

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
