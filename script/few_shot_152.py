import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import Flowers102
from torch.utils.data import DataLoader
import torchvision.models as models
import numpy as np

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
batch_size = 64  # You can adjust the batch size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize a pre-trained ResNet-50 model and move it to the GPU
feature_extractor = models.resnet152(pretrained=True)
feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-2]).to(device)  # Remove the classification layer

# Define the prototype-based few-shot learner
class FewShotLearner(nn.Module):
    def __init__(self, feature_extractor, num_classes):
        super(FewShotLearner, self).__init__()
        self.feature_extractor = feature_extractor
        self.num_classes = num_classes
        self.out_size = self.calculate_output_size()
        self.prototypes = nn.Parameter(torch.randn(num_classes, self.out_size).to(device))

    def calculate_output_size(self):
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            features = self.feature_extractor(dummy_input)
            return features.view(features.size(0), -1).size(1)

    def forward(self, x):
        x = x.to(device)
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        dists = torch.cdist(features, self.prototypes)
        return -dists

# Create a few-shot learner
num_classes = 102
few_shot_learner = FewShotLearner(feature_extractor, num_classes)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(few_shot_learner.parameters(), lr=0.001, momentum=0.9)

# Test the model on the test set
# Early stopping parameters
patience = 5  # Number of epochs with no improvement to wait
best_val_accuracy = 0.0
early_stopping_counter = 0

# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    few_shot_learner.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = few_shot_learner(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    few_shot_learner.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = few_shot_learner(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = 100 * correct / total

    print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= patience:
        print('Early stopping - no improvement in validation accuracy for {} epochs.'.format(patience))
        break
    
few_shot_learner.eval()
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = few_shot_learner(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_loss /= len(test_loader)
test_accuracy = 100 * correct / total

print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')