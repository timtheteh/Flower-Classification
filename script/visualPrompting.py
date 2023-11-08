import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102
import os
from PIL import Image
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm

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


# Load the pretrained ResNet50 model
resnet50 = models.resnet50(pretrained=True)
# Remove the classification head (fully connected layers)
resnet50 = nn.Sequential(*list(resnet50.children())[:-2])


from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN

# Define a function to generate attention masks using a Faster R-CNN model
# def generate_attention_masks(images, model):
#     model.eval()

#     with torch.no_grad():
#         predictions = model(images)

#     attention_masks = []
#     for prediction in predictions:
#         # Extract the region proposals and their scores for each image in the batch
#         proposals = prediction['boxes']
#         scores = prediction['scores']

#         # You can adjust this threshold to select ROIs based on their confidence score
#         threshold = 0.5
#         selected_indices = scores > threshold

#         # Create an attention mask for each image in the batch
#         attention_mask = torch.zeros_like(images[0, 0, :, :])  # Assuming images is a batch of shape (N, C, H, W)
#         attention_mask[proposals[selected_indices, 1], proposals[selected_indices, 0]] = 1.0
#         attention_masks.append(attention_mask)

#     return attention_masks

def generate_attention_masks(images, model):
    model.eval()

    with torch.no_grad():
        predictions = model(images)

    attention_masks = []
    for prediction in predictions:
        # Extract the region proposals and their scores for each image in the batch
        proposals = prediction['boxes']
        scores = prediction['scores']

        # You can adjust this threshold to select ROIs based on their confidence score
        threshold = 0.5
        selected_indices = scores > threshold

        # Convert selected_indices to integers
        selected_indices = selected_indices.nonzero(as_tuple=False).squeeze(dim=1).long()

        # Create an attention mask for each image in the batch
        attention_mask = torch.zeros_like(images[0, 0, :, :])  # Assuming images is a batch of shape (N, C, H, W)
        attention_mask[proposals[selected_indices, 1].long(), proposals[selected_indices, 0].long()] = 1.0  # Explicitly cast to long
        attention_masks.append(attention_mask)

    return attention_masks

# # Define a transformation for the Faster R-CNN model
# transform_rcnn = T.Compose([
#     T.ToPILImage(),
#     T.Resize((224, 224)),
#     T.ToTensor(),
# ])

# Load a pre-trained Faster R-CNN model
# You can choose the architecture you prefer and adjust it accordingly
model_rcnn = fasterrcnn_resnet50_fpn(pretrained=True)

# Set the model to evaluation mode
model_rcnn.eval()



class ResNetWithAttention(nn.Module):
    def __init__(self, resnet, rcnn, num_classes):
        super(ResNetWithAttention, self).__init__()
        self.resnet = resnet
        self.rcnn = rcnn
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)  # Assuming 2048 is the output feature size of the ResNet

    def forward(self, x):
        # Generate attention masks using the Faster R-CNN
        attention_masks = generate_attention_masks(x, self.rcnn)
        # Concatenate the list of attention masks into a single tensor
        attention_mask = torch.stack(attention_masks)
        # Apply the attention mask to the input images
        x = x * attention_mask.unsqueeze(1)  # Ensure dimensions match
        # Pass the modified image through the ResNet
        x = self.resnet(x)
        # Apply global average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # Pass through the classification layer
        x = self.fc(x)
        return x

# Create the combined model
num_classes = 102  # Number of classes in your dataset
combined_model = ResNetWithAttention(resnet50, model_rcnn, num_classes)

# Define the loss function, optimizer, and training loop
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(combined_model.parameters(), lr=0.001)


# Training loop with early stopping
num_epochs = 10
best_val_loss = float('inf')
patience = 3  # Number of epochs to wait for improvement
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
combined_model.to(device)

for epoch in range(num_epochs):
    combined_model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0  
    for inputs, labels in train_loader:
        # print("labels: ", labels)
        # print("labels shape: ", labels.shape)
        # print("labels type: ", labels.dtype)
        optimizer.zero_grad()
        outputs = combined_model(inputs)
        # print("outputs: ", outputs)
        # print("outputs shape: ", outputs.shape)
        # print("labels shape: ", labels.shape)
        # labels = labels.squeeze(1)  # Squeeze to make sure it's a 1D tensor
        # Ensure that labels are of type Long (int64) and in the expected shape
        labels = labels.view(-1)  # Reshape to a 1D tensor if needed
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_accuracy = 100 * train_correct / train_total
    train_loss = 100 * train_loss / len(train_loader)

    # Validation
    combined_model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0  
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = combined_model(inputs)
            labels = labels.view(-1)  # Reshape to a 1D tensor if needed
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    val_loss = 100 * val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f}% | Val Loss: {val_loss:.4f}%")
    print(f"Train Accuracy: {train_accuracy:.4f}% | Val Accuracy: {val_accuracy:.4f}%")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping. No improvement in validation loss.")
            break

# Save the trained model
torch.save(combined_model.state_dict(), 'resnet_with_attention.pth')



# Load the saved model (if not already loaded)
combined_model.load_state_dict(torch.load('resnet_with_attention.pth'))
combined_model.to(device)
combined_model.eval()  # Set the model to evaluation mode

test_loader = DataLoader(test_dataset, batch_size=batch_size)

test_loss = 0.0
test_correct = 0
test_total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = combined_model(inputs)
        labels = labels.view(-1)  # Reshape to a 1D tensor if needed
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_accuracy = 100 * test_correct / test_total
test_loss = 100 * test_loss / len(test_loader)

print(f"Test Accuracy: {test_accuracy}%")
print(f"Test Loss: {test_loss}%")