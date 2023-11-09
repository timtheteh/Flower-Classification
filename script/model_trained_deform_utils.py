
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torchvision.models import resnet152, ResNet152_Weights, resnet18
from torchvision.ops import deform_conv2d, DeformConv2d
import numpy as np
import time

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)


class MyModel(nn.Module):
	def __init__(self, layer):
		super(MyModel, self).__init__()
		self.layer = layer
		pretrained = resnet18(pretrained=True)
		features = list(pretrained.children())

		if self.layer == 2:
			new_pretrained = nn.Sequential(*features[:-4], *features[-2:])
		elif self.layer == 3:
			new_pretrained = nn.Sequential(*features[:-3], *features[-2:])
		elif self.layer == 4:
			new_pretrained = pretrained

		
		
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		tmp = {1: 64, 2: 128, 3: 256, 4: 512}
		self.fc = nn.Linear(tmp[self.layer], 102)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		if self.layer == 1:
			x = self.layer1(x)
		elif self.layer == 2:
			x = self.layer1(x)
			x = self.layer2(x)
		elif self.layer == 3:
			x = self.layer1(x)
			x = self.layer2(x)
			x = self.layer3(x)
		elif self.layer == 4:
			x = self.layer1(x)
			x = self.layer2(x)
			x = self.layer3(x)
			x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x

class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, in_channels, out_channels, stride=1):
			super(BasicBlock, self).__init__()
			self.stride = stride

			self.kern1 = 3
			self.pad1 = 1
			self.offset = nn.Conv2d(in_channels, out_channels=2*self.kern1*self.kern1, kernel_size=(self.kern1,self.kern1), stride=stride, padding=(self.pad1, self.pad1), bias=False)
			nn.init.constant_(self.offset.weight, 0.)
			self.regular_conv = nn.Conv2d(in_channels, out_channels=2*self.kern1*self.kern1, kernel_size=(self.kern1,self.kern1), stride=stride, padding=(self.pad1, self.pad1), bias=False)
			# self.conv1 = DeformConv2d(in_channels, out_channels, kernel_size=(kern1,kern1), stride=stride, padding=(pad1, pad1), bias=False)
			
			# self.bn1 = nn.BatchNorm2d(out_channels)
			self.bn1 = nn.BatchNorm2d(18)
			self.relu = nn.ReLU(inplace=True)
			
			# self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
			self.conv2 = nn.Conv2d(18, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
			self.bn2 = nn.BatchNorm2d(out_channels)
			
			if stride != 1 or in_channels != out_channels:
					self.downsample = nn.Sequential(
							nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
							nn.BatchNorm2d(out_channels)
					)
			else:
					self.downsample = None

	def forward(self, x):
		# print("shape start: ", x.shape)
		residual = x

		a = self.offset(x)
		x = deform_conv2d(input=x,
												offset=a,
												weight=self.regular_conv.weight,
												bias=self.regular_conv.bias,
												padding=self.pad1,
												mask=None,
												stride=self.stride)
		# x = self.conv1(x, a)
		# print("shape after deform: ", x.shape)

		x = self.bn1(x)
		x = self.relu(x)

		x = self.conv2(x)
		x = self.bn2(x)

		if self.downsample is not None:
				residual = self.downsample(residual)

		x += residual
		x = self.relu(x)

		return x


def train_val(model, train_loader, val_loader, name, lr=0.01, gamma=0.9, num_epochs=40):
	# Loss and optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
	scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
	# Initialize early stopping parameters
	early_stopping_patience = 5
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
				save_path = name + '.pth'
				torch.save(model.state_dict(), save_path)
			else:
				no_improvement_count += 1
				if no_improvement_count >= early_stopping_patience:
					print(f"Early stopping at epoch {epoch + 1} (Best epoch: {best_epoch + 1})")
					break  # Stop training
			scheduler.step()


def test(model, name, test_loader):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	criterion = nn.CrossEntropyLoss()
  
	# Load the best model weights
	save_path = name + '.pth'
	best_model_weights = torch.load(save_path)
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







