import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torchvision.models import resnet152, ResNet152_Weights, resnet18
from torchvision.ops import deform_conv2d, DeformConv2d
import numpy as np

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

class MyModel(nn.Module):
	def __init__(self):
		super(MyModel, self).__init__()

		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
		# self.bn1 = nn.BatchNorm2d(64)		
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		
		self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
		# relu
		# maxpool 
		# conv2
		# relu
		# maxpool
		self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
		# relu
		# maxpool 
		self.conv4 = nn.Conv2d(128, 252, kernel_size=3, stride=1, padding=1, bias=False)
		# relu
		# maxpool 

		self.conv5 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1, bias=True)
		self.conv6 = nn.Conv2d(256, 512, kernel_size=5, stride=1, padding=1, bias=True)
		self.conv7 = nn.Conv2d(512, 1024, kernel_size=6, stride=1, padding=1, bias=True)
		self.conv8 = nn.Conv2d(1024, 512, kernel_size=7, stride=1, padding=1, bias=True)
		self.conv9 = nn.Conv2d(512, 252, kernel_size=3, stride=1, padding=1, bias=True)


		self.offset1 = nn.Conv2d(252, out_channels=252, kernel_size=3, stride=1, padding=1, bias=False)
		self.regular_conv = nn.Conv2d(252, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
		
		self.fc1 = nn.Linear(4096, 512)
		self.fc2 = nn.Linear(512, 102)

		self.fc3 = nn.Linear(4032, 672)
		self.fc4 = nn.Linear(672, 102)
	
	def forward(self, x):
		x = self.conv1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		x = self.conv2(x)
		x = self.relu(x)
		x = self.maxpool(x)
		x = self.conv2(x)
		x = self.relu(x)
		x = self.maxpool(x)
		x = self.conv3(x)
		x = self.relu(x)
		x = self.maxpool(x)
		x = self.conv4(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.conv5(x)
		x = self.relu(x)
		x = self.maxpool(x)
		x = self.conv6(x)
		x = self.relu(x)
		x = self.maxpool(x)
		x = self.conv7(x)
		x = self.relu(x)
		x = self.maxpool(x)
		x = self.conv8(x)
		x = self.relu(x)
		x = self.maxpool(x)
		x = self.conv9(x)
		x = self.relu(x)
		x = self.maxpool(x)
		#print(x.shape)
		#print(self.offset1)
		#print(self.offset1.weight.shape)
		#print(self.offset1.in_channels)
		# offset = self.offset1(x)
		#print("GOT offset")
		#print(offset.shape)
		#print(x.shape)
		#print(self.regular_conv.weight.shape)
		"""
		x = deform_conv2d(input=x,
												offset=offset,
												weight=self.regular_conv.weight,
												bias=self.regular_conv.bias,
												padding=1,
												mask=None,
												stride=1)
		"""
		#print("DONE deform")
		#print(x.shape)
		x = x.view(x.size(0), -1)
		#print(x.shape)
		# x = self.fc1(x)
		#print(x.shape)
		# x = self.fc2(x)
		x = self.fc3(x)
		x = self.fc4(x)
		return nn.functional.softmax(x, dim=1)


def train_val(model, train_loader, val_loader, name, lr=0.01, gamma=0.9, num_epochs=100):
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
					if epoch <3:
						print(predicted, labels)
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
