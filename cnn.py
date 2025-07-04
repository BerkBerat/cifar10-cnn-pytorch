"""
Problem Definition - Classification Problem Using the CIFAR10 Dataset
CIFAR10 - Dataset
CNN - Architecture to be Used
"""
# %% Library
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Optional: Select the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Dataset
def get_data_loaders(batch_size = 64):
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 3 variables for RGB channels
        ])
    
    # Download CIFAR10. Create train and test dataset.
    train_set = torchvision.datasets.CIFAR10(root="./data", train = True, download = True, transform = transform)
    test_set = torchvision.datasets.CIFAR10(root="./data", train = False, download = True, transform = transform)
    
    # DataLoader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = False)
    
    return train_loader,test_loader

# %% Visualize Dataset
def imshow(img):
    # Convert before normalizing data
    img = img / 2 + 0.5 # The inverse of the normalization process
    np_img = img.numpy() # Convert tensor to numpy array
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()
    
def get_sample_images(train_loader):
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    return images, labels

def visualize(n):
    train_loader, test_loader = get_data_loaders()
    images, labels = get_sample_images(train_loader)
    plt.figure()
    for i in range(n):
        plt.subplot(1, n, i+1)
        imshow(images[i])
        plt.title(f"Label:{labels[i].item()}")
        plt.axis("off")
    plt.show()

# %% Build CNN Model
class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, padding = 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(64*8*8, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        """
         First Conv Layer : image(3*32*32) -> conv(32) -> relu(32) -> pool(16)
         Second Conv Layer : conv(16) -> relu(16) -> pool(8) -> image(8*8)
         Flatten Layer
         Fully Connected Layer 1 : fc1 -> relu -> dropout
         Output Layer : fc2 -> output
        """
        x = self.pool(self.relu(self.conv1(x))) # First conv block
        x = self.pool(self.relu(self.conv2(x))) # Second conv block
        x = x.view(-1, 64*8*8) # Flatten Layer
        x = self.dropout(self.relu(self.fc1(x))) # FC Layer
        x = self.fc2(x) # Output layer
        
        return x

# Define Loss Function and Optimizer
define_loss_and_optimizer = lambda model:(
    nn.CrossEntropyLoss(), # For MultiClass Classification Problem
    optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9) # SGD for Optimizer
    )

# %% Train
def train_model(model, train_loader, criterion, optimizer, epochs = 5):
    model.train()
    train_losses = []
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch: {epoch +1}/{epochs}, Loss:{avg_loss:.5f}")
        
    # Loss Graph
    plt.figure()
    plt.plot(range(1,epochs+1),train_losses,marker = "o", linestyle = "-", label = "Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Traning Loss")
    plt.legend()
    plt.show()

# %% Test
def test_model(model, test_loader, dataset_type):
    model.eval()
    correct = 0 
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    print(f"{dataset_type} Accuracy: {100* correct / total} %")

# %% Main
if __name__ == "__main__":
    train_loader, test_loader = get_data_loaders() # Dataset Loading
    visualize(3) # Data Visulization
    
    # Training
    model = CNN().to(device)
    criterion, optimizer = define_loss_and_optimizer(model)
    train_model(model, train_loader, criterion, optimizer, epochs = 10)
    
    # Test
    test_model(model, test_loader, dataset_type = "test") # Test Accuracy: 62.57 %
    test_model(model, train_loader, dataset_type = "train") # Train Accuracy: 65.506 %