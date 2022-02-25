import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt

transform = transforms.ToTensor()

train_data = datasets.FashionMNIST(root='../data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='../data', train=False, download=True, transform=transform)
class_names = ['T-shirt','Trouser','Sweater','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Boot']

# Create data loaders
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=True)

# Examine a batch of images
X_train, y_train = next(iter(train_loader))

print(X_train.shape)
print(y_train)


# Use DataLoader, make_grid and matplotlib to display the first batch of 10 images.
def show_images(images, nmax=10):
    fig, ax = plt.subplots(figsize=(2, 5))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid((images.detach()[:nmax]), nrow=5).permute(1, 2, 0))
    plt.show()

show_images(X_train)
# display the labels as well

print(y_train[0])
# Downsampling
# If a 28x28 image is passed through a Convolutional layer using a 5x5 filter, a step size of 1, and no padding,

# create the conv layer and pass in one data sample as input, then printout the resulting matrix size
conv1 = nn.Conv2d(1,1,5,1)
x = X_train[0].view(1,1,28,28)
x = F.relu(conv1(x))
print(x.shape)
# If the sample from question 3 is then passed through a 2x2 MaxPooling layer
# create the pooling layer and pass in one data sample as input, then printout the resulting matrix size
x = F.max_pool2d(x,2,2)
print(x.shape)


# Define a convolutional neural network

# Define a CNN model that can be trained on the Fashion-MNIST dataset.
class CNN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)

        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X,2,2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X,2,2)
        X = X.view(-1,5*5*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.log_softmax(self.fc3(X),dim=1)
        return X
# The model should contain two convolutional layers, two pooling layers, and two fully connected layers.
# You can use any number of neurons per layer so long as the model takes in a 28x28 image and returns an output of 10.
# and then printout the count of parameters of your model
model = CNN1()
print(model)


# Define loss function & optimizer
# Define a loss function called "criterion" and an optimizer called "optimizer".
# You can use any loss functions and optimizer you want,
# although we used Cross Entropy Loss and Adam (learning rate of 0.001) respectively.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Train and test the model
# try with any epochs you want
# and printout some interim results
import time
start_time = time.time()
epochs = 10

for i in range(epochs):
    trn_corr = 0
    tst_corr = 0

    # Run the training batches
    for b, (X_train, y_train) in enumerate(train_loader):
        b+=1

        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)

        predicted = torch.max(y_pred, 1)[1] #provide the predicted digit
        batch_corr = (predicted == y_train).sum()
        trn_corr+=batch_corr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if b%1000 == 0:
            print(f'epoch: {i} batch: {b} loss: {loss.item()} accuracy: {trn_corr.item() * 100/ (10 * b)} %')
# Remember, always experiment with different architecture and different hyper-parameters, such as
# different activation function, different loss function, different optimizer with different learning rate
# different size of convolutional kernels, and different combination of convolutional layers/pooling layers/FC layers
# to make the best combination for solving your problem in real world

