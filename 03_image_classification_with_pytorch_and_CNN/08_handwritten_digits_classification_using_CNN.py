# Perform standard imports

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Load the MNIST dataset
transforms = transforms.ToTensor()

# Load the training set
train_data = datasets. MNIST(root='../data', train=True, download=True, transform=transforms)
# Load the test set
test_data = datasets. MNIST(root='../data', train=False, download=True, transform=transforms)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)#start with small size #intotal 60000 images 28*28*1, break into patches
test_loader = DataLoader(test_data, batch_size=10, shuffle=True)

# Breaking down the convolutional layers for illustration purpose
#input Image---> conv1 ----> Relu ---> pooling --->conv2--->Relu
# Define layer1 - 1 color channel, 6 filters (output channels), kernel size 3 and stride 1, no padding.
conv1 = nn.Conv2d(1,6,3,1)
# Define layer2 - 6 channel, 16 filters (output channels) , kernel size 3 and stride 1, no padding.
conv2 = nn.Conv2d(6,16,3,1)

# Grab the first MNIST record
X_train = train_data[0][0]
y_train = train_data[0][1]
print(X_train.shape)
print(y_train)
#We need to creat a 4 dimensional tensor to be passed into the model, so it will be something like 1,1,28,28
x = X_train.view(1,1,28,28)
print(x.shape)
# Perform the first convolution/activation
x = F.relu(conv1(x))
# Run the first pooling layer set the kernel size 2 with stride length 2.
x = F.max_pool2d(x,2,2)
print(x.shape)
# Perform the second convolution/activation
x = F.relu(conv2(x))
print(x.shape)
# Run the second pooling layer
x = F.max_pool2d(x,2,2)
# Flatten the data
x = x.view(-1,5*5*16)
print(x.shape)
# Create the actual model class
#input Image---> conv1 ----> Relu ---> pooling --->conv2--->Relu--->fc1--->relu-->fc2--->relu--->fc3--->softmax
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
# Initiate our model and print out the parameters
model = CNN1()
print(model)
# Define loss function & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Train the model
import time
start_time = time.time()
epochs = 3

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

        if b%3000 == 0:
            print(f'epoch: {i} batch: {b} loss: {loss.item()} accuracy: {trn_corr.item() * 100/ (10 * b)} %')

    # Run the testing batches
    with torch.no_grad():
        for b,(X_test, y_test) in enumerate(test_loader):
            y_val = model(X_test)

            predicted = torch.max(y_val.data,1)[1]
            tst_corr +=(predicted == y_test).sum()
    print(f'Test accuracy:{tst_corr.item()*100/(len(test_data))}%')

# Evaluate the model
x =2019
plt.figure(figsize=(1,1))
plt.imshow(test_data[x][0].reshape(28,28))