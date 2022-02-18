# Import torch and NumPy

import torch
import numpy as np
# Set the random seed for NumPy and PyTorch both to "42"
#   This allows us to share the same "random" results.
np.random.seed(42)
torch.manual_seed(42)

# Create a NumPy array called "arr" that contains 6 random integers between 0 (inclusive) and 5 (exclusive)
arr = np.random.randint(0,5,6)

# Create a tensor "x" from the array above
x = torch.tensor(arr)

# Change the dtype of x from 'int32' to 'int64'
x = x.type(torch.LongTensor)
print (x.dtype)

# Reshape x into a 3x2 tensor
x=x.reshape(3,2)
print(x)

# Return the left-hand column of tensor x
print(x[:,0])

# Without changing x, return a tensor of square values of x
print(torch.square(x))

# Create a tensor "y" with the same number of elements as x, that can be matrix-multiplied with x
y = torch.ones(2,3,dtype=int)
print (y)
# Find the matrix product of x and y
print (x.mm(y))

# Create a Simple linear model using torch.nn
# the model will take 1000 input and output 20 multi-class classification results.
# the model will have 3 hidden layers which include 200, 120, 60 respectively.
import torch.nn as nn
import torch.nn.functional as F
class MultiLayerPercepton(nn.Module):
    def __init__(self, in_sz=1000, out_sz=20, layers=[200,120,60]):
        super().__init__()

        self.fc1 = nn.Linear(in_sz, layers[0])
        self.fc2 = nn.Linear(layers[0],layers[1])
        self.fc2 = nn.Linear(layers[1],layers[2])
        self.fc4 = nn.Linear(layers[2], out_sz)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = F.log_softmax (self.fc4(X), dim=1)
        # it is designed for multi-class classification and you need to specify the dimension that the calculation is on
        return X



# initiate the model and printout the number of parameters

model = MultiLayerPercepton()
print (model)

for param in model.parameters():
    print(param.numel())