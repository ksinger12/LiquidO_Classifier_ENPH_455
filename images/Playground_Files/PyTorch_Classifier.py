# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torchvision
from torchviz import make_dot


PATH = './'
OUTPUT_PATH = './trained_model.pth'
data_dir = PATH
batch_size = 3
classes = ('Electron', 'Positron', 'Gamma Ray')

# transform = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()])
# TODO: Define transforms for the training data and testing data
train_transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(255), torchvision.transforms.CenterCrop(224), torchvision.transforms.ToTensor()])

test_transforms = torchvision.transforms.Compose([torchvision.transforms.Resize(255), torchvision.transforms.CenterCrop(224), torchvision.transforms.ToTensor()])

# Pass transforms in here, then run the next cell to see how the transforms look
train_data = torchvision.datasets.ImageFolder(data_dir + 'Train', transform=train_transforms)
test_data = torchvision.datasets.ImageFolder(data_dir + 'Test', transform=test_transforms)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

data_iter = iter(test_loader)

# input[channel] = (input[channel] - mean[channel]) / std[channel] #normalizing image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Convolutional neural network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Is RGB necessary? should would gray scale be better
        self.conv1 = nn.Conv2d(3, 6, 5)  # in_channels=3 since image is RGB, out_channels=6 -> output will have 6 feature maps, kernel_size=5 -> size of out kernenl (feature detector)
        self.pool = nn.MaxPool2d(2, 2)  # down-sampling dimensions of our image to allow for assumptions to be made about features in certain regions -> changing image into 2x2 while retaining important features
        self.conv2 = nn.Conv2d(6, 16, 5)  # in_channels must be the same as the out_channels of previous convolution
        # why is the above output after flattening 44944 ?
        self.fc1 = nn.Linear(44944, 120)  # fully connected layer
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, batch_size)

        '''
        See AlexNet Elec474 colab
        conv2d 3,64,11, stride=4, padding=2
        maxpool2d 3, stride=2
        conv2d 64,192,5, padding=2
        maxpool2d
        ...
        '''

    def forward(self, x):
        x = self.pool(f.relu(self.conv1(x)))  # relu is used as an activation function -> converts the sum of inputs into a single output -> introduces non-linearity (without it, we have a linear function that can't predict well)
        x = self.pool(f.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Training the network
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)  # something wrong here
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
torch.save(net.state_dict(), OUTPUT_PATH)

dataiter = iter(test_loader)
images, labels = dataiter.next()

# plt.imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(3)))

# Load back the trained model
net = Net()
net.load_state_dict(torch.load(OUTPUT_PATH))

# Algorithm predicts
outputs = net(images)

#  Higher the energy for a class, the more the network thinks that the image is of the particular class
# This is the highest energy
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(3)))

#make_dot(outputs, params=dict(list(Net.named_parameters(net)))).render("cnn_torchviz", format="png")
