import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import VGG
from LearningRateFinder import LRFinder


"""
Script used to find the learning rate for the created VGG16 model with the given image set. 
"""

OUTPUT_DIM = 3
PATH = './images/'
BATCH_SIZE = 64

# Initialized parameter details
pre_trained_size = 224  # 224x224 images
pre_trained_mean = [0.485, 0.456, 0.406]
pre_trained_std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
                           transforms.Resize(pre_trained_size),
                           transforms.RandomRotation(5),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=pre_trained_mean,
                                                std=pre_trained_std)
                       ])

train_data = torchvision.datasets.ImageFolder(PATH + 'Train', transform=train_transforms)
train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)

model = VGG.VGG16()

# Downloading pre-trained model, setting last layer to be our class size, and loading in trained parameters
pre_trained_model = models.vgg16_bn(pretrained=True)

features = pre_trained_model.classifier[-1].in_features
last_fc_layer = nn.Linear(features, OUTPUT_DIM)
pre_trained_model.classifier[-1] = last_fc_layer

model.load_state_dict(pre_trained_model.state_dict())

# Training the model
optimizer = optim.Adam(model.parameters(), lr=1e-7)  # Initial learning rate: 1e-7
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()

model = model.to(device)
criterion = criterion.to(device)

# Retrieves the rate at which we should learn
lr_finder = LRFinder(model, optimizer, criterion, device)
learning_rates, losses = lr_finder.range_test(train_loader, 10, 10)  # When to end learning rate & # of iterations

# basically want to find steepest slope with learning_rates values and the average of the slope. Current way is to
# observe plots of losses vs lrs on a log x-axis
