import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import VGG
from LearningRateFinder import LRFinder


OUTPUT_DIM = 3
PATH = './images/'
batch_size = 64

pre_trained_size = 224
pre_trained_mean = [0.485, 0.456, 0.406]
pre_trained_std = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
                           transforms.Resize(pre_trained_size),
                           transforms.RandomRotation(5),  # add some shifting
                           transforms.ToTensor(),
                           transforms.Normalize(mean=pre_trained_mean,
                                                std=pre_trained_std)
                       ])

train_data = torchvision.datasets.ImageFolder(PATH + 'Train', transform=train_transforms)
train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)

model = VGG.VGG16()

# Downloading pre-trained model, setting last layer to be our class size, and loading in trained parameters
pre_trained_model = models.vgg16_bn(pretrained=True)

IN_FEATURES = pre_trained_model.classifier[-1].in_features
final_fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)
pre_trained_model.classifier[-1] = final_fc

model.load_state_dict(pre_trained_model.state_dict())

# Training the model
START_LR = 1e-7  # Initial learning rate - super small

optimizer = optim.Adam(model.parameters(), lr=START_LR)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()  # type of cost function -> computes the softmax activation function on suppled predections as well as the loss via negaive log likelihood

model = model.to(device)
criterion = criterion.to(device)

# Running the learning rate finder
END_LR = 10
NUM_ITER = 10

# Retrieves the rate at which we should learn
lr_finder = LRFinder(model, optimizer, criterion, device)
lrs, losses = lr_finder.range_test(train_loader, END_LR, NUM_ITER)

# basically want to find steepest slope with lrs values and the average of the slope. Current way is to observe
# plots of losses vs lrs on a log x-axis
