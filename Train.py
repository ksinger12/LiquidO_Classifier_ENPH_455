import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
# from torchviz import make_dot
import torchvision.models as models

import torchvision.transforms as transforms
import numpy as np

import copy
import random
import time

import VGG
from LearningRateFinder import LRFinder

'''
For tighter GPU restrictions:
import os

Uncomment for GPU:
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.cuda.device_count()  # print 1
torch.cuda.set_per_process_memory_fraction(0.5, 0)
'''

PATH = './images/'
OUTPUT_PATH = './trained_model.pth'
data_dir = PATH

# Constants
batch_size = 64
classes = ('Electron', 'Positron', 'Gamma Ray')
OUTPUT_DIM = len(classes)
EPOCHS = 20


# Setting seed - ensures reproducible results
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

model = VGG.VGG16()

# make_dot(model, params=dict(list(model.named_parameters()))).render("cnn_torchviz", format="png")

# Downloading pre-trained model, setting last layer to be our class size, and loading in trained parameters
pre_trained_model = models.vgg16_bn(pretrained=True)

IN_FEATURES = pre_trained_model.classifier[-1].in_features
final_fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)
pre_trained_model.classifier[-1] = final_fc

model.load_state_dict(pre_trained_model.state_dict())


# The size, means, and standard deviations refer to what the pre-trained model was used to align with their features
# This prevents conflicts between our data and the data the model was trained on
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

test_transforms = transforms.Compose([
                           transforms.Resize(pre_trained_size),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=pre_trained_mean,
                                                std=pre_trained_std)
])

train_data = torchvision.datasets.ImageFolder(data_dir + 'Train', transform=train_transforms)
test_data = torchvision.datasets.ImageFolder(data_dir + 'Test', transform=test_transforms)

# Validation split (creating validation set)
VALID_RATIO = 0.9

n_train_examples = int(len(train_data) * VALID_RATIO)
n_valid_examples = len(train_data) - n_train_examples

train_data, valid_data = torch.utils.data.random_split(train_data, [n_train_examples, n_valid_examples])

# Making sure validation data uses test transforms
valid_data = copy.deepcopy(valid_data)  # stops the transformations on one set from effecting the other
valid_data.dataset.transform = test_transforms

print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')

train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

data_iter = iter(test_loader)

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

# reset model to initial parameters
model.load_state_dict(torch.load('init_params.pt'))

FOUND_LR = 5e-4
params = [
          {'params': model.features.parameters(), 'lr': FOUND_LR / 10},
          {'params': model.classifier.parameters()}
         ]
optimizer = optim.Adam(params, lr=FOUND_LR)  # Adam algorithm optimizer


# Telling us how long an epoch takes
def epoch_time(start, end):
    elapsed_time = end - start
    elapsed_min = int(elapsed_time / 60)
    elapsed_sec = int(elapsed_time - (elapsed_min * 60))
    return elapsed_min, elapsed_sec


best_valid_loss = float('inf')

for epoch in range(EPOCHS):

    start_time = time.monotonic()

    train_loss, train_acc = VGG.train(model, train_loader, optimizer, criterion, device)
    valid_loss, valid_acc = VGG.evaluate(model, valid_loader, criterion, device)

    # if this is the best validation loss we've seen, save it
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'VGG-16_trained_model.pt')

    end_time = time.monotonic()

    epoch_min, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_min}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')


model.load_state_dict(torch.load('VGG-16_trained_model.pt'))
test_loss, test_acc = VGG.evaluate(model, test_loader, criterion, device)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')
