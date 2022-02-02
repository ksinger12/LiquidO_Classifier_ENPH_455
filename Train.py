import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchviz import make_dot
import torchvision.models as models

import torchvision.transforms as transforms
import numpy as np

import copy
import random
import time
import os

import VGG
from LearningRateFinder import LRFinder

'''
The example I am basing this code off of can be found here: https://colab.research.google.com/github/bentrevett/pytorch-image-classification/blob/master/4_vgg.ipynb
with the strategy stemming from here: https://arxiv.org/abs/1506.01186

The one thing to note is that model used pre-trained models, meaning their weightings did not start from scratch. 
This mainly speeds up the system and at this point (Dec 17th, 2021) is not a luxury I have.
Doing this may be something I want to look to do in the future
'''

PATH = './images/'
OUTPUT_PATH = './trained_model.pth'
data_dir = PATH

# Constants
batch_size = 128  # can be increased / should be with GPU
OUTPUT_DIM = 3  # number of classes
classes = ('Electron', 'Positron', 'Gamma Ray')

# Setting seed - ensures reproducible results
SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

model = VGG.VGG16()

#make_dot(model, params=dict(list(model.named_parameters()))).render("cnn_torchviz", format="png")

# dowloading pre-trained VGG-16 batch normalized model trained on the ImageNet dataset (meaning output is 1000 classes)
# had certification issue. Solved using first response:
# https://stackoverflow.com/questions/50236117/scraping-ssl-certificate-verify-failed-error-for-http-en-wikipedia-org
pretrained_model = models.vgg16_bn(pretrained=True)

# making last layer of pre-trained model match my output dimensions (3 for 3 classes)
IN_FEATURES = pretrained_model.classifier[-1].in_features
final_fc = nn.Linear(IN_FEATURES, OUTPUT_DIM)
pretrained_model.classifier[-1] = final_fc

# The only issue with using the pre-trained model in this case is that it only gives the final output layer, not the
# intermediate like our model. Instead, we will load the parameters of the pre-trained model in
model.load_state_dict(pretrained_model.state_dict())

'''
Note about pre-trained models:
All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of 
shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] 
and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
'''

# Defining the transforms to apply to the images loaded in
# The size, means, and standard deviations refer to what the pre-trained model was used to align with their features
# This prevents conflicts between our data and the data the model was trained on
pretrained_size = 224
pretrained_means = [0.485, 0.456, 0.406]
pretrained_stds= [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
                           transforms.Resize(pretrained_size),
                           transforms.RandomRotation(5),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=pretrained_means,
                                                std=pretrained_stds)
                       ])

test_transforms = transforms.Compose([
                           transforms.Resize(pretrained_size),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=pretrained_means,
                                                std=pretrained_stds)
])

# Loading in the data and setting up the classes
# To obtain data from the server, use: https://stackoverflow.com/questions/30553428/copying-files-from-server-to-local-computer-using-ssh
# Shows how to copy from one server to local. I will need to construct a folder of generated files.

# To add data from server based on my setup:
# scp -r kylesinger@neutrino.phy.queensu.ca:/home/kylesinger/simulation/build/images/ /Users/kylesinger/Desktop/Everything/5th_Year/455/LiquidO_Classifier_ENPH_455


# Pass transforms in here, then run the next cell to see how the transforms look
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
valid_loader= torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

data_iter = iter(test_loader)

# Uncomment for GPU:
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# torch.cuda.device_count()  # print 1
# torch.cuda.set_per_process_memory_fraction(0.5, 0)

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
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


EPOCHS = 20

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

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')


model.load_state_dict(torch.load('VGG-16_trained_model.pt'))
test_loss, test_acc = VGG.evaluate(model, test_loader, criterion, device)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

# put in home directory for .bashrc in neturino server:

# environment vairiables for GPU
# export PATH=/usr/local/cuda/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# export CUDA_HOME=/usr/local/cuda

# what's running on GPUs right now: - use to check if program is running on GPU
# nvidia-smi

