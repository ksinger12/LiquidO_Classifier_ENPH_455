import torch.optim as optim
import torchvision
# from torchviz import make_dot

import torchvision.transforms as transforms

import copy
import time

import VGG
from Setup import *

'''
# For tighter GPU restrictions:
import os

# Uncomment for GPU:
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.cuda.device_count()  # print 1
torch.cuda.set_per_process_memory_fraction(0.5, 0)
'''

classes = ('Electron', 'Positron', 'Gamma Ray')

model = VGG.VGG16()
model.load_state_dict(pre_trained_model.state_dict())

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

train_data = torchvision.datasets.ImageFolder(PATH + 'Train', transform=train_transforms)
test_data = torchvision.datasets.ImageFolder(PATH + 'Test', transform=test_transforms)

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
optimizer = optim.Adam(model.parameters(), lr=START_LR)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()  # type of cost function -> computes the softmax activation function on suppled predections as well as the loss via negaive log likelihood

model = model.to(device)
criterion = criterion.to(device)

# reset model to initial parameters
model.load_state_dict(torch.load('init_params.pt'))

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
