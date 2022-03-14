import torch.optim as optim
import torchvision
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

number_training_examples = int(len(train_data) * VALIDATION_SET_RATIO)
number_validation_examples = len(train_data) - number_training_examples

train_data, valid_data = \
    torch.utils.data.random_split(train_data, [number_training_examples, number_validation_examples])
valid_data = copy.deepcopy(valid_data)
valid_data.dataset.transform = test_transforms

print(f'Number of training examples: {len(train_data)}')
print(f'Number of validation examples: {len(valid_data)}')
print(f'Number of testing examples: {len(test_data)}')

train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)

data_iter = iter(test_loader)

# Training the model
optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

model = model.to(device)
model.load_state_dict(torch.load('init_params.pt'))

params = [
          {'params': model.features.parameters(), 'lr': LEARNING_RATE / 10},
          {'params': model.classifier.parameters()}
         ]
optimizer = optim.Adam(params, lr=LEARNING_RATE)  # Adam algorithm optimizer

min_validation_loss = float('inf')

for epoch in range(EPOCHS):
    start_time = time.monotonic()

    train_loss, train_accuracy = VGG.train(model, train_loader, optimizer, criterion, device)
    valid_loss, valid_accuracy = VGG.evaluate(model, valid_loader, criterion, device)

    if valid_loss < min_validation_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'VGG16_trained_model.pt')

    end_time = time.monotonic()
    minutes, secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {minutes}m {secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Accuracy: {train_accuracy * 100:.2f}%')
    print(f'\t Validation Loss: {valid_loss:.3f} |  Validation Accuracy: {valid_accuracy * 100:.2f}%')
