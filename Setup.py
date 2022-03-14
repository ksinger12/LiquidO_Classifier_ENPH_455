import torch
import torch.nn as nn
import numpy as np
import random
import torchvision.models as models

# Constants
PATH = './images/'
OUTPUT_PATH = './trained_model.pth'
EPOCHS = 20
INITIAL_LR = 1e-7  # Initial learning rate - super small
LEARNING_RATE = 5e-4  # found as average from lrs such that losses changes minimally (plot losses vs. lrs to see)
VALIDATION_SET_RATIO = 0.9
BATCH_SIZE = 64
classes = ('Electron', 'Positron', 'Gamma Ray')

# Setting seed - ensures reproducible results
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

pre_trained_model = models.vgg16_bn(pretrained=True)

features = pre_trained_model.classifier[-1].in_features
final_fc = nn.Linear(features, len(classes))
pre_trained_model.classifier[-1] = final_fc

# The size, means, and standard deviations refer to what the pre-trained model was used to align with their features
pre_trained_size = 224  # Image size: 224x224
pre_trained_mean = [0.485, 0.456, 0.406]
pre_trained_std = [0.229, 0.224, 0.225]


def epoch_time(start, end):
    """
    Length of time for an epoch to run
    :param start: Start time
    :param end: End time
    :return: Amount of minutes and seconds passed
    """
    elapsed_time = end - start
    elapsed_min = int(elapsed_time / 60)
    elapsed_sec = int(elapsed_time - (elapsed_min * 60))
    return elapsed_min, elapsed_sec
