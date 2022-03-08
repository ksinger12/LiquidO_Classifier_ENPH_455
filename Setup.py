import torch
import torch.nn as nn
import numpy as np
import random
import torchvision.models as models

PATH = './images/'
OUTPUT_PATH = './trained_model.pth'
EPOCHS = 20
START_LR = 1e-7  # Initial learning rate - super small
FOUND_LR = 5e-4  # found as average from lrs such that losses changes minimally (plot losses vs. lrs to see)
batch_size = 64
classes = ('Electron', 'Positron', 'Gamma Ray')

# Setting seed - ensures reproducible results
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

pre_trained_model = models.vgg16_bn(pretrained=True)

features = pre_trained_model.classifier[-1].in_features
final_fc = nn.Linear(features, len(classes))
pre_trained_model.classifier[-1] = final_fc

# The size, means, and standard deviations refer to what the pre-trained model was used to align with their features
# This prevents conflicts between our data and the data the model was trained on
pre_trained_size = 224
pre_trained_mean = [0.485, 0.456, 0.406]
pre_trained_std = [0.229, 0.224, 0.225]