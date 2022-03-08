import torch
import torch.nn as nn


class VGG16(nn.Module):

    kernel_size = 7

    def __init__(self, output_dim=3):
        super().__init__()

        # VGG-16 configuration, number denotes convolution layer, M denotes max pooling layers
        vgg16_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

        self.features = get_vgg_layers(vgg16_config, batch_norm=True)
        self.avgpool = nn.AdaptiveAvgPool2d(self.kernel_size)
        self.classifier = nn.Sequential(
            nn.Linear(512 * self.kernel_size * self.kernel_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)

        return x, h


def get_vgg_layers(config, batch_norm, in_channels=3):
    layers = []

    for c in config:
        assert c == 'M' or isinstance(c, int)
        if c == 'M':
            layers += [nn.MaxPool2d(kernel_size=2)]
        else:
            conv = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv, nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            else:
                layers += [conv, nn.ReLU(inplace=True)]
            in_channels = c

    return nn.Sequential(*layers)


def calculate_accuracy(y_pred, y):
    max_prediction = y_pred.argmax(1, keepdim=True)
    correct = max_prediction.eq(y.view_as(max_prediction)).sum()

    return correct.float() / y.shape[0]


def train(model, iterator, optimizer, criterion, device):
    """
        Using gradient descent to train the models.
        Parameters:
            model -> VGG16 defined model
            iterator -> labelled training images (in batches)
            optimizer -> optimizer object to hold current state & update parameters based on computed gradients
            criterion -> type of cost function -> computes the softmax activation function on supplied predictions as well as the
            loss via negative log likelihood
            device -> cpu or gpu to run code on
    """
    epoch_loss = 0
    epoch_acc = 0

    model.train()  # placing model in train mode

    # Iterating over our data loader returning batches of (image, label)
    for (x, y) in iterator:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()  # clear gradients from last batch
        y_pred, _ = model(x)  # pass images in to get the predictions
        # PyTorch name for cost/loss function -> calculating the loss between predictions and actual labels
        loss = criterion(y_pred, y)
        acc = calculate_accuracy(y_pred, y)  # accuracy between predictions and labels

        loss.backward()  # calculating gradients of each parameter
        optimizer.step()  # updating the parameters

        # updating metrics
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()  # placing model in evaluation mode

    with torch.no_grad():  # not using gradients -> shouldn't calculate them to save time and memory
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)

            # Getting predictions as in the training function, train()
            y_pred, _ = model(x)
            loss = criterion(y_pred, y)
            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)
