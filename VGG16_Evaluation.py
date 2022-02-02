import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as f

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


class VGG16(nn.Module):
    def __init__(self, output_dim=3):
        super().__init__()

        vgg16_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

        # Algorithm setup, convolutions and pooling
        self.features = get_vgg_layers(vgg16_config, batch_norm=True)

        # Kernal size for average pooling
        self.avgpool = nn.AdaptiveAvgPool2d(7)  # Specifies the output size we want, 7x7

        # Fully connected layer
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
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
        h = x.view(x.shape[0], -1)  # flatten x
        x = self.classifier(h)
        return x, h


def get_vgg_layers(config, batch_norm):
    layers = []
    in_channels = 3

    for c in config:
        assert c == 'M' or isinstance(c, int)
        if c == 'M':
            layers += [nn.MaxPool2d(kernel_size=2)]
        else:
            conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = c

    return nn.Sequential(*layers)


def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)

            y_pred, _ = model(x)
            loss = criterion(y_pred, y)
            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def plot_confusion_matrix(labels, pred_labels, classes):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    cm = confusion_matrix(labels, pred_labels)
    cm = ConfusionMatrixDisplay(cm, display_labels=classes)
    cm.plot(values_format='d', cmap='Blues', ax=ax)
    plt.xticks(rotation=20)
    plt.savefig("./Images/confusion_matrix.pdf", dpi=600, bbox_inches='tight')
    plt.show()


def get_predictions(m, iterator, d):
    m.eval()

    image = []
    label = []
    prob = []

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(d)
            y_pred, _ = model(x)

            y_prob = f.softmax(y_pred, dim=-1)

            image.append(x.cpu())
            label.append(y.cpu())
            prob.append(y_prob.cpu())

    image = torch.cat(image, dim=0)
    label = torch.cat(label, dim=0)
    prob = torch.cat(prob, dim=0)

    return image, label, prob


test_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

test_data = torchvision.datasets.ImageFolder('./images/Test', transform=test_transforms)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=128)

model = VGG16()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('VGG-16_trained_model.pt', map_location=torch.device('cpu')))
test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

classes = ('Electron', 'Gamma Ray', 'Positron')

# outputs, _ = model(images)

images, labels, probs = get_predictions(model, test_loader, device)
pred_labels = torch.argmax(probs, 1)

#  Higher the energy for a class, the more the network thinks that the image is of the particular class
# This is the highest energy
# _, predicted = torch.max(outputs, 1)

# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(len(labels))))
# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(len(predicted))))

plot_confusion_matrix(labels, pred_labels, classes)
