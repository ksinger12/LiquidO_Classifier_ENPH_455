import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as f

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

import VGG


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

model = VGG.VGG16()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('VGG-16_trained_model.pt', map_location=torch.device('cpu')))
test_loss, test_acc = VGG.evaluate(model, test_loader, nn.CrossEntropyLoss(), device)

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
