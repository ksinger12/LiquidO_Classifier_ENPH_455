import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as f

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

import numpy as np
import VGG


def plot_confusion_matrix(true_label, predicted, types):
    """
    Plotting the confusion matrix to show accuracy of classifications
    :param true_label: true labels of images
    :param predicted: predicted labels of images
    :param types: classes of images
    :return: no return. Saves and plots the matrix
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)

    matrix = confusion_matrix(true_label, predicted)
    matrix = ConfusionMatrixDisplay(matrix, display_labels=types)

    # Plotting
    matrix.plot(values_format='d', cmap='Blues', ax=ax)
    plt.xticks(rotation=20)
    plt.savefig("./Images/confusion_matrix.pdf", dpi=600, bbox_inches='tight')
    plt.show()


def normalize_image(img):
    """
    Normalizes the images to make plotting the images easier
    :param img:
    :return:
    """
    image_min = img.min()
    image_max = img.max()
    img.clamp_(min=image_min, max=image_max)
    img.add_(-image_min).div_(image_max - image_min + 1e-5)

    return img


def plot_incorrect(incorrect, types):
    """
    Plots the incorrectly labels to a max of 50 images
    :param incorrect: list of incorrect images ordered from most to least incorrect
    :param types: types of images, aka classes
    :return: no return. Saves and outputs a plot
    """
    num_img_printing = len(incorrect)

    if num_img_printing > 50:
        num_img_printing = 50  # never printing more than 50 images

    rows = int(np.sqrt(num_img_printing))
    cols = int(np.sqrt(num_img_printing))

    fig = plt.figure(figsize=(25, 20))

    for i in range(rows*cols):
        ax = fig.add_subplot(rows, cols, i+1)

        # Incorrect image, true label, and probability
        img, t_label, pr = incorrect[i]
        img = img.permute(1, 2, 0)

        # Obtaining the probability and label of incorrect prediction
        incorrect_prob, incorrect_label = torch.max(pr, dim=0)
        img = normalize_image(img)

        ax.imshow(img.cpu().numpy())
        ax.set_title(f'True Label: {types[t_label]} ({pr[t_label]:.3f})\n'
                     f'Pred. Label: {types[incorrect_label]} ({incorrect_prob:.3f})')
        ax.axis('off')

    fig.subplots_adjust(hspace=0.4)
    plt.savefig("./Images/incorrect_classifications.pdf", dpi=600, bbox_inches='tight')
    plt.show()


def get_predictions(m, loader, d):
    """
    Retrieves the predicted labels for testing images
    :param m: model
    :param loader: test image loader
    :param d: device to use (cpu or cuda)
    :return: lists of images, labels, and probability of prediction
    """
    m.eval()

    imgs = []
    img_labels = []
    img_probs = []

    with torch.no_grad():
        for (x, y) in loader:
            x = x.to(d)

            # Retrieving prediction and probability of prediction
            y_pred, _ = model(x)
            y_prob = f.softmax(y_pred, dim=-1)

            # Saving the predictions
            imgs.append(x.cpu())
            img_labels.append(y.cpu())
            img_probs.append(y_prob.cpu())

    return torch.cat(imgs, dim=0), torch.cat(img_labels, dim=0), torch.cat(img_probs, dim=0)


classes = ('Electron', 'Gamma Ray', 'Positron')

test_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_data = torchvision.datasets.ImageFolder('./images/Test', transform=test_transforms)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=128)

model = VGG.VGG16()

# Setting up device and loading in the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('VGG-16_trained_model.pt', map_location=torch.device('cpu')))
test_loss, test_acc = VGG.evaluate(model, test_loader, nn.CrossEntropyLoss(), device)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc * 100:.2f}%')

# Retrieving the predictions and obtaining the predicted labels
images, labels, probabilities = get_predictions(model, test_loader, device)
pred_labels = torch.argmax(probabilities, 1)

plot_confusion_matrix(labels, pred_labels, classes)

# Getting incorrectly labelled images for the image set
incorrect_examples = []
for image, label, prob, correct in zip(images, labels, probabilities, torch.eq(labels, pred_labels)):
    if not correct:
        incorrect_examples.append((image, label, prob))

# Sorting incorrect examples by most incorrect (to ensure the most incorrect get plotted)
incorrect_examples.sort(reverse=True, key=lambda x: torch.max(x[2], dim=0).values)
plot_incorrect(incorrect_examples, classes)
