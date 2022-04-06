# LiquidO_Classifier_ENPH_455
Electron, gamma ray, and positron classifier based on photon images from LiquidO detector.

## Description
A Visual Geometry Group 16 (VGG16) convolutional neural network (CNN) built to classify three types of images:
Electrons, Gamma Rays, Positrons. 
The CNN has 16 layers with weights where the initial input is a 224x224 coloured image. The layers are structured as follows:

Input(224x224x3) &rarr; 2\*convolution(224x224x64) &rarr; max pool &rarr; 2\*convolution(112x112x128) &rarr; max pooling &rarr; 3\*convolution(56x56x256) &rarr; max pooling &rarr; 3\*convolution(28x28x512) &rarr; 3\*convolution(14x14x512) &rarr; max pooling &rarr; fully connected(1x1x4096) &rarr; fully connected(1x1x4096) &rarr; fully connected(1x1x1000) &rarr; soft max

Where the brackets represent: (height, width, channel), the max pooling layers use a 2x2 filter with a stride of 2, and the convolutions use a 3x3 filter with a padding of 1. 
Note that feature size is 7x7x512 after the last max pooling layer and channels refers to the number of features per pixel at each layer.


## Instructions to Run Code
To run the code, python version 3.\*.\* must be installed. To check your version, run
```
python -V
```

To download the code and train the system follow the following commands:

```
git clone 
cd LiquidO_Classifier_ENPH_455
pip install -r requirements.txt
python3 Train.py
```

To evaluate the code on the evaluation set:
```
python3 VGG16_Evaluation.py
```

### GPU Specific Setup with Neutrino Server:
If the code was cloned onto your personal machine, the files:  
VGG.py, Train.py, IteratorWrapper.py, LearningRateFinder.py, ExponentialLearningRate.py  
will need to be copied to the server. To copy them over, use the following setup:
```
Unix/Linux:
scp username@remoteHost:/remote/dir/file.txt /local/dir/

Windows:
pscp.exe username@remoteHost:/remote/dir/file.txt d:\
```
For me, this looked like:  
```
scp -r username@neutrino.phy.queensu.ca:/home/username/simulation/build/ /local/directory/path/
```

Ensure that before running the code, all of the dependencies (seen in requirements.txt) have been installed.
Note that if the version of CUDA changes from X.X.X, the version of PyTorch may need to be updated as well.

To use the GPU for neutrino accounts, the following must be put into the .bashrc file:
```
environment vairiables for GPU
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda
```
To see if anything is running on the GPU before training the CNN and/or to check the CUDA version, 
the following command can be executed:
```
nvidia-smi
```
## Learning Rate
The learning rate is found analytically by plotting the outputs of the following call:
```
lrs, losses = lr_finder.range_test(train_loader, ending_rate, number_of_iterations)
```
Where losses is the y-axis and lrs is the x-axis plotted on a logarithmic axis. The steepest part of the slope is the
learning rate where the average of rates across the slope is used.
The Learning_Rate_Execution script can be used to obtain the data for plotting purposes. For now, a found learning rate
used as defined in the Setup script.

## Pre-Trained Model
Using the pre-trained VGG-16 batch normalized model trained on the ImageNet dataset (meaning output is 1000 classes).
The output of the set was modified to have 3 to correspond with the number of classes in this classificaiton problem.
The only issue with the pre-trained model is it only returns the final output layer of the CNN thus we only load in the 
parameters that have been trained.  

The pre-trained models expects input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of 
shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] 
and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

> **⚠ WARNING: Potential certificaiton issue **:  
> Initially I ran into a certification error. The error was solved using the following: \
> For mac users: go to Macintosh HD > Applications > Python 3.\* folder > double click on Install Certificates.command" file

## Training Details
### Optimizer: Adam
Adam is a one-dimensional method of gradient descent used to find parameters that make up the data's cost function. The 
results can then be used to classify images when combined with the neural network.

### Loss Function: Cross-Entropy
Cross-entropy cost function. Computes the soft-max activation function on supplied predictions as well as the loss 
via negative log likelihood.

## Final Note
All results from this model and details about the project can be found in the thesis paper in this repo [here](https://github.com/ksinger12/LiquidO_Classifier_ENPH_455/blob/bf3acab91866e55b43fce7505e895704cc386fd8/Classifying%20Particle%20Interactions%20with%20LiquidO%20using%20Deep%20Learning.pdf).

# References
Adam optimizer: 
https://arxiv.org/pdf/1412.6980.pdf

Cross-entropy: 
https://machinelearningmastery.com/cross-entropy-for-machine-learning/

The code is based off the following:
https://colab.research.google.com/github/bentrevett/pytorch-image-classification/blob/master/4_vgg.ipynb (used for implementing NN with PyTorch)

Strategy source:
https://arxiv.org/abs/1506.01186

The pre-trained model used (used to prevent the weights starting from scratch which takes longer to train).
The data source is from VGG-16 batch normalized model trained on the ImageNet dataset 

Other helpful links: 
https://colab.research.google.com/github/bentrevett/pytorch-image-classification/
