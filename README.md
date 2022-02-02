# LiquidO_Classifier_ENPH_455
Electron, gamma ray, and positron classifier based on photon images from LiquidO detector.

## Description
A Visual Geometry Group 16 (VGG16) convolutional neural network (CNN) built to classify three types of images:
Electrons, Gamma Rays, Positrons. 
TheCNN has 16 layers with weights where the initial input is a 224x224 coloured image. The layers are structured as follows:

Input(224x224x3) &rarr; 2convolution(224x224x64) &rarr; max pool &rarr; 2*convolution(112x112x128) &rarr; max pooling &rarr; 3*convolution(56x56x256) &rarr; max pooling &rarr; 3*convolution(28x28x512) &rarr; 3*convolution(14x14x512) &rarr; max pooling &rarr; fully connected(1x1x4096) &rarr; fully connected(1x1x4096) &rarr; fully connected(1x1x1000) &rarr; soft max

Where the brackets represent: (height, width, channel), the max pooling layers use a 2x2 filter with a stride of 2, and the convolutions use a 3x3 filter with a padding of 1. 
Note that feature size is 7x7x512 after the last max pooling layer and channels refers to the number of features per pixel at each layer.



