# Reproducing Model Results
To reproduce the results (model) in the thesis paper, follow the following steps:

1. Train the model using the 15,000 non-shifted image dataset.
- Model should have a 99.8% testing accuracy on the non-shifted image testing set.

2. Train a new model using the 15,000 shifted image dataset where the initial input features are the the model trained in (1).
- Change the line here: https://github.com/ksinger12/LiquidO_Classifier_ENPH_455/blob/be7f53d08f23b50e44d88db0aed7509007550eef/Setup.py#L25
to the following:
```
import VGG
...
pretrained_model = VGG.VGG16()
pretrained_model.load_state_dict(torch.load('./name_of_trained_model.pt'))
```
- Note: the name of the file will need be modified with the original output file from (1). Also change the new output file name to not override the input file.
- Model should have a 99.29% testing accuracy on the shifted image testing set and 99.33% accuracy on the non-shifted image set.
