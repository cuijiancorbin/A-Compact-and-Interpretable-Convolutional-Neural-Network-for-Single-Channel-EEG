# A Compact and Interpretable Convolutional Neural Network for Cross-Subject Driver Drowsiness Detection from Single-Channel EEG 

Pytorch implementation of the paper "A Compact and Interpretable Convolutional Neural Network for Cross-Subject Driver Drowsiness Detection from Single-Channel EEG".

The project contains 3 code files. They are implemented with Python 3.6.6.

"CampactCNN.py" contains the model.
required library: torch

"LeaveOneOut_acc.py" contains the leave-one-subject-out method to get the classifcation accuracies.
It requires the computer to have cuda supported GPU installed.
required library:torch,scipy,numpy,sklearn

"VisualizationTech.py" contains the visualization technique based on the CAM method (Class Activation Map).
It requires the computer to have cuda supported GPU installed.
required library:torch,scipy,numpy,matplotlib,mne

The processed dataset has been uploaded to:
https://figshare.com/articles/dataset/EEG_driver_drowsiness_dataset/14273687

If you have any problems, please Contact Dr. Cui Jian at cuij0006@ntu.edu.sg
