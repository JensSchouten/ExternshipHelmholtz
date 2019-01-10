# ExternshipHelmholtz

These scripts were used in the externship project of Jens Schouten at the HelmholtzZentrum in Munich, titled: Classification of blood cells in acute lymphoblastic leukemia using convolutional neural networks. 

Below some explanation of the different scripts is given. All scripts were created using Python 3.

## Linear CNN scripts
### Cross_Validation_Splits.py
Script that evenly divides the images of the different classes in n-folds so that these cross-validation folds contain images from all classes. Classes with, in this case, four images or less are excluded and get a higher fold number. 

### Linear_CNN_DA.py
Script that uses data augmentation to make each class have the same number of images.

### Linear_CNN_Binary_LPOCV.py
Script for training a binary classifier between lymphoblast cells and other types of blood cells using leave-five-out cross-validation. 

### Linear_CNN_Multiclass_LPOCV.py
Script for training a multiclass classifier between the different classes of leukocytes using leave-five-out cross-validation. 


## ResNeXt scripts
### ResNext_DA.py
Script that uses data augmentation to make each class have the same number of images.

### ResNext_network.py
Network structure of the ResNeXt. This script is created by Xie et al. [2]
