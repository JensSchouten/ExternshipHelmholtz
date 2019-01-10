# ExternshipHelmholtz

These scripts were used in the externship project of Jens Schouten at the HelmholtzZentrum in Munich, titled: Classification of blood cells in acute lymphoblastic leukemia using convolutional neural networks. This work has been done using the ALL-IDB dataset of Labati et al. [1].

Below some explanation of the different scripts is given. The networks were trained on a Nvidia GeForce GTX TITAN X GPU with Python version 3.5, keras version 2.2.4 and tensorflow-GPU version 1.4.0. 

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

### ResNext_From_Scratch.py
Script that trains a ResNeXt from scratch on the single-cell images in the ALL-IDB dataset to classify between multiple classes of leukocytes using five-fold cross-validation.

### ResNext_Transfer_Learning.py
Script that uses transfer learning on a pre-trained ResNeXt, that is trained for a similar task, using the ALL-IDB dataset to classify between multiple classes of leukocytes using five-fold cross-validation.

## References
[1] R. D. Labati, V. Piuri, and F. Scotti, “All-idb: The acute lymphoblastic leukemia
image database for image processing,” in Image processing (ICIP), 2011 18th IEEE
international conference on, pp. 2045–2048, IEEE, 2011.

[2] S. Xie, R. Girshick, P. Dollár, Z. Tu, and K. He, “Aggregated residual transformations
for deep neural networks,” in Computer Vision and Pattern Recognition
(CVPR), 2017 IEEE Conference on, pp. 5987–5995, IEEE, 2017.
