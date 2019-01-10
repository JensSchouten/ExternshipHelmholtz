### IMPORT PACKAGES
import numpy as np
import math 
import random
from keras.preprocessing.image import ImageDataGenerator
img_gen = ImageDataGenerator()


### DATA AUGMENATATION FUNCTION
def augmentation(images, labels, nrofimagesout):
    
    ## Get unique labels from label set
    uniqueLabels = np.unique(labels)
    
    ## Make new labels from 0 to the length of the unique labels
    newlabelvalues = list(range(len(uniqueLabels)))
    
    ## Create data augmented images
    newlabelsTrain = []
    newimagesTrain = []
    for i in range(len(uniqueLabels)):
        
        ## Get indices of the images of each unique label
        imageIndices, = np.where(labels == uniqueLabels[i])
        imageIndices = list(imageIndices)
        
        ## Calculate the number of augmented image per image with same label
        perImage = int(math.ceil(nrofimagesout/len(imageIndices)))    
        
        ## Shuffle images with same label
        random.shuffle(imageIndices)
        count = len(imageIndices)
        
        
        ### CREATING AUGMENTED IMAGES FOR EACHT IMAGE
        for j in imageIndices:
            currentImage = images[j]
            
            ## Add original image and label to lists
            newlabelsTrain.append(newlabelvalues[i])
            newimagesTrain.append(currentImage[:,:,0:3])
            
            ## Sample angels for rotation
            angles = random.sample(list(range(359)), perImage)
            
            ## Create augmented images
            for k in range(perImage):
                
                ## Decide if horizontal and/or vertical flip is done
                horflip = bool(random.randint(0,1))
                verflip = bool(random.randint(0,1))
                
                ## Check if number of images per class is not high than max
                if count < nrofimagesout:
                    
                    ## Create transform library and apply on image
                    transforms = {'theta': angles[k], 'flip_horizontal': horflip, 'flip_vertical': verflip}    
                    ImageNew = img_gen.apply_transform(currentImage, transforms)
                    
                    ## Add new image and label to the lists
                    newlabelsTrain.append(newlabelvalues[i])
                    newimagesTrain.append(ImageNew[:,:,0:3])
                    
                    count = count + 1
    
    ## Create right array of images
    newimagesTrain = np.stack(newimagesTrain,axis=0)  

    ## Return image matrix and label array
    return newimagesTrain, newlabelsTrain
