### IMPORT PACKAGES
import numpy as np
import os
import pandas as pd
from skimage import io
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras import layers, models
from datetime import datetime
import pickle
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
import random

### IMPORT SCRIPTS
from Linear_CNN_DA import augmentation


### INPUT
## Paths
path_to_images = ""     # To images
path_to_labels = ""     # To labels
label_column = ""       # Column of right labels
path_to_cv = ""         # To cross-validation splits
path_to_save = ""       # To save folder

## Model name
modelname = ""

## Number of epochs
epoch_nr = 500

## Label numbers
labels = [3,4,6,7,8]

## Image dimensions
img_height = 257
img_width = 257
img_channels = 3
output = 5


### MODEL SETUP
def setup_sequential_model():
        model = models.Sequential()
        
        model.add(layers.Conv2D(4, (3, 3), activation='relu', input_shape=input_shape))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(8, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(8, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(8, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(16, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(16, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(output, activation='softmax'))

        return model

if K.image_data_format() == 'channels_first':
        input_shape = (img_channels, img_width, img_height)
else:
        input_shape = (img_width, img_height, img_channels)

### CREATE DATA GENERATORS
test_datagen = ImageDataGenerator(rescale=1./255)
train_datagen = ImageDataGenerator(rescale=1./255)
   

### IMPORT IMAGES
pathIm, dirsIm, filesIm = next(os.walk(path_to_images))
        
## Put all images in a 3D matrix and save image numbers
imagesAll = []    
imageNrs = []
for i in range(len(filesIm)):
    im = io.imread(pathIm + filesIm[i],0) 
    im = im[:,:,0:3]
    imagesAll.append(im)
    imageNrs.append(int(filesIm[i][2:5])-1)
    
imagesAll = np.stack(imagesAll,axis=0)  


### IMPORT LABELS
df = pd.read_excel(path_to_labels)

labelsAll = df[[label_column]].values.flatten()
labelsAll = labelsAll[imageNrs]


### IMPORT CROSS-VALIDATION SPLIT
df = pd.read_excel(path_to_cv, header=None)
cvIndices = df.values.flatten()

## Remove all classes with four or less images
indices, = np.where(cvIndices < 5)

imagesAll = imagesAll[indices,:,:]
labelsAll = labelsAll[indices]


### SHUFFLE IMAGES AND LABELS TO GET RANDOM DISTRIBUTION
indices = list(range(len(labelsAll)))
random.shuffle(indices)

labelsAll = labelsAll[indices]
imagesAll = imagesAll[indices,:,:]


### TRAINING AND TESTING FOR EACH FOLD
comparesVal = []
comparesTest = []
for CV in range(0,len(labelsAll),5):
    
    ### CREATE NEW NETWORK
    model = setup_sequential_model()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    
    ### CREATE IMAGE AND LABEL SETS
    ## Create the testing images and labels
    if CV + 5 > len(labelsAll):
        idxTest = list(range(CV,len(labelsAll)))
        imagesTest = imagesAll[idxTest,:,:]
        labelsTest = labelsAll[idxTest]
    else:    
        idxTest = list(range(CV,CV+5))
        imagesTest = imagesAll[idxTest,:,:]
        labelsTest = labelsAll[idxTest]
    
    ## Create the validation images and labels
    idxTrain = [i for j, i in enumerate(list(range(len(labelsAll)))) if j not in idxTest]     
    idxVal = random.sample(idxTrain, 45)
    
    imagesVal = imagesAll[idxVal,:,:]
    labelsVal = labelsAll[idxVal]
    
    ## Redefine training images and labels 
    idxTrain_new = [i for j, i in enumerate(list(range(len(labelsAll)))) if j not in idxVal and j not in idxTest]
    
    imagesTrain = imagesAll[idxTrain_new,:,:]
    labelsTrain = labelsAll[idxTrain_new]   
    
    
    ### APPLY DATA AUGMENTATION ON TRAINING IMAGES TO FIX IMBALANCE
    newimagesTrain, newlabelsTrain = augmentation(imagesTrain, labelsTrain, 150)
       
    
    ### CREATE LABEL MATRICES AS INPUT
    ## For training
    newlabelsTrainMatrix = np.zeros((len(newlabelsTrain), output))
    for i in range(len(newlabelsTrain)):
        newlabelsTrainMatrix[i, newlabelsTrain[i]] = 1

    ## For validation
    labelsValMatrix = np.zeros((len(labelsVal), output))  
    newlabelsVal = []
    for i in range(len(labelsVal)):
        idx, = np.where(labels == labelsVal[i])
        labelsValMatrix[i, int(idx)] = 1
        newlabelsVal.append(int(idx))
      
    
    ### TRAIN NETWORK
    ## Create generators
    train_generator = train_datagen.flow(
            newimagesTrain, newlabelsTrainMatrix,
            batch_size=15,
            shuffle=True)
    
    val_generator = test_datagen.flow(
            imagesVal, labelsValMatrix,
            batch_size=5,
            shuffle=False)
    
    ## Use of callbacks
    checkpoint = ModelCheckpoint(path_to_save+modelname+ '_' + str(CV) + '_checkpoint' + '.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    csvlog = CSVLogger(path_to_save+modelname+ '_' + str(CV) + '_train_log.csv',append=False)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1)
    
    ## Training
    startTime = datetime.now()    
    history = model.fit_generator(
                train_generator,
                steps_per_epoch=50,
                epochs=epoch_nr,
                validation_data=val_generator,
                validation_steps=9,
                callbacks=[checkpoint, csvlog, early_stopping])
    
    Time = datetime.now() - startTime

    
    ### TESTING THE BEST NETWORK OF THIS FOLD     
    ## Load best network
    model = setup_sequential_model()
    model = load_model(path_to_save+modelname+ '_' + str(CV) + '_checkpoint' + '.hdf5')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    
    ## Create generator
    test_generator = test_datagen.flow(
                imagesTest, labelsTest,
                batch_size=1,
                shuffle=False)    
    
    ## Testing the network
    predictionsTest = model.predict_generator(
            test_generator, 
            steps=len(labelsTest),
            verbose=1)
    
    ## Create array with labels and predictions
    test_pred_array = []
    for i in range(len(predictionsTest)):
        indexMax = np.argmax(predictionsTest[i,:])
        test_pred_array.append(labels[indexMax]) 
        
    compareTest = np.column_stack((labelsTest,test_pred_array))
    comparesTest.extend(compareTest)
    
    
    ### TESTING NETWORK ON VALIDATION SET
    predictionsVal = model.predict_generator(
            val_generator, 
            steps=9,
            verbose=1)
    
    ## Create array with labels and predictions
    val_pred_array = []
    for i in range(len(predictionsVal)):
        indexMax = np.argmax(predictionsVal[i,:])
        val_pred_array.append(labels[indexMax]) 
        
    compareVal = np.column_stack((labelsVal,val_pred_array))
    comparesVal.extend(compareVal)    
    
    
    ### SAVING VARIABLES
    with open(path_to_save + modelname + '_' + str(CV) + '.pkl', 'wb') as f:
        pickle.dump([Time, compareVal, compareTest, idxTest, idxTrain_new, idxVal], f)
 
    
### SAVING RESULTS
comparesTest = np.stack(comparesTest,axis=0)  
comparesVal = np.stack(comparesVal,axis=0)  
with open(path_to_save + modelname + '.pkl', 'wb') as f:
    pickle.dump([comparesTest, comparesVal], f)
   
    
    
    
