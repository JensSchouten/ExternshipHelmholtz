### IMPORT PACKAGES
from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator
from skimage import io
import numpy as np
import os
import pandas as pd
from datetime import datetime
import pickle
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
import cv2

### IMPORT SCRIPTS
from ResNext_network import residual_network
from Data_Augmentation_Transfer_Learning import augmentation


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

## Image dimensions
img_height = 400
img_width = 400
img_channels = 3
output = 5
folds = 5


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
    
    ## Resize images to 400 by 400 pixels
    imLinearRescaled = cv2.resize(im, dsize=(400,400), interpolation=cv2.INTER_LINEAR)
    
    imagesAll.append(imLinearRescaled)
    imageNrs.append(int(filesIm[i][2:5])-1)
    
imagesAll = np.stack(imagesAll,axis=0)  


### IMPORT LABELS
df = pd.read_excel(path_to_labels)

labelsAll = df[[label_column]].values.flatten()
labelsAll = labelsAll[imageNrs]


### IMPORT CROSS-VALIDATION SPLIT
df = pd.read_excel(path_to_cv, header=None)
cvIndices = df.values.flatten()


### TRAINING AND TESTING FOR EACH FOLD
compares = []
for CV in range(folds):
    
    ### CREATE NEW NETWORK
    image_tensor = layers.Input(shape=(img_height, img_width, img_channels))
    network_output = residual_network(image_tensor)

    model = models.Model(inputs=[image_tensor], outputs=[network_output])
    
    ## Compile network
    adam = optimizers.Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['acc'])  
    
    
    ### CREATE IMAGE AND LABEL SETS
    ## Create the testing images and labels
    idxTest, = np.where(cvIndices == CV)
    idxTest = list(idxTest)  
    
    imagesTest = imagesAll[idxTest,:,:]
    labelsTest = labelsAll[idxTest]
    
    ## Create the training images and labels      
    idxTrain = np.int64([i for j, i in enumerate(range(len(labelsAll))) if j not in idxTest])
    idxInTrain, = np.where(cvIndices[idxTrain] < folds)    
    idxTrain = list(idxTrain[list(idxInTrain)])
    
    imagesTrain = imagesAll[idxTrain,:,:]
    labelsTrain = labelsAll[idxTrain]
    
    
    ### APPLY DATA AUGMENTATION ON TRAINING IMAGES TO FIX IMBALANCE
    newimagesTrain, newlabelsTrain = augmentation(imagesTrain, labelsTrain, 150)
       
    
    ### CREATE LABEL MATRICES AS INPUT
    ## For training
    newlabelsTrainMatrix = np.zeros((len(newlabelsTrain), len(np.unique(newlabelsTrain))))
    for i in range(len(newlabelsTrain)):
        newlabelsTrainMatrix[i, newlabelsTrain[i]] = 1
    
    ## For testing
    labelsTestMatrix = np.zeros((len(labelsTest), len(np.unique(labelsTest))))  
    newlabelsTest = []
    for i in range(len(labelsTest)):
        idx, = np.where(np.unique(labelsTest) == labelsTest[i])
        labelsTestMatrix[i, int(idx)] = 1
        newlabelsTest.append(int(idx))
    
    
    ### TRAIN NETWORK
    ## Create generators
    train_generator = train_datagen.flow(
            newimagesTrain, newlabelsTrainMatrix,
            batch_size=15,
            shuffle=True)
    
    val_generator = test_datagen.flow(
            imagesTest, labelsTestMatrix,
            batch_size=5,
            shuffle=True)
    
    ## Use of callbacks
    checkpoint = ModelCheckpoint(path_to_save + modelname + '_' + str(CV) + '_checkpoint' + '.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    csvlog = CSVLogger(path_to_save + modelname + '_' + str(CV) + '_train_log.csv',append=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=75, verbose=1)
    
    ## Training
    startTime = datetime.now()    
    history = model.fit_generator(
                train_generator,
                steps_per_epoch=50,
                epochs=epoch_nr,
                validation_data=val_generator,
                validation_steps=10,
                callbacks=[checkpoint, csvlog, early_stopping])
    
    Time = datetime.now() - startTime
    
       
    ### TESTING THE BEST NETWORK OF THIS FOLD     
    ## Load best network
    image_tensor = layers.Input(shape=(img_height, img_width, img_channels))
    network_output = residual_network(image_tensor)

    model = models.Model(inputs=[image_tensor], outputs=[network_output])
    model.load_weights(path_to_save+modelname+ '_' + str(CV) + '_checkpoint' + '.hdf5')
    
    ## Compile network
    adam = optimizers.Adam(lr=0.0001)
    model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['acc']) 
        
    ## Create generator
    test_generator = test_datagen.flow(
                imagesTest, labelsTest,
                batch_size=1,
                shuffle=False)    
        
    ## Testing the network
    predictions = model.predict_generator(
            test_generator, 
            steps=len(labelsTest),
            verbose=1)
    
    ## Create array with labels and predictions
    uniqueLabelsTest = np.unique(labelsTest)
    test_pred_array = []
    for i in range(len(predictions)):
        indexMax = np.argmax(predictions[i,:])
        test_pred_array.append(uniqueLabelsTest[indexMax])    
    
    compare = np.column_stack((labelsTest,test_pred_array))
    compares.extend(compare)   
    
    
    ### SAVING VARIABLES
    with open(path_to_save + modelname + '_' + str(CV) + '.pkl', 'wb') as f:
        pickle.dump([predictions, Time, compare], f)


### SAVING RESULTS
compares = np.stack(compares,axis=0)  
with open(path_to_save + modelname + '.pkl', 'wb') as f:
    pickle.dump([compares], f)
    
    
