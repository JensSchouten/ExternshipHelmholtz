### IMPORT PACKAGES
import numpy as np
from datetime import datetime
import os
import pandas as pd
from skimage import io
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras import layers, models
import pickle
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
import random

### INPUT
## Data augmentation? Yes(1), no(0)
data_augmentation = 1  

## Paths images, labels and save folder
path_to_images = ""
path_to_save = ""
path_to_labels = ""

## Model name
modelname = ""

## Number of epochs
epoch_nr = 500


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
        model.add(layers.Dense(1, activation='sigmoid'))

        return model


img_width, img_height = 257, 257
if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
else:
        input_shape = (img_width, img_height, 3)


### CREATE DATA GENERATORS
test_datagen = ImageDataGenerator(rescale=1./255)
if data_augmentation == 1:
    train_datagen = ImageDataGenerator(
            rotation_range=359,
            horizontal_flip=True,
            vertical_flip=True,
            rescale=1./255,    
            fill_mode='nearest')        
elif data_augmentation == 0:
    train_datagen = ImageDataGenerator(rescale=1./255)


### IMPORT IMAGES
pathIm, dirsIm, filesIm = next(os.walk(path_to_images))
        
## Put all images in a 3D matrix
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

labelsAll = df[['Binary']].values.flatten()
labelsAll = labelsAll[imageNrs]


### SHUFFLE IMAGES AND LABELS TO GET RANDOM DISTRIBUTION
indices = list(range(len(labelsAll)))
random.shuffle(indices)

labelsAll = labelsAll[indices]
imagesAll = imagesAll[indices,:,:]


### TRAINING AND TESTING FOR EACH FOLD
compares = []
for CV in range(0,len(labelsAll),5):
    
    ### CREATE NEW NETWORK
    model = setup_sequential_model()
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    
    
    ### CREATE IMAGE AND LABEL SETS
    ## Create the testing images and labels
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
    
    
    ### TRAIN NETWORK
    ## Create generators
    train_generator = train_datagen.flow(
            imagesTrain, labelsTrain,
            batch_size=5,
            shuffle=False)
    
    val_generator = test_datagen.flow(
            imagesVal, labelsVal,
            batch_size=5,
            shuffle=False)
    
    ## Use of callbacks
    checkpoint = ModelCheckpoint(path_to_save + modelname + '_' + str(CV) + '_checkpoint' + '.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    csvlog = CSVLogger(path_to_save+modelname+ '_' + str(CV) + '_train_log.csv',append=False)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=1)
    
    ## Training
    startTime = datetime.now()    
    history = model.fit_generator(
                train_generator,
                steps_per_epoch=40,
                epochs=epoch_nr,
                validation_data=val_generator,
                validation_steps=10,
                callbacks = [checkpoint, csvlog, early_stopping])    
    Time = datetime.now() - startTime

      
    ### TESTING THE BEST NETWORK OF THIS FOLD     
    ## Load best network
    model = setup_sequential_model()
    model = load_model(path_to_save + modelname+ '_' + str(CV) + '_checkpoint' + '.hdf5')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
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
    compares.extend(np.column_stack((labelsTest,predictions)))

    
### SAVING RESULTS
compares = np.stack(compares,axis=0)  
with open(path_to_save + modelname + '.pkl', 'wb') as f:
    pickle.dump([Time, compares], f)




    



