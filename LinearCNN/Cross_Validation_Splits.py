### IMPORT PACKAGES
import numpy as np
import os
import pandas as pd
import random
import xlsxwriter

### INPUT
## Paths
path_to_data = ""
path_to_labels = ""
labels_column = ""
path_to_save = ""
file_name = ""

## Train-test split precentages
precTrain = 0.8
precTest = 0.2

## Number of folds
folds = 5

## Minimal number of images per class
thresholdIncluded = 5

## Not include thrombocytes? Yes(1), no(0)
removeThrombo = 1
labelThrombo = 9


### IMPORT LABELS
df = pd.read_excel(path_to_labels)
Labels = np.array(df[[labels_column]])

## Only import labels of images that are included
pathImages, dirsImages, filesImages = next(os.walk(path_to_data))
labels = []
for i in range(len(filesImages)):
    labels.append(int(Labels[int(filesImages[i][-9:-6])-1]))
    
uniqueLabels = list(np.unique(labels))
    
### CREATE CROSS-VALIDATION SPLIT
## Array with only zeros
cv = np.zeros((len(labels),1))

## For each unique label
for i in uniqueLabels:
    ## Indices of images with this label
    idx, = np.where(np.array(labels) == i)
    idx = list(idx)
    
    ## If not included, cv fold becomes higher than the number of folds
    if (i == labelThrombo and removeThrombo == 1) or len(idx) < thresholdIncluded:
        cv[idx] = folds+1
        
    ## Else, randomly the images are divided into the folds
    else:   
        random.shuffle(idx)
        
        for j in range(folds):
            indicesFold = idx[j::folds]
            cv[indicesFold] = j


### SAVE CROSS-VALIDATION INDICES
workbook = xlsxwriter.Workbook(path_to_save + file_name)
worksheet = workbook.add_worksheet()

col = 0
for row, data in enumerate(cv):
    worksheet.write_column(row, col, data)

workbook.close()
    
