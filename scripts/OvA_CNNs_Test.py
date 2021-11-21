import os
import sys
import warnings
import numpy as np
import random
from tqdm import tqdm
import tensorflow as tf
import keras
from tqdm import tqdm
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Input, Conv1D, MaxPooling1D, \
Dropout, BatchNormalization, Activation, Concatenate, Masking, GlobalAveragePooling1D
from keras.models import Model
from keras import backend as K
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from tslearn.preprocessing import TimeSeriesResampler, TimeSeriesScalerMinMax
from tslearn.datasets import UCR_UEA_datasets
from tslearn.utils import to_time_series, to_time_series_dataset, from_sktime_dataset, to_sktime_dataset
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sktime.utils.data_processing import from_3d_numpy_to_nested
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# Paths should be defined here
DATA_PATH = 'Multivariate_ts' # Path to the Multivariate_ts folder that contains all the datasets
MODELS_PATH = '/content/drive/MyDrive/Master Thesis/Models/OvA_CNNs' # Path to the Models folder that contains the models
dataset = sys.argv[1]

exceptions = ['CharacterTrajectories', 'JapaneseVowels', 'InsectWingbeat', 'SpokenArabicDigits']
if dataset in exceptions:
    loader = UCR_UEA_datasets()
    x_train, y_train, x_test, y_test = loader.load_dataset(dataset)

else:
    x_train, y_train = load_from_tsfile_to_dataframe(
        os.path.join(DATA_PATH, f"{dataset}/{dataset}_TRAIN.ts")
    )
    x_test, y_test = load_from_tsfile_to_dataframe(
        os.path.join(DATA_PATH, f"{dataset}/{dataset}_TEST.ts")
    )

    x_train = from_sktime_dataset(x_train)
    x_test = from_sktime_dataset(x_test)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)

print('Dataset loaded')

# Normalize 

scalers = []
for i in range(x_train.shape[-1]):
    scaler = MinMaxScaler()
    x_train[:,:,i] = scaler.fit_transform(x_train[:,:,i])
    x_test[:,:,i] = scaler.transform(x_test[:,:,i])
    scalers.append(scaler)

n_classes = len(np.unique(y_train))

# Mask NaN values

mask_value = -1e-8   # NaN values will be masked for a constant length

x_train_masked = x_train.copy()
x_test_masked = x_test.copy()
x_train_masked[np.isnan(x_train_masked)] = mask_value
x_test_masked[np.isnan(x_test_masked)] = mask_value

# Load models

models = []
for i in range(n_classes):
    model_path = os.path.join(MODELS_PATH, f'{dataset}/OvA_CNN_{dataset}_class_{i}')
    model = load_model(model_path)
    models.append(model)

def predict_OvA(X, models):
     '''
        Prediction function for the OvA-CNNs
    '''

    preds_all = []
    for x in X:
        preds = []
        probs = []

        for i in range(n_classes):
            prob = models[i].predict(np.expand_dims(x, axis=0))
            pred = prob.argmax()
            preds.append(pred)
            probs.append(prob)

        # Ensemble all the models
        # Choose the model with the highest confidence for the positive class
        
        if (np.asarray(preds)==1).any():  # At least one model predicts 1
            idxs = np.where(np.asarray(preds)==1)[0]
            prob_max = probs[idxs[0]][0,1]
            final_pred = idxs[0]
            for idx in idxs:
                _prob = probs[idx][0,1]
                if _prob > prob_max:
                    final_pred = idx
                    prob_max = probs[final_pred][0,1]
        else:  # All models predict 0
            final_pred = n_classes
        
        preds_all.append(final_pred)
    
    return np.asarray(preds_all)


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
preds_test = predict_OvA(x_test_masked, models)

# Iterate through every given dataset to be used as unknowns
for unk_dataset in sys.argv[2:]:

    if unk_dataset in exceptions:
        loader = UCR_UEA_datasets()
        unk_train, unk_train_l, unk_test, unk_test_l = loader.load_dataset(unk_dataset)

    else:
        unk_test, _ = load_from_tsfile_to_dataframe(
            os.path.join(DATA_PATH, f"{unk_dataset}/{unk_dataset}_TEST.ts")
        )

        unk_test = from_sktime_dataset(unk_test)

    # Assert 1:1 ratio for knowns and unknowns
    if len(unk_test) >= len(x_test):
        x_unk = unk_test[np.random.choice(unk_test.shape[0], size=len(x_test), replace=False), :]
    # Add extra data from the train set of the unknown dataset
    else:
        unk_train, _ = load_from_tsfile_to_dataframe(
            os.path.join(DATA_PATH, f"{unk_dataset}/{unk_dataset}_TRAIN.ts")
        )
        unk_train = from_sktime_dataset(unk_train)
        x_unk = np.concatenate((unk_train, unk_test), axis=0)

        # Try to have 1:1 ratio again, nothing will happen if the unknowns are still less than the test set
        if len(x_unk) > len(x_test):
            x_unk = x_unk[np.random.choice(x_unk.shape[0], size=len(x_test), replace=False), :]

    # Resample the lenght of the time series
    x_unk = TimeSeriesResampler(sz=x_train.shape[1]).fit_transform(x_unk[:, :, :x_train.shape[2]])
    y_unk = np.ones((len(x_unk))) * n_classes # The label of the unknowns will be equal to the number of classes

    # Normalize each channel
    for i in range(x_unk.shape[-1]):
        x_unk[:,:,i] = scalers[i].transform(x_unk[:,:,i]) 

    # Mask NaN values
    x_unk[np.isnan(x_unk)] = mask_value

    # Known and unknwon combined results
    # x_combined = np.concatenate((x_test_masked, x_unk), axis=0)
    y_combined = np.concatenate((y_test, y_unk), axis=0)
    preds_unk = predict_OvA(x_unk, models)
    preds_combined = np.concatenate((preds_test, preds_unk), axis=0)
    print(f'Classification Result for the {dataset} combined with {unk_dataset} Dataset as unknown')
    print(classification_report(y_combined, preds_combined))
    cm=confusion_matrix(y_combined, preds_combined)
    print(cm, cm.shape)
    print('-------------')













