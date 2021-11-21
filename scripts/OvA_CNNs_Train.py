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
    scaler = StandardScaler()
    x_train[:,:,i] = scaler.fit_transform(x_train[:,:,i])
    x_test[:,:,i] = scaler.transform(x_test[:,:,i])
    scalers.append(scaler)

n_classes = len(np.unique(y_train))

# Mask NaN values

mask_value = -10   # NaN values will be masked for a constant length

x_train_masked = x_train.copy()
x_test_masked = x_test.copy()
x_train_masked[np.isnan(x_train_masked)] = mask_value
x_test_masked[np.isnan(x_test_masked)] = mask_value

# Define the CNN model
def cnn_model(in_shape, mask_value, n_classes=2):
    model = Sequential()

    model.add(Masking(mask_value=mask_value, input_shape=in_shape))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2)) 
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    return model

def get_new_labels(labels, pos_class_idx):
    '''
        Changes the labels of the given class to 1, and all the others to 0
    '''
    new_labels = labels.copy()
    pos_samples = np.where(new_labels == pos_class_idx)
    neg_samples = np.where(new_labels != pos_class_idx)
    new_labels[pos_samples] = 1
    new_labels[neg_samples] = 0
    return new_labels

# Learning

in_shape = x_train[0].shape 
models = []
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, 
                                              min_lr=0.0001) 

# Compile and train each model
for i in range(n_classes):
    model = cnn_model(in_shape, mask_value=mask_value)
    model.compile(keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'],)

    file_path = os.path.join(MODELS_PATH, f'{dataset}/OvA_CNN_{dataset}_class_{i}')
    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)
    callbacks = [reduce_lr, model_checkpoint]
    # Create new labels
    train_labels = get_new_labels(y_train, i)
    test_labels = get_new_labels(y_test, i)

    # Compute class weights
    class_weights = class_weight.compute_class_weight('balanced', np.unique(train_labels), train_labels)
    class_weights = dict(enumerate(class_weights))

    print(f'Training the model for the {i}. class')
    # Train the model.
    model.fit(
    x_train_masked,
    to_categorical(train_labels),
    epochs=10,
    batch_size=128,
    validation_data=(x_test_masked, to_categorical(test_labels)),
    class_weight=class_weights,
    verbose=2,
    callbacks = callbacks
    )
    print(f'Finished training the model for the {i}. class')
    print('------------------')
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
print('Open Set One-vs-All CNNs Classification Report')
print(classification_report(y_test, preds_test))
print(confusion_matrix(y_test, preds_test))
print('-------------')





