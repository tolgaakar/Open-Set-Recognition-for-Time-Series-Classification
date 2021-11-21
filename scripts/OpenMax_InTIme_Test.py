import os
import sys
import numpy as np
import pandas as pd
import random
import math
import pickle
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Input, Conv1D, MaxPooling1D, \
Dropout, BatchNormalization, Activation, Concatenate, Masking, GlobalAveragePooling1D, \
Conv1DTranspose, Reshape, PReLU
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical
from tslearn.preprocessing import TimeSeriesResampler, TimeSeriesScalerMinMax, TimeSeriesScalerMeanVariance
from tslearn.datasets import UCR_UEA_datasets
from tslearn.utils import to_time_series, to_time_series_dataset, from_sktime_dataset, to_sktime_dataset
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy.stats import weibull_min
from scipy.special import softmax
import scipy.spatial.distance as distance
import scipy.optimize as optimize
import warnings
warnings.filterwarnings("ignore")


# Load dataset

# Paths should be defined here
DATA_PATH = 'Multivariate_ts'
AUG_PATH = 'augmented'
MODELS_PATH = '/content/drive/MyDrive/Master Thesis/Models/OM-IT'
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

n_classes = len(np.unique(y_train))

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


def inception_module(input_tensor, stride=1, activation='linear', bottleneck_size=32, kernel_size=40, nb_filters=32):

    if int(input_tensor.shape[-1]) > 1:
        bottleneck_size = math.ceil(input_tensor.shape[-1] / 4)
        input_inception = Conv1D(filters=bottleneck_size, kernel_size=1,
                                              padding='same', activation=activation, use_bias=False)(input_tensor)
    else:
        input_inception = input_tensor


    # kernel_size_s = [40, 20, 10]
    kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]

    conv_list = []

    for i in range(len(kernel_size_s)):
        conv_list.append(Conv1D(filters=nb_filters, kernel_size=kernel_size_s[i],
                                             strides=stride, padding='same', activation=activation, use_bias=False)(input_inception))

    max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)

    conv_6 = Conv1D(filters=nb_filters, kernel_size=1,
                                 padding='same', activation=activation, use_bias=False)(max_pool_1)

    conv_list.append(conv_6)

    x = keras.layers.Concatenate(axis=2)(conv_list)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    return x


def shortcut_layer(input_tensor, out_tensor):
    shortcut_y = Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                        padding='same', use_bias=False)(input_tensor)
    shortcut_y = BatchNormalization()(shortcut_y)

    x = keras.layers.Add()([shortcut_y, out_tensor])
    x = Activation('relu')(x)
    return x

def InceptionTime(input_shape, nb_classes, depth=6, model_num=0):
    '''
        Defines single InceptionTime model
    '''
    input_layer = Input(input_shape)
    masking = Masking(mask_value=mask_value, input_shape=input_shape)(input_layer)

    x = masking
    input_res = masking

    for d in range(depth):  # for each inception module

        x = inception_module(x)
        # x = InceptionModule()(x)

        if d % 3 == 2:  # residual connection
            x = shortcut_layer(input_res, x)
            input_res = x

    gap_layer = GlobalAveragePooling1D()(x)

    output_layer = Dense(nb_classes)(gap_layer)
    output_layer = Activation('softmax')(output_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                    metrics=['accuracy'])
    
    #print(model.summary())

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                    min_lr=0.0001) 

    file_path = os.path.join(MODELS_PATH, f'{dataset}/{dataset}_InceptionTime_{model_num+1}.hdf5')

    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                        save_best_only=True)

    callbacks = [reduce_lr, model_checkpoint]

    return model, callbacks


# Build the Inception Time ensemble 
class InceptionTime_Ensemble:
    def __init__(self, models, callbacks, x_train, y_train, epochs, batch_size, val_data):
        self.models = models
        self.callbacks = callbacks
        self.x_train = x_train
        self.y_train = y_train
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_data = val_data

    
    def fit(self):
        cnt = 1
        for model, callbacks_ in tqdm(zip(self.models, self.callbacks)):
            print(f'\n Fitting Model {cnt}')
            model.fit(self.x_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, 
                      validation_data=self.val_data, callbacks=callbacks_, verbose=0)
            print(f'\n Finished')
            cnt += 1

    def load_weights(self, folder_path):
        for i in range(len(self.models)):
            file_path = os.path.join(folder_path, f'{dataset}_InceptionTime_{i+1}.hdf5')
            self.models[i] = load_model(file_path)

    def predict(self, x):
        pred = np.zeros(shape=(x.shape[0], self.models[0].output_shape[1]))
        for model in self.models:
            pred += model.predict(x)
        pred = pred / len(self.models)
        return pred

    def get_last_layer_activation_vectors(self, x):
        av = np.zeros(shape=(x.shape[0], self.models[0].output_shape[1]))
        for model in self.models:
            pen_layer_output = K.function([model.layers[0].input], [model.layers[-2].output])
            av += pen_layer_output(x)[0]
        av = av / len(self.models)
        return av 
    
    def predict_openmax(self, x, mav_list, weibull_models, alpha):
        openmax_scores = np.zeros(shape=(x.shape[0], self.models[0].output_shape[1] + 1))
        alpha = alpha # number of top classes to revise
        # for each sample in x
        for n in range(len(x)):
            av = self.get_last_layer_activation_vectors(np.expand_dims(x[n], axis=0))[0]
            ranks = np.argsort(av)[::-1]
            y_hats = []
            y_hat_unknown = 0 # score for the unknown class
            # for each known class
            for i in range(n_classes):
                dist = np.linalg.norm(av - mav_list[i])
                shape, loc, scale = weibull_models[i]
                weibull_score = weibull_min.cdf(dist, shape, loc, scale)
                r_i = np.where(ranks==i)[0][0]
                R_alpha = max(0, ((alpha - r_i) / alpha))
                w_i = 1 - R_alpha * weibull_score # probability of belonging to the given class
                # w_i = 1 - weibull_score  
                y_hat_i = av[i] * w_i  # openmax score for the given class
                y_hats.append(y_hat_i)
                y_hat_unknown += av[i] * (1 - w_i) 
            
            # append the unknown class score
            y_hats.append(y_hat_unknown)
            openmax_scores[n] = softmax(y_hats)
        
        return openmax_scores

# Create the ensemble
num_ensembles = 5
in_shape = x_train[0].shape
n_epochs = 250
batch_size = 128
models = []
callbacks = []
for i in range(num_ensembles):
    model, callbacks_ = InceptionTime(in_shape, n_classes, model_num=i)
    models.append(model)
    callbacks.append(callbacks_)

val_data = (x_test_masked, to_categorical(y_test))
inception_ensemble = InceptionTime_Ensemble(models, callbacks, x_train_masked, to_categorical(y_train), n_epochs, batch_size, val_data)

# Load model weights
folder_path = os.path.join(MODELS_PATH, f'{dataset}')
inception_ensemble.load_weights(folder_path)

# Load MAVs
mav_path = os.path.join(MODELS_PATH, f'{dataset}/{dataset}_MAVs.pkl')
with open(mav_path, 'rb') as f:
    mav_list = pickle.load(f)

# Load Weibull models
weibull_path = os.path.join(MODELS_PATH, f'{dataset}/{dataset}_Weibull.pkl')
with open(weibull_path, 'rb') as f:
    weibull_models = pickle.load(f)


# Testing on the known samples
alpha = int(sys.argv[2])
preds_test = inception_ensemble.predict_openmax(x_test_masked, mav_list, weibull_models, alpha)
print(f'OpenMax-InceptionTime classification report for the test set of the {dataset} dataset')
print(classification_report(y_test, np.argmax(preds_test, axis=1)))
cm=confusion_matrix(y_test, np.argmax(preds_test, axis=1))
print(cm)

# Iterate through every given dataset to be used as unknowns
for unk_dataset in sys.argv[3:]:

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
    preds_unk = inception_ensemble.predict_openmax(x_unk, mav_list, weibull_models, alpha)
    preds_combined = np.concatenate((preds_test, preds_unk), axis=0)
    print(f'Classification Result for the {dataset} combined with {unk_dataset} Dataset as unknown')
    print(classification_report(y_combined, np.argmax(preds_combined, axis=1)))
    cm = confusion_matrix(y_combined, np.argmax(preds_combined, axis=1))
    print(cm, cm.shape)
    print('-------------')





















