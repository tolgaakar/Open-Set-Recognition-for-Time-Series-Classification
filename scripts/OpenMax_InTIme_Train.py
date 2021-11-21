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

inception_ensemble.fit()

# Closed set classification report 
closed_set_preds = inception_ensemble.predict(x_test_masked)
closed_set_acc = accuracy_score(y_test, np.argmax(closed_set_preds, axis=1))
print(f'Closed set classification report for the test set of the {dataset} dataset')
print(classification_report(y_test, np.argmax(closed_set_preds, axis=1)))

# OpenMax

def get_correct_classified(pred, y):
    """
    Returns the indices of correctly classified samples
    """
    pred = (pred > 0.5) * 1
    correct_idxs = np.all(pred == y, axis=1)
    return correct_idxs

pred = inception_ensemble.predict(x_train_masked)
idxs = get_correct_classified(pred, to_categorical(y_train))
x_correct = x_train_masked[idxs]
y_correct = y_train[idxs]
x_correct.shape, y_correct.shape

def split_data_by_class(x, y):
    """
    Returns sorted splits for each class
    """
    # Sort by the target values so that it can be split for each class 
    sorted_idxs = np.argsort(y, axis=0)
    x = x[sorted_idxs]
    y = y[sorted_idxs]

    # Group by target, i.e. one split for each class
    splits = np.split(x, np.unique(y, return_index = True)[1][1:])
    print(f"Number of splits: {len(splits)}")
    return splits

splits = split_data_by_class(x_correct, y_correct)
if len(splits) != n_classes:
    print(f'There are {n_classes - len(splits)} classes with zero correctly classified samples')
    print('It is not possible to fit Weibull model for each class')
    sys.exit()


def calc_MAVs(splits):
    """
    Takes a class split and returns Activation Vectors for each sample and Mean Activation Vectors for each class.
    """

    # Calculate MAVs for each class (split) 
    mav_list = []
    av_list = []
    
    for split in splits:
       activation_vectors = inception_ensemble.get_last_layer_activation_vectors(split)
       av_list.append(activation_vectors)
       mav = np.mean(activation_vectors, axis=0)  # MAV for the given class
       mav_list.append(mav)

    # Save MAVs
    mav_path = os.path.join(MODELS_PATH, f'{dataset}/{dataset}_MAVs.pkl')
    with open(mav_path, 'wb') as f:
        pickle.dump(mav_list, f)
    
    return av_list, mav_list

av_list, mav_list = calc_MAVs(splits)

# Calculate distances to MAVs

def calc_distances(activations, mav):
    """
    Returns the distances between the activations of correctly classified samples and MAV of the given class
    """
    distances = np.empty(len(activations))
    for i in range(len(activations)):
        distances[i] = np.linalg.norm(activations[i] - mav)
    return distances

def get_top_distances(distances, ratio=0.1):
    """
    Returns the top r% largest distances of the given array
    """
    sorted_dists = np.sort(distances)
    return sorted_dists[-round(len(distances)*ratio):]

top_distances_all = []
for c_i in range(n_classes):
    # distances between each sample from the given class and its MAV
    distances = calc_distances(av_list[c_i], mav_list[c_i])
    top_dists = get_top_distances(distances)
    top_distances_all.append(top_dists)


def fit_weibull(distances_all, n_classes):
    """
    Returns one Weibull model (set of parameters) for each class
    """
    weibull_models = []
    for i in range(n_classes):
        shape, loc, scale = weibull_min.fit(distances_all[i])
        weibull_models.append([shape, loc, scale])

    # Save Weibull models
    weibull_path = os.path.join(MODELS_PATH, f'{dataset}/{dataset}_Weibull.pkl')
    with open(weibull_path, 'wb') as f:
        pickle.dump(weibull_models, f)

    return weibull_models


weibull_models = fit_weibull(top_distances_all, n_classes)


# Grid Search
print('\n Starting grid search for the optimal alpha value (top n classes to recalibrate)')


def load_augmented_data(dataset):
    aug_data_path = os.path.join(AUG_PATH, f"{dataset}/{dataset}_train_augmented.npy")
    aug_data = np.load(aug_data_path)
    return aug_data

aug_train = load_augmented_data(dataset)
values = [2, int(n_classes*0.25), int(n_classes*0.5), int(n_classes*0.75), n_classes]
values = np.unique(values) # Top 2 classes, top 25%, top 50%, top 75% classes and all classes


acc_list = []
for alpha in tqdm(values):
    preds_kn = inception_ensemble.predict_openmax(x_train_masked, mav_list, weibull_models, alpha)
    acc_kn = accuracy_score(y_train, np.argmax(preds_kn, axis=1))

    aug_labels = np.ones(len(aug_train)) * n_classes
    preds_aug = inception_ensemble.predict_openmax(aug_train, mav_list, weibull_models, alpha)
    acc_aug = accuracy_score(aug_labels, np.argmax(preds_aug, axis=1))

    score = (acc_kn * acc_aug)/(acc_kn + acc_aug)
    w_score = (acc_kn/closed_set_acc)**4 * score

    acc_list.append([alpha, acc_kn, acc_aug, w_score])

acc_list = np.asarray(acc_list)
df = pd.DataFrame(acc_list)
df[0] = df[0].astype(str)
df.columns = ['Alpha', 'Acc for Knowns', 'Recall for Unknowns', 'WeightedScore']

optimals = acc_list[np.where(acc_list[:, -1] == np.max(acc_list[:, -1]))]
print('\n Optimal Alpha Value(s)')
print(optimals)



















