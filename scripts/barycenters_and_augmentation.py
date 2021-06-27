# $python barycenters_and_augmentation.py <dataset_name>

import os
import random
import numpy as np
import pandas as pd
from tslearn.preprocessing import TimeSeriesResampler, TimeSeriesScalerMinMax, TimeSeriesScalerMeanVariance
from tslearn.barycenters import softdtw_barycenter, dtw_barycenter_averaging_subgradient
from tslearn.utils import to_time_series, to_time_series_dataset, from_sktime_dataset, to_sktime_dataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sktime.utils.data_io import load_from_tsfile_to_dataframe

def get_augmented_data(x, mean=0, std=0.25):
    """Returns a distorted version of the train data to be used in hyperparameter optimization"""

    aug_data = x.copy()
    if np.unique(np.isnan(aug_data)):
        aug_data[np.isnan(aug_data)] = mean
    for i in range(len(x)):
        for j in range(x[0].shape[1]):
            noise = np.random.normal(mean, std, len(x_train[0]))
            aug_data[i, :, j] += noise

    cut_idx = int(aug_data.shape[1] / 2)
    temp1 = aug_data[:, cut_idx:]
    temp2 = aug_data[:, :cut_idx]

    aug_data = np.hstack((temp1, temp2))
    
    return np.flip(aug_data)


# Paths should be defined here
DATA_PATH = 'Multivariate_ts'
BC_PATH = 'barycenters'
AUG_PATH = 'augmented'
datasets = sys.argv[1] # Alternatively a list of datasets can be passed as well

for dataset in datasets:

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

    print(f'{dataset} loaded')

    # Normalize data
    scalers = []
    for i in range(x_train.shape[-1]):
        scaler = StandardScaler()
        x_train[:,:,i] = scaler.fit_transform(x_train[:,:,i])
        x_test[:,:,i] = scaler.transform(x_test[:,:,i])
        scalers.append(scaler)

    n_classes = len(np.unique(y_train))

    # Mask NaN values

    mask_value = 0
    x_train[np.isnan(x_train)] = mask_value
    x_test[np.isnan(x_test)] = mask_value

    # Calculate Barycenters

    # Sort by the target values so that it can be split for each class 
    sorted_idxs = np.argsort(y_train, axis=0)
    x_train = x_train[sorted_idxs]
    y_train = y_train[sorted_idxs]

    # Group by target, i.e. one split for each class
    splits = np.split(x_train, np.unique(y_train, return_index = True)[1][1:])

    bc_dir = f'barycenters/{dataset}'
    os.makedirs(bc_dir, exist_ok=True)
    bc_list = [] 

    # Calculate barycenters for each class (split)
    print(f'Calculating barycenters for {dataset}...')
    c_ = 0
    for split in splits:
        bc = softdtw_barycenter(split) # barycenters for the given class 
        bc_list.append(bc)
        np.save(os.path.join(bc_dir, f'{dataset}_bc_{c_}'), bc) # save files
        c_ += 1

    print(f'Creating augmented samples for {dataset}...')
    aug_train = get_augmented_data(x_train)
    aug_train[np.isnan(aug_train)] = mask_value

    # Save files
    aug_dir = f'augmented/{dataset}'
    os.makedirs(aug_dir, exist_ok=True)
    np.save(os.path.join(aug_dir, f'{dataset}_train_augmented'), aug_train)
















