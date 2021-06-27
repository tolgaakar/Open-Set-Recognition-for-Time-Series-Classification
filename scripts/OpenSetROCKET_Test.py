import os
import sys
import numpy as np
import random
import math
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import pickle
from tslearn.preprocessing import TimeSeriesResampler, TimeSeriesScalerMinMax, TimeSeriesScalerMeanVariance
from tslearn.datasets import UCR_UEA_datasets
from tslearn.barycenters import softdtw_barycenter, dtw_barycenter_averaging_subgradient
from tslearn.metrics import soft_dtw, dtw, cdist_ctw, dtw_path_from_metric, cdist_soft_dtw
from tslearn.utils import to_time_series, to_time_series_dataset, from_sktime_dataset, to_sktime_dataset
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import weibull_min, genpareto
import scipy.spatial.distance as distance
from scipy.spatial.distance import cosine, correlation, cityblock
from scipy.signal import correlate
import scipy.optimize as optimize
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sktime.transformations.panel.rocket import Rocket
from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV
from sklearn.svm import SVC
from sktime.utils.data_processing import from_3d_numpy_to_nested
import joblib
import warnings
warnings.filterwarnings("ignore")


# Load dataset

# dataset = "CharacterTrajectories"
DATA_PATH = 'Multivariate_ts'
BC_PATH = 'barycenters'
AUG_PATH = 'augmented'
TRANSFORM_PATH = 'ROCKET_transforms'
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

mask_value = 0
x_train[np.isnan(x_train)] = mask_value
x_test[np.isnan(x_test)] = mask_value

# Sort by the target values so that it can be split for each class 
sorted_idxs = np.argsort(y_train, axis=0)
x_train = x_train[sorted_idxs]
y_train = y_train[sorted_idxs]

# ROCKET Transform

rocket_dir = f'{TRANSFORM_PATH}/{dataset}'
rocket_file = f'{rocket_dir}/{dataset}_ROCKET.joblib'
if os.path.isfile(rocket_file):
    rocket = joblib.load(rocket_file)
    x_train_transform = pd.read_pickle(f'{rocket_dir}/{dataset}_train_ROCKET_transformed.pkl')
    x_test_transform = pd.read_pickle(f'{rocket_dir}/{dataset}_test_ROCKET_transformed.pkl')

else:
    x_train_transform = to_sktime_dataset(x_train)
    x_test_transform = to_sktime_dataset(x_test)
    rocket = Rocket()
    rocket.fit(x_train_transform)
    x_train_transform = rocket.transform(x_train_transform)
    x_test_transform = rocket.transform(x_test_transform)
print('ROCKET transformation is complete')


# Fit classifier

classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
classifier.fit(x_train_transform, y_train)

preds = classifier.predict(x_test_transform)
closed_set_acc = accuracy_score(y_test, preds)

print('Closed Set Model Classification Report')
print(classification_report(y_test, preds))

with open(f'{dataset}_ROCKET_Ridge.pkl', 'wb') as f:
    pickle.dump(classifier, f)



# Group by target, i.e. one split for each class
splits = np.split(x_train, np.unique(y_train, return_index = True)[1][1:])

# Calculate barycenters for each class (split)
# bc_list = [] 
# for split in splits:
#     bc = softdtw_barycenter(split) # barycenters for the given class 
#     bc_list.append(bc)

def load_barycenters(dataset_name):
    bc_list = []
    for c in range(n_classes):
        bc_data_path = os.path.join(BC_PATH, f"{dataset_name}/{dataset_name}_bc_{c}.npy")
        bc = np.load(bc_data_path)
        bc_list.append(bc)
    return bc_list

bc_list = load_barycenters(dataset)
print('Barycenters are loaded')



# Calculate DTW distances 

distances_all = []
for i in range(n_classes):
    distances_for_class = []
    for x in splits[i]:
        distance = dtw_path_from_metric(x, bc_list[i], metric='sqeuclidean')[1]
        distances_for_class.append(math.log(distance))
    distances_all.append(distances_for_class)


STD_COEF_1 = float(sys.argv[2])

top_distances_for_class = []
for dist_for_class in distances_all:
    mean = np.median(dist_for_class)
    std = np.std(dist_for_class)
    top_distances_for_class.append(mean + STD_COEF_1*std)


def calc_distances_to_each_bc(x, bc_list):
    distances = []
    for bc in bc_list:
        dist = dtw_path_from_metric(x, bc, metric='sqeuclidean')[1]
        distances.append(math.log(dist))
    return np.asarray(distances)


# Calculate cross-correlation

cc_all = []
for i in range(n_classes):
    cc_for_each_class = []
    for x in splits[i]:
        cc_for_each_dim = []
        for c in range(x_train.shape[2]):
            cc = np.max(correlate(x[:, c], bc_list[i][:, c]))
            cc_for_each_dim.append(cc)
        cc_for_each_class.append(cc_for_each_dim)
    cc_all.append(cc_for_each_class)


STD_COEF_2 = float(sys.argv[3])

bottom_cc_for_class = []
for cc_for_class in cc_all:
    mean = np.median(cc_for_class, axis=0)
    std = np.std(cc_for_class, axis=0)
    bottom_cc_for_class.append(mean - STD_COEF_2*std)




def unknown_detector(x, bc_list):
    """Returns True if the sample is further away (DTW distance) from each class than the most extreme train samples"""

    bc_distances = calc_distances_to_each_bc(x, bc_list)
    #print(bc_distances)
    if all(bc_distances > top_distances_for_class):
        return True
    else:
        #print(np.where(bc_distances < top_distances_for_class))
        return False

def cc_unknown_detector(x, bc_list):
    """Returns True if the sample is not cross-correlated with any of the known classes"""

    ccs = []
    for i in range(n_classes):
        __ = []
        for j in range(x.shape[1]):
            cc = np.max(correlate(x[:, j], bc_list[i][:, j]))
            __.append(cc)
        ccs.append(__)
    ccs = np.asarray(ccs)
    
    cnt = 0
    flags = (ccs > bottom_cc_for_class)
    # print(flags)
    for row in flags:
        if np.isin(False, row):
            cnt += 1

    if cnt == n_classes:
        return True
    else:
        return False



def modify_preds(x, bc_list, preds):
    """Modify the prediction to be unknown if one of the detectors return True"""

    for n in range(len(x)):
        if cc_unknown_detector(x[n], bc_list) or unknown_detector(x[n], bc_list):
            preds[n] = n_classes
    return preds

# Results for the known dataset
preds = classifier.predict(x_test_transform)
modified_preds = modify_preds(x_test, bc_list, preds.copy())
print('Open Set Model Classification Report')
print(classification_report(y_test, modified_preds))
cm = confusion_matrix(y_test, modified_preds)
print(cm, cm.shape)
print('-------------')

# Iterate through every given dataset to be used as unknowns
for unk_dataset in sys.argv[4:]:
    
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

    # For large ones
    # loader = UCR_UEA_datasets()
    # _, _, unk_test, _ = loader.load_dataset(unk_dataset)
    # x_unk = unk_test[np.random.choice(unk_test.shape[0], size=len(x_test), replace=False), :]

    y_unk = np.ones((len(x_unk))) * n_classes # The label of the unknowns will be equal to the number of classes

    # Resample the lenght of the time series
    x_unk = TimeSeriesResampler(sz=x_train.shape[1]).fit_transform(x_unk[:, :, :x_train.shape[2]])

    # Normalize each channel
    for i in range(x_unk.shape[-1]):
        x_unk[:,:,i] = scalers[i].transform(x_unk[:,:,i]) 

    # Mask NaN values
    x_unk[np.isnan(x_unk)] = mask_value

    x_unk_transform = to_sktime_dataset(x_unk)
    x_unk_transform = rocket.transform(x_unk_transform)

    # Results for the given dataset
    preds_unk = classifier.predict(x_unk_transform)
    modified_preds_unk = modify_preds(x_unk, bc_list, preds_unk.copy())
    print('Accuracy on the unknown samples:', accuracy_score(y_unk, modified_preds_unk))

    # Known and unknwon combined results
    x_combined = np.concatenate((x_test, x_unk), axis=0)
    x_combined_transform = np.concatenate((x_test_transform, x_unk_transform), axis=0)
    y_combined = np.concatenate((y_test, y_unk), axis=0)

    preds_combined = classifier.predict(x_combined_transform)
    modified_preds = modify_preds(x_combined, bc_list, preds_combined.copy())
    print(f'Classification Result for the {dataset} combined with {unk_dataset} Dataset as unknown')
    print(classification_report(y_combined, modified_preds))
    cm=confusion_matrix(y_combined, modified_preds)
    print(cm, cm.shape)
    print('-------------')











