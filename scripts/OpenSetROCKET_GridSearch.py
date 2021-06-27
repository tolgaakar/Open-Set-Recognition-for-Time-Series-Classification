# $python OpenSetROCKET_GridSearcg.py <dataset_name>

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

# Paths should be defined here
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
# if the transformations are already computed
if os.path.isfile(rocket_file):
    rocket = joblib.load(rocket_file)
    x_train_transform = pd.read_pickle(f'{rocket_dir}/{dataset}_train_ROCKET_transformed.pkl')
    x_test_transform = pd.read_pickle(f'{rocket_dir}/{dataset}_test_ROCKET_transformed.pkl')

else:
    x_train_transform = to_sktime_dataset(x_train)
    x_test_transform = to_sktime_dataset(x_test)
    rocket = Rocket()
    rocket.fit(x_train_transform)
    # Save the transformer
    os.makedirs(rocket_dir, exist_ok=True)
    joblib.dump(rocket, os.path.join(rocket_dir, f'{dataset}_ROCKET.joblib'))
    x_train_transform = rocket.transform(x_train_transform)
    x_test_transform = rocket.transform(x_test_transform)
print('ROCKET transformation is complete')



# Save the transformations
x_train_transform.to_pickle(os.path.join(rocket_dir, f'{dataset}_train_ROCKET_transformed.pkl'))
x_test_transform.to_pickle(os.path.join(rocket_dir, f'{dataset}_test_ROCKET_transformed.pkl'))


# Fit classifier

classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
classifier.fit(x_train_transform, y_train)

preds = classifier.predict(x_test_transform)


print('Closed Set Model Classification Report')
print(classification_report(y_test, preds))

with open(f'{dataset}_ROCKET_Ridge_gridsearch.pkl', 'wb') as f:
    pickle.dump(classifier, f)




# Group by target, i.e. one split for each class
splits = np.split(x_train, np.unique(y_train, return_index = True)[1][1:])

## Use this if the barycenters are not pre-computed
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


# Augmented Unknowns
# Use this function to create the augmented ones if they are not done previously
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

def load_augmented_data(dataset_name):
    aug_data_path = os.path.join(AUG_PATH, f"{dataset_name}/{dataset_name}_train_augmented.npy")
    aug_data = np.load(aug_data_path)
    return aug_data



# aug_train = get_augmented_data(x_train)
# aug_test = get_augmented_data(x_test)
aug_train = load_augmented_data(dataset)


aug_train[np.isnan(aug_train)] = mask_value
aug_train_transform = to_sktime_dataset(aug_train)
aug_train_transform = rocket.transform(aug_train_transform)
print('Augmented data has been created')

# Grid Search for the coefficients of the thresholds, alpha and beta

# values = np.linspace(0, 1, num=11, endpoint=True)
values = [0, 0.5, 0.75, 1, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 4]

preds_kn = classifier.predict(x_train_transform)
preds_un = classifier.predict(aug_train_transform)
closed_set_acc = accuracy_score(y_train, preds_kn)

acc_list = []

for v1 in tqdm(values):
    for v2 in values:
        STD_COEF_1 = v1
        STD_COEF_2 = v2

        # Set the distance threshold, alpha
        top_distances_for_class = []
        for dist_for_class in distances_all:
            mean = np.median(dist_for_class)
            std = np.std(dist_for_class)
            top_distances_for_class.append(mean + STD_COEF_1*std)


        # Set the cross-correlation threshold, beta
        bottom_cc_for_class = []
        for cc_for_class in cc_all:
            mean = np.median(cc_for_class, axis=0)
            std = np.std(cc_for_class, axis=0)
            bottom_cc_for_class.append(mean - STD_COEF_2*std)


        modified_preds = modify_preds(x_train, bc_list, preds_kn.copy())
        acc_kn = accuracy_score(y_train, modified_preds)

        aug_labels = np.ones(len(aug_train)) * n_classes
        modified_preds = modify_preds(aug_train, bc_list, preds_un.copy())
        acc_un = accuracy_score(aug_labels, modified_preds)

        score = (acc_kn * acc_un)/(acc_kn + acc_un)
        w_score = (acc_kn/closed_set_acc)**4 * score

        acc_list.append([STD_COEF_1, STD_COEF_2, acc_kn, acc_un, score, w_score])

acc_list = np.asarray(acc_list)
df = pd.DataFrame(acc_list)
df[0] = df[0].astype(str)
df.columns = ['Coef1', 'Coef2', 'Closed Set Acc', 'Unknown Acc', 'Score', 'WeightedScore']


optimals = acc_list[np.where(acc_list[:, 5] == np.max(acc_list[:, 5]))]
print('Optimal Combinations')
print(optimals)


# Plot the grid search

fig, ax = plt.subplots(figsize=(20,10))
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
# y = 1-x line
ax.plot(np.flip(lims), lims, '--', c='crimson')
# scatter plot
ax = sns.scatterplot(data=df, x='Unknown Acc', y='Closed Set Acc', hue='Coef1', style='Coef2', s=200)
plt.legend(bbox_to_anchor=(1.001, 1),borderaxespad=0)
for o in optimals:
    annotate_str = f'Coef1={o[0]}, Coef2={o[1]}'
    coords = (o[3], o[2])
    plt.annotate(annotate_str, coords)
plt.tight_layout()
plt.show()





