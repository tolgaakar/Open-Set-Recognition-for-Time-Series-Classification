## !! pip install tensorflow==2.4.1 required !! ##

import os
import sys
import numpy as np
import random
from tqdm import tqdm
import tensorflow as tf
import math
import pickle
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Input, Conv1D, MaxPooling1D, \
Dropout, BatchNormalization, Activation, Concatenate, Masking, GlobalAveragePooling1D, \
Conv1DTranspose, Reshape, PReLU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback, ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tslearn.preprocessing import TimeSeriesResampler, TimeSeriesScalerMinMax, TimeSeriesScalerMeanVariance
from tslearn.datasets import UCR_UEA_datasets
from tslearn.utils import to_time_series, to_time_series_dataset, from_sktime_dataset, to_sktime_dataset
from sktime.utils.data_io import load_from_tsfile_to_dataframe
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.spatial.distance as distance
import scipy.optimize as optimize
import tensorflow_probability as tfp
import warnings
warnings.filterwarnings("ignore")


latent_dim = 8
n_epochs = 500
batch_size = 128
gau_threshold = 0.5


# Load dataset

# Paths should be defined here
DATA_PATH = 'Multivariate_ts'
AUG_PATH = 'augmented'
MODELS_PATH = '/content/drive/MyDrive/Master Thesis/Models/LCVAE'
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
    scaler = MinMaxScaler()
    x_train[:,:,i] = scaler.fit_transform(x_train[:,:,i])
    x_test[:,:,i] = scaler.transform(x_test[:,:,i])
    scalers.append(scaler)
    

# Mask NaN values

mask_value = -1e-8   # NaN values will be masked for a constant length

x_train_masked = x_train.copy()
x_test_masked = x_test.copy()
x_train_masked[np.isnan(x_train_masked)] = mask_value
x_test_masked[np.isnan(x_test_masked)] = mask_value


in_shape = x_train[0].shape 
length, n_channels = in_shape

class Sampling(layers.Layer):
    """Uses (z_mean, z_var) to sample z, the vector encoding a digit 
    with Reparametrization Trick."""

    def call(self, inputs):
        z_mean, z_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + (z_var ** 0.5) * epsilon


def ConvBlock(inputs, out_ch, kernel_size, strides, padding, latent_dim):
    """Applies 1D convolutions, and then calculates mu and var"""

    x = inputs
    h = Conv1D(out_ch, kernel_size, strides=strides, padding=padding)(x)
    h = BatchNormalization()(h)
    h = PReLU()(h)
    h_flat = Flatten()(h)
    mu = Dense(latent_dim)(h_flat)
    var = Dense(latent_dim)(h_flat)
    var = K.softplus(var) + 1e-8
    return h, mu, var


def TransposeConvBlock(inputs, in_ch, out_ch, kernel_size, strides, padding, unflat_dim, latent_dim):
    """Applies transpose 1D convolutions, and then calculates mu and var"""

    x = inputs
    x = Dense(unflat_dim * in_ch)(x)
    x = Reshape((unflat_dim, in_ch))(x)
    h = PReLU()(x)
    h = Conv1DTranspose(out_ch, kernel_size, strides=strides, padding=padding)(h)
    h = BatchNormalization()(h)
    h_flat = Flatten()(h)
    mu = Dense(latent_dim)(h_flat)
    var = Dense(latent_dim)(h_flat)
    var = K.softplus(var) + 1e-8
    return h, mu, var 


def FinalConvBlock(inputs, in_ch, out_ch, kernel_size, strides, padding, unflat_dim):
    """Transpose 1D convolutions at the end of the decoder, creates the reconstructed sample"""

    x = inputs
    x = Dense(unflat_dim * in_ch)(x)
    x = Reshape((unflat_dim, in_ch))(x)
    x = PReLU()(x)
    x_rec = Conv1DTranspose(out_ch, kernel_size, strides=strides, padding=padding)(x)
    x_rec = K.tanh(x_rec)
    return x_rec

# Define the LCVAE Model 

encoder_inputs = keras.Input(shape=in_shape)
masking = Masking(mask_value=mask_value, input_shape=in_shape)(encoder_inputs)

# Encoder (Upward path)
# Block 1
enc1_1, mu_up1_1, var_up1_1 = ConvBlock(masking, 64, 1, 1, 'same', latent_dim*4)
enc1_2, mu_up1_2, var_up1_2 = ConvBlock(enc1_1, 64, 3, 2, 'same', latent_dim*4)

# Block 2
enc2_1, mu_up2_1, var_up2_1 = ConvBlock(enc1_2, 128, 3, 2, 'same', latent_dim*2)
enc2_2, mu_up2_2, var_up2_2 = ConvBlock(enc2_1, 128, 3, 2, 'same', latent_dim*2)

# Block 3
enc3_1, mu_up3_1, var_up3_1 = ConvBlock(enc2_2, 256, 3, 2, 'same', latent_dim)
enc3_2, z_mu, z_var = ConvBlock(enc3_1, 256, 3, 2, 'same', latent_dim)

z = Sampling()([z_mu, z_var])

# Classifier
pred = Dense(n_classes)(z)
pred = Activation('softmax')(pred)

# Block 3
dec3_1, mu_dn3_1, var_dn3_1 = TransposeConvBlock(z, 256, 256, 2, 2, 'same', enc3_2.shape[1], latent_dim)
prec_up3_1 = var_up3_1 ** (-1)
prec_dn3_1 = var_dn3_1 ** (-1)
qvar3_1 = (prec_up3_1 + prec_dn3_1) ** (-1)
qmu3_1 = (mu_up3_1 * prec_up3_1 + mu_dn3_1 * prec_dn3_1) * qvar3_1
de_latent3_1 = Sampling()([qmu3_1, qvar3_1])

dec2_2, mu_dn2_2, var_dn2_2 = TransposeConvBlock(de_latent3_1, 256, 128, 1, 2, 'same', enc3_1.shape[1], latent_dim*2)
prec_up2_2 = var_up2_2 ** (-1)
prec_dn2_2 = var_dn2_2 ** (-1)
qvar2_2 = (prec_up2_2 + prec_dn2_2) ** (-1)
qmu2_2 = (mu_up2_2 * prec_up2_2 + mu_dn2_2 * prec_dn2_2) * qvar2_2
de_latent2_2 = Sampling()([qmu2_2, qvar2_2])

# Block 2
dec2_1, mu_dn2_1, var_dn2_1 = TransposeConvBlock(de_latent2_2, 128, 128, 2, 2, 'same', enc2_2.shape[1], latent_dim*2)
prec_up2_1 = var_up2_1 ** (-1)
prec_dn2_1 = var_dn2_1 ** (-1)
qvar2_1 = (prec_up2_1 + prec_dn2_1) ** (-1)
qmu2_1 = (mu_up2_1 * prec_up2_1 + mu_dn2_1 * prec_dn2_1) * qvar2_1
de_latent2_1 = Sampling()([qmu2_1, qvar2_1])

dec1_2, mu_dn1_2, var_dn1_2 = TransposeConvBlock(de_latent2_1, 128, 64, 1, 2, 'same', enc2_1.shape[1], latent_dim*4)
prec_up1_2 = var_up1_2 ** (-1)
prec_dn1_2 = var_dn1_2 ** (-1)
qvar1_2 = (prec_up1_2 + prec_dn1_2) ** (-1)
qmu1_2 = (mu_up1_2 * prec_up1_2 + mu_dn1_2 * prec_dn1_2) * qvar1_2
de_latent1_2 = Sampling()([qmu1_2, qvar1_2])

# Block 1
dec1_1, mu_dn1_1, var_dn1_1 = TransposeConvBlock(de_latent1_2, 64, 64, 2, 2, 'same', enc1_2.shape[1], latent_dim*4)
prec_up1_1 = var_up1_1 ** (-1)
prec_dn1_1 = var_dn1_1 ** (-1)
qvar1_1 = (prec_up1_1 + prec_dn1_1) ** (-1)
qmu1_1 = (mu_up1_1 * prec_up1_1 + mu_dn1_1 * prec_dn1_1) * qvar1_1
de_latent1_1 = Sampling()([qmu1_1, qvar1_1])

x_rec = FinalConvBlock(de_latent1_1, 64, n_channels, 1, 1, 'same', length)


outputs = [z, z_mu, z_var, \
           qmu3_1, qvar3_1, qmu2_2, qvar2_2, qmu2_1, qvar2_1, qmu1_2, qvar1_2, qmu1_1, qvar1_1, \
           pred, x_rec, \
           mu_dn3_1, var_dn3_1, mu_dn2_2, var_dn2_2, mu_dn2_1, var_dn2_1, mu_dn1_2, var_dn1_2, mu_dn1_1, var_dn1_1]

lnet = Model(encoder_inputs, outputs, name=f"LCVAE_Model_{dataset}")
#lnet.summary()


# Label encoder that maps the labels into the latent space

y_one_hot_input = keras.Input(shape=(n_classes,))
y_latent = layers.Dense(latent_dim, name='y_latent')(y_one_hot_input)
label_encoder = keras.Model(y_one_hot_input, y_latent, name='label_encoder')
#label_encoder.summary()

class DeterministicWarmup(object):
    """Generates the iterable warm up object to determine the value of beta in the loss function"""

    def __init__(self, n=100, t_max=1):
        self.t = 0
        self.t_max = t_max
        self.inc = 1 / n

    def __iter__(self):
        return self

    def __next__(self):
        t = self.t + self.inc
        self.t = self.t_max if t > self.t_max else t  # 0->1
        return self.t

m = tf.keras.metrics.Accuracy()
warmup = DeterministicWarmup(n=n_epochs, t_max=1)

def kl_normal(qm, qv, pm, pv, yh):  
    """ Calculates the KL Loss"""
    
    element_wise = 0.5 * (tf.math.log(pv) - tf.math.log(qv) + qv / pv + tf.square(qm - pm - yh) / pv - 1)
    kl = tf.reduce_sum(element_wise, -1)
    return kl


class LCVAE(keras.Model):
    """ Creates the Model object for the LCVAE"""

    def __init__(self, lnet, label_encoder, **kwargs):
        super(LCVAE, self).__init__(**kwargs)
        self.lnet = lnet
        self.label_encoder = label_encoder
        self.beta = K.variable(value = warmup.t)
        self.beta.__trainable = False


    def call(self, inputs):
        
        lnet_outputs = self.lnet(inputs)
        return lnet_outputs

    
    def train_step(self, data):
        if isinstance(data, tuple):
            x, y = data
        
        with tf.GradientTape() as tape:
            z, z_mu, z_var, \
            qmu3_1, qvar3_1, \
            qmu2_2, qvar2_2, qmu2_1, qvar2_1, qmu1_2, qvar1_2, qmu1_1, qvar1_1, \
            pred, x_rec, \
            pmu3_1, pvar3_1, \
            pmu2_2, pvar2_2, pmu2_1, pvar2_1, pmu1_2, pvar1_2, pmu1_1, pvar1_1 = self.call(x)


            # Classification Loss
            clf_loss = tf.reduce_mean(
                keras.losses.categorical_crossentropy(y, pred)
            )

            # Reconstruction Loss
            reconstruction_loss = tf.reduce_mean(
                tf.norm((x - x_rec), ord=1)
            )

            # KL Divergence 
            y_latent = self.label_encoder(y)
            pm, pv = tf.zeros(tf.shape(z_mu)), tf.ones(tf.shape(z_var))
            kl_latent = kl_normal(z_mu, z_var, pm, pv, y_latent)
            kl3_1 = kl_normal(qmu3_1, qvar3_1, pmu3_1, pvar3_1, 0)
            kl2_2 = kl_normal(qmu2_2, qvar2_2, pmu2_2, pvar2_2, 0)
            kl2_1 = kl_normal(qmu2_1, qvar2_1, pmu2_1, pvar2_1, 0)
            kl1_2 = kl_normal(qmu1_2, qvar1_2, pmu1_2, pvar1_2, 0)
            kl1_1 = kl_normal(qmu1_1, qvar1_1, pmu1_1, pvar1_1, 0)


            kl_loss = (kl_latent + kl3_1 + kl2_2 + kl2_1 + kl1_2 + kl1_1) / 6 # mean kl loss for the layers
            kl_loss = kl_loss
            kl_loss = tf.reduce_mean(kl_loss) # mean kl loss for the batch
            

            total_loss = 100*clf_loss + reconstruction_loss + self.beta*kl_loss ####

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Classification Accuracy
        m.update_state(tf.math.argmax(y, axis=1), tf.math.argmax(pred, axis=1))
        return {
            "total_loss": total_loss,
            "accuracy": m.result(),
            "classification_loss": clf_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def test_step(self, data):
        if isinstance(data, tuple):
            x, y = data

        z, z_mu, z_var, \
        qmu3_1, qvar3_1, \
        qmu2_2, qvar2_2, qmu2_1, qvar2_1, qmu1_2, qvar1_2, qmu1_1, qvar1_1, \
        pred, x_rec, \
        pmu3_1, pvar3_1, \
        pmu2_2, pvar2_2, pmu2_1, pvar2_1, pmu1_2, pvar1_2, pmu1_1, pvar1_1 = self.call(x)


        # Classification Loss
        clf_loss = tf.reduce_mean(
            keras.losses.categorical_crossentropy(y, pred)
        )

        # Reconstruction Loss
        reconstruction_loss = tf.reduce_mean(
            tf.norm((x - x_rec), ord=1)
        )

        
        # KL Divergence 
        y_latent = self.label_encoder(y)
        pm, pv = tf.zeros(tf.shape(z_mu)), tf.ones(tf.shape(z_var))
        kl_latent = kl_normal(z_mu, z_var, pm, pv, y_latent)
        kl3_1 = kl_normal(qmu3_1, qvar3_1, pmu3_1, pvar3_1, 0)
        kl2_2 = kl_normal(qmu2_2, qvar2_2, pmu2_2, pvar2_2, 0)
        kl2_1 = kl_normal(qmu2_1, qvar2_1, pmu2_1, pvar2_1, 0)
        kl1_2 = kl_normal(qmu1_2, qvar1_2, pmu1_2, pvar1_2, 0)
        kl1_1 = kl_normal(qmu1_1, qvar1_1, pmu1_1, pvar1_1, 0)

        kl_loss = (kl_latent + kl3_1 + kl2_2 + kl2_1 + kl1_2 + kl1_1) / 6 # mean kl loss for the layers
        kl_loss = kl_loss
        kl_loss = tf.reduce_mean(kl_loss) # mean kl loss for the batch


        total_loss = 100*clf_loss + reconstruction_loss + self.beta*kl_loss ####
        return {
            "loss": total_loss,
            "accuracy": m.result(),
            "classification_loss": clf_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }  


        @property
        def metrics(self):
            # We list our `Metric` objects here so that `reset_states()` can be
            # called automatically at the start of each epoch
            # or at the start of `evaluate()`.
            # If you don't implement this property, you have to call
            # `reset_states()` yourself at the time of your choosing.
            return m


# Build model

lcvae = LCVAE(lnet, label_encoder)
lcvae.compile(optimizer=keras.optimizers.Adam())
lcvae.build((None, in_shape[0], in_shape[1]))
print(lcvae.summary())


# Increase beta from 0 to 1 linearly
wu_cb = LambdaCallback(on_epoch_begin=lambda epoch, log: K.set_value(lcvae.beta, next(warmup)))


reduce_lr = ReduceLROnPlateau(monitor='total_loss', factor=0.2, patience=5, min_lr=0.0001)

file_path = os.path.join(MODELS_PATH, f'{dataset}/{dataset}_LCVAE')
checkpoint = ModelCheckpoint(file_path, monitor='reconstruction_loss', 
                             verbose=0, save_best_only=True, save_weights_only=True, 
                             mode='min')


lcvae.fit(x_train_masked, to_categorical(y_train), epochs=n_epochs, batch_size=batch_size, 
        validation_data=(x_test_masked, to_categorical(y_test)), callbacks=[checkpoint, reduce_lr, wu_cb])

# Load best weights
lcvae.load_weights(file_path)

print(f'Closed set classification report for the test set of the {dataset} dataset')
preds = lcvae.predict(x_test_masked)[13]
print(classification_report(y_test, np.argmax(preds, axis=1)))


# Set reconstruction threshold

def load_augmented_data(dataset):
    aug_data_path = os.path.join(AUG_PATH, f"{dataset}/{dataset}_train_augmented.npy")
    aug_data = np.load(aug_data_path)
    return aug_data

aug_train = load_augmented_data(dataset)



def get_all_reconstruction_errors(train_set, model):
    rec_loss_list = []
    for x in train_set:
        x = np.expand_dims(x, axis=0)
        reconstruction = model.predict(x)[14]

        reconstruction_loss = tf.norm((x - reconstruction), ord=1)
        rec_loss_list.append(reconstruction_loss)
    
    return np.asarray(rec_loss_list)


rec_loss_train = get_all_reconstruction_errors(x_train_masked, lcvae)
rec_loss_train_sorted = np.sort(rec_loss_train) # sort ascending

rec_mean = np.mean(rec_loss_train)
rec_std = np.std(rec_loss_train)

print(f'Mean: {rec_mean}, Std: {rec_std}')

plt.hist(rec_loss_train_sorted, bins=25, histtype='stepfilled')
plt.title(f'Histogram of the Reconstruction Loss for the Train Set of the {dataset} dataset')
plt.xlabel('Reconstruction Loss')
plt.show()


rec_loss_aug = get_all_reconstruction_errors(aug_train, lcvae)
rec_loss_aug_sorted = np.sort(rec_loss_aug) # sort ascending

rec_mean = np.mean(rec_loss_aug)
rec_std = np.std(rec_loss_aug)

print(f'Mean: {rec_mean}, Std: {rec_std}')

plt.hist(rec_loss_aug_sorted, bins=25, histtype='stepfilled')
plt.title('Histogram of the Reconstruction Loss for the Known Unknowns')
plt.xlabel('Reconstruction Loss')
plt.show()

rec_thresh = (rec_loss_train_sorted[-1] + rec_loss_aug_sorted[0])/2
print('Reconstruction Threshold:', rec_thresh)


rec_loss_test = get_all_reconstruction_errors(x_test_masked, lcvae)
rec_loss_test_sorted = np.sort(rec_loss_test) # sort ascending

rec_mean = np.mean(rec_loss_test)
rec_std = np.std(rec_loss_test)

print(f'Mean: {rec_mean}, Std: {rec_std}')

plt.hist(rec_loss_test_sorted, bins=25, histtype='stepfilled')
plt.title('Histogram of the Reconstruction Loss for the Test Set')
plt.xlabel('Reconstruction Loss')
plt.show()


# Compute the Gaussian models

print('Computing the Gaussian models...')

def get_correct_classified(pred, y):
    """
    Returns the indices of correctly classified samples
    """
    pred = (pred > 0.5) * 1
    correct_idxs = np.all(pred == y, axis=1)
    return correct_idxs


outputs = lcvae.predict(x_train_masked)

z, z_mu, z_var, pred, x_rec = outputs[0], outputs[1], outputs[2], outputs[13], outputs[14]
idxs = get_correct_classified(pred, to_categorical(y_train))
z_correct = z[idxs] ###
y_correct = y_train[idxs]
z_correct.shape, y_correct.shape

# If the number of observations for a class is less than the number of features (latent_dim),
# covariance matrices can bu singular and thus return NaN values as distance to that class
print('Class labels:', np.unique(y_correct, return_counts=True)[0])
sample_counts_for_class = np.unique(y_correct, return_counts=True)[1]
print('Observation counts:', sample_counts_for_class)

if not all(sample_counts_for_class >= latent_dim):
    print(f'There are not enough observations for some classes to calculate the covariance matrix. Each class has to have at least {latent_dim} observations')
    sys.exit()


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
    # print(f"Number of splits: {len(splits)}")
    return splits


splits = split_data_by_class(z_correct, y_correct)


def calculate_gaussian_models(splits):
    """
    Returns a multivariate Gaussian model (mu and sigma) for a given class split
    """    
    gau = []

    for split in splits:
        mu = np.mean(split, axis=0)
        sigma = np.cov(split, rowvar=False)
        gau.append([mu, sigma])

     # Save Gaussian models
    gau_path = os.path.join(MODELS_PATH, f'{dataset}/{dataset}_GaussianModels.pkl')
    with open(gau_path, 'wb') as f:
        pickle.dump(gau, f)

    return gau

gau_models = calculate_gaussian_models(splits)
print('Gaussian models are computed for each class')
















