# OS-ROCKET: Open Set Recognition for Time Series Classification

This repository contains the first generic open set model for time series classification developed by me as part of my master thesis for the University of Hildesheim. The proposed method is applicable to multiple datasets, and can work with any classifier. In this case, the ROCKET[[1]](#1) and InceptionTime[[2]](#2) are used as the classifiers. Class-specific time series barycenters are used to achieve unknown detection, by looking at the Euclidean DTW (dynamic time warping) distance and the cross-correlation between the barycenters and an input. Experimental results indicate that the proposed method achieves near-perfect unknown detection results (with a recall of over 90%) by trading off some of its closed set classification accuracy on the UEA multivariate time series archive with 30 datasets. 

You can find the details of this work in the [OSCforTSC_Paper.pdf](https://github.com/tolgaakar/Open-Set-Recognition-for-Time-Series-Classification/blob/main/OSRforTSC_Paper.pdf) file. 


Prerequisites: Install the sktime and tslearn libraries and download the UEA archive
```
$pip install tensorflow==2.7.0
$pip install sktime==0.5.3
$pip install numba==0.51.2
$pip install tslearn==0.5.2
```
PS: For the LCVAE ipmlementation tensorflow==2.4.1 is required

```
$wget http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_ts.zip
$unzip Multivariate2018_ts.zip
```

Step 1: Compute the barycenters and augment the train data to use as the known unknowns for grid search
```
$python barycenters_and_augmentation.py ArticularyWordRecognition
```


Step 2: Run OpenSetROCKET_GridSearch.py to train the classifier, as well as to get the optimal threshold values for the unknown detector for the given dataset
```
$python OpenSetROCKET_GridSearch.py ArticularyWordRecognition
```


Step 3: Test the OS-ROCKET model with the given thresholds and unknown datasets as arguments
```
$python OpenSetROCKET_Test.py ArticularyWordRecognition 2.75 3.25 PEMS-SF SpokenArabicDigits
```

## Overall Performance

| Method  | Closed Set Clasification Accuracy | Recall for the Unknowns | Open Set Macro F1 Score |
| ------------- | ------------- | ------------- | ------------- |
| OvA-CNNs | **0.627** | 0.247 | 0.500 |
| OpenMax InceptionTime | 0.440 | 0.832 | 0.515 |
| LCVAE | 0.481  | 0.663| 0.463 |
| Open Set ROCKET | 0.594 | **0.901** | **0.640** |
| Open Set InceptionTime  | 0.590 | **0.901** | **0.660** |


## References
<a id="1">[1]</a> 
Dempster,   A.,   Petitjean,   F.,   and  Webb,   G.  I. (2020).   Rocket:   exceptionally  fast  and  accurate  time  series  classification  using  random  convolutional  kernels. Data  Mining  and  Knowledge Discovery, 34(5):1454–1495.

<a id="2">[2]</a> 
Fawaz, H.I., Lucas, B., Forestier, G., Pelletier, C., Schmidt, D.F., Weber, J., Webb,G.I., Idoumghar, L., Muller, P.A., Petitjean, F.: Inceptiontime: Finding alexnet fortime series classification. Data Mining and Knowledge Discovery34(6), 1936–1962(2020)
