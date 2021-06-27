# OS-ROCKET: Open Set Recognition for Time Series Classification

This repository contains the first generic open set model for time series classification, OS-ROCKET, developed by me as part of my master thesis for the University of Hildesheim. OS-ROCKET is applicable to multiple datasets, and can work with any classifier. In this case, the ROCKET[[1]](#1) is used as the classifier. Class-specific time series barycenters are used to achieve unknown detection, by looking at the Euclidean DTW (dynamic time warping) distance and the cross-correlation between the barycenters and an input. Experimental results indicate that the proposed method achieves near-perfect unknown detection results (with a recall of 0.93) by trading off some of its closed set classification accuracy (around 0.12 less) on the UEA multivariate time series archive with 30 datasets. 

![OS ROCKET Diagram](https://github.com/tolgaakar/OS-ROCKET-Open-Set-Recognition-for-Time-Series-Classification/blob/main/OSRocketDiagram.png?raw=true)

### How to run
Prerequisites: Install the sktime and tslearn libraries and download the UEA archive
```
$pip install sktime
$pip install numba
$pip install tslearn
```

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

## References
<a id="1">[1]</a> 
Dempster,   A.,   Petitjean,   F.,   and  Webb,   G.  I. (2020).   Rocket:   exceptionally  fast  and  accurate  time  series  classification  using  random  convolutional  kernels. Data  Mining  and  Knowledge Discovery, 34(5):1454–1495.
