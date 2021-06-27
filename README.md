# OS-ROCKET: Open Set Recognition for Time Series Classification
Implementation of the OS-ROCKET algorithm for open set recognition for time series classifciation.

This repository contains the first generic open set model for time series classification, OS-ROCKET, that is applicable to multiple datasets, and can work with any classifier. In this case, the ROCKET[[1]](#1) is used as the classifier. Class-specific time series barycenters are used to achieve unknown detection, by looking at the Euclidean DTW (dynamic time warping) distance and the cross-correlation between the barycenters and an input. Experimental results indicate that the proposed method achieves near-perfect unknown detection results (with a recall of 0.93) by trading off some of its closed set classification accuracy (around 0.12 less) on the UEA multivariate time series archive with 30 datasets. 

![alt text](https://github.com/tolgaakar/OS-ROCKET-Open-Set-Recognition-for-Time-Series-Classification/blob/main/image.jpg?raw=true)

## References
<a id="1">[1]</a> 
Dempster,   A.,   Petitjean,   F.,   and  Webb,   G.  I. (2020).   Rocket:   exceptionally  fast  and  accurate  time  series  classification  using  random  convolutional  kernels. Data  Mining  and  Knowledge Discovery, 34(5):1454–1495.
