# CDAC-COVID-19-Medical-Image-Analysis
COVID-19 Medical image analysis

**Note: All the models in this repo are currently at a research stage and not yet intended as production-ready models (not meant for direct clinical diagnosis), and we are working continuously to improve them as new data becomes available. Please do not use COVID-Net for self-diagnosis and seek help from your local health authorities.**

The COVID-19 pandemic continues to have a devastating effect on the health and well-being of the global population.  A critical step in the fight against COVID-19 is effective screening of infected patients, with one of the key screening approaches being radiological imaging using chest radiography.  It was found in early studies that patients present abnormalities in chest radiography images that are characteristic of those infected with COVID-19.  Motivated by this, a number of artificial intelligence (AI) systems based on deep learning have been proposed and results have been shown to be quite promising in terms of accuracy in detecting patients infected with COVID-19 using chest radiography images. But usually these models claim high accuracy of prediction sidelining the most important factor-False Negatives. False Negatives can greatly affect the pandemic in long run. With our network we introduce some unique loss functions and training strategy to minimize false negatives at the same time giving high accuracy.


## Requirements

The main requirements are listed below:

* Tested with Pytorch
* OpenCV 4.2.0
* Python 3.7
* Numpy
* Pandas
* Matplotlib


### Dataset
We have used the open source dataset from https://github.com/UCSD-AI4H/COVID-CT/issues it is currently the largest COVID19 dataset available. We hope to add more dataset in future.

### Data distribution

Chest radiography images distribution
|  Type | Normal | COVID-19 | Total |
|:-----:|:------:|:--------:|:-----:|
| train |  xxxx  |   xxx    | xxxxx |
|  test |   xxx  |    xx    |   xxx |


## Training and Evaluation
The network takes as input an image of shape (N, 256, 256, 3) and outputs the softmax probabilities as (N, 1), where N is the number of batches. We have used blah blah


## Results
These are the final results for 
