# CDAC-COVID-19-Medical-Image-Analysis
## COVID-19
The coronavirus outbreak came to light on December 31, 2019 when China informed the World Health Organisation of a cluster of cases of pneumonia of an unknown cause in Wuhan City in Hubei Province. Among the tens of thousands of cases detected, several cases of COVID-19 are asymptomatic. These most common symptoms of the virus are fever and a dry cough. Some people may also experience aches, headache, tightness or shortness of breath. 

![alt text](https://github.com/AryanRaj315/CDAC-COVID-19-Medical-Image-Analysis/blob/master/Images/Proactive%20approach.png)
Our aim in this project is to demonstrate the use of Artificial Intelligence algorithms to detect the disease using automatic X-ray analysis to support radiologists in turn reducing the diagnosis time significantly.


## AI for COVID-19 diagnostics

The COVID-19 pandemic continues to have a devastating effect on the health and well-being of the global population.  A critical step in the fight against COVID-19 is effective screening of infected patients, with one of the key screening approaches being radiological imaging using chest radiography.  It was found in early studies that patients present abnormalities in chest radiography images that are characteristic of those infected with COVID-19.  Motivated by this, a number of artificial intelligence (AI) systems based on deep learning have been proposed and results have been shown to be quite promising in terms of accuracy in detecting patients infected with COVID-19 using chest radiography images. But usually these models claim high accuracy of prediction sidelining the most important factor-False Negatives. False Negatives can greatly affect the pandemic in long run. With our network we introduce some unique loss functions and training strategy to minimize false negatives at the same time giving high accuracy.

**Note: All the models in this repo are currently at a research stage and not yet intended as production-ready models (not meant for direct clinical diagnosis), and we are working continuously to improve them as new data becomes available. Please do not use these for self-diagnosis and seek help from your local health authorities.**


## Requirements

The main requirements are listed below:

* Tested with Pytorch
* OpenCV 4.2.0
* Python 3.7
* Numpy
* Pandas
* Matplotlib


### Dataset
We have used the open source dataset from https://github.com/UCSD-AI4H/COVID-CT/issues it is currently the largest COVID19 dataset available. We hope to add more dataset in the future.

### Data distribution

Chest radiography images distribution
|  Type | Normal | COVID-19 | Total |
|:-----:|:------:|:--------:|:-----:|
| Train |   156  |    279   |  435  |
|  Val  |    39  |     70   |  109  |

![alt text](https://github.com/AryanRaj315/CDAC-COVID-19-Medical-Image-Analysis/blob/master/Images/train-covid.png) ![alt text](https://github.com/AryanRaj315/CDAC-COVID-19-Medical-Image-Analysis/blob/master/Images/val-covid.png)


## Training and Evaluation
Our training strategy was completely focused on reducing the false negatives while maitaining accuracy and sensitivity. We employed unique loss function to achieve the same. A combination of high weighted FBeta loss along with BCE is employed. The network takes as input an image of shape (N, 256, 256, 3) and outputs the softmax probabilities as (N, 1), where N is the number of batches. The dataset is split maintaing the ratio. Heavy augmentation and multi stage training achieved some good and robust results.

## Results
![alt text](https://github.com/AryanRaj315/CDAC-COVID-19-Medical-Image-Analysis/blob/master/Images/loss.png) ![alt text](https://github.com/AryanRaj315/CDAC-COVID-19-Medical-Image-Analysis/blob/master/Images/accuracy.png)


## Frontend for Mobile devices
#### Homepage shows latest updates related to Covid-19 along with necessary navigation tools
![home](https://github.com/AryanRaj315/CDAC-COVID-19-Medical-Image-Analysis/blob/master/Images/c_homepage.jpg | width=150)
#### Diagnostic system to predict results
![model](https://github.com/AryanRaj315/CDAC-COVID-19-Medical-Image-Analysis/blob/master/Images/c_diagnostic.jpg | width=150) ![charts](https://github.com/AryanRaj315/CDAC-COVID-19-Medical-Image-Analysis/blob/master/Images/c_charts.jpg | width=150)
#### Total infections district wise
![cases]("https://github.com/AryanRaj315/CDAC-COVID-19-Medical-Image-Analysis/blob/master/Images/c_cases.jpg | width=150)
