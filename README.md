# CT_3D_Classification
## Motivation and Related Work
A computerized tomography (CT) scan is the most widespread imaging examination worldwide
for diagnosis and screening many disorders, including adenomas, neoplasma and hemorrhages Hara
et al. Automated CT image interpretation could provide many benefits in medical settings,
such as workflow prioritization and clinical decision support, especially for time-sensitive cerebral
hemorrhages where immediate diagnosis is crucial. Recently, rapid CTC
detection of colorectal neoplasia is being assessed as a population screening tool. Here, our aim is
to develop a deep learning algorithm that detects colorectal neoplasia using CT images.
<br>

## Goals
Classify olyps by size cased by brain hemorrhage/colorectal neoplasms, from CT scans, to distinguish benign and malignant types.
<br>
Originally, our approach was to convert the image volumes into numpy arrays. However, even the
most popular 3D Convolutional Neural Networks that use this approach found difficulties. Con-
verting into the same dimensions has been limited to only 20 slices for
hundreds of patients for a decent accuracy of 70 percent. We propose that our model will somehow
do better.
<br>
## Dataset
The dataset was obtained from the **Cancer Imaging Archive CT Colonography dataset**, which can be found and downloaded here: https://wiki.cancerimagingarchive.net/display/Public/CT+COLONOGRAPHY#bc030a1baaff4fc28093435d2a56b883
<br>
It was ready-to-use and contained prone and supine DICOM images from same-day validated
images- 243 negative cases, 69 cases with 6 to 9 mm polyps, and 35 cases which have at least one > 10 mm polyp and their histological type(836 patients). There are 3,451 series and 941,771 image
from this end data. Separate files were available for download containing the labels for each patient data. 
<br>
Finally due to class imbalance was huge, we took only **Supine** data from each. Samples which did not have the Supine identication were discarded.
<br>
## Preprocessing
1. Set up a labeler to extract the mentions from radioloy reports. Then make classification
to positive or negative interpreting from the mentions. For the TCIA CT colongraphy dataset, the
DICOM files are labeled into 2 categories based on the polyp of presence(0 for no polyp and 1 for
polyps exist)
<br>
3. The DICOM images were inputted and concatenate as 3D data in order to create input into 3D CNN models
<br>
4. Data transformation: The initial resolution was 512 x 512 which was scaled down to 112 x 112 pixel resolution
<br>
5. Data enhancement: Scaling, Cubic interpolation to introduce some noise
<br>

## Transfer Learning
We plan to use a **3D Convolutional Neural Network (3D CNN) architecture**. **We used a pre-trained
weights from the ResNet-18 model**. Due to the similarity in domain and size of 3D DICOM volumes
and action recognition from video data. We hypothesized that the pre-trained weights are sufficient
enough to accurately predict the presence of polyps in our data.
<br>
We utilized a log-softmax function with the criterion as Negative Log Likelihood Loss.
<br>

## Optimization
We observed the performance and generalization errors of our model, and optimize to resolve potential
issues of under/overfitting.
<br>
We used SGD with momentum to helps accelerate gradients vectors in the right directions, thus leading to faster converging
<br>
The training parameters include a weight decay of 0.001 and 0.9 for momentum. And we chose 100 epochs and batch size of 2.
<br>
We run it on NYU Prince HPC, using totally 8 hours for training.
## Evaluation Metrics
We propose that the model will successfully classify negative/positive cases of polyps. After reporting
AUCs, accuracy, and precision, we plan to improve the performance of the model by optimizing the
hyper-parameters and through regularization.

