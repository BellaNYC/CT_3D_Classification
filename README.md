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

## Dataset
The dataset was obtained from the **Cancer Imaging Archive CT Colonography dataset** (https://wiki.cancerimagingarchive.net/display/Public/CT+COLONOGRAPHY#bc030a1baaff4fc28093435d2a56b883).
It was ready-to-use and contained prone and supine DICOM images from same-day validated
images- 243 negative cases, 69 cases with 6 to 9 mm polyps, and 35 cases which have at least one
> 10 mm polyp and their histological type(836 patients). There are 3,451 series and 941,771 image
from this end data. Samples which did not have the Supine identication were discarded, resulting
in negative cases, z cases with 6 to 9 mm polyps and AAAAAAAA cases with at least 1 polyp>10mm
in size. Separate les were available for download containing the labels for each patient data. The
initial resolution was 512 x 512 which was scaled down to 112 x 112 pixel resolution.The images
were appended in order to create input into 3D CNN models.













## References:
https://wiki.cancerimagingarchive.net/display/Public/CT+COLONOGRAPHY#bc030a1baaff4fc28093435d2a56b883
<br>
