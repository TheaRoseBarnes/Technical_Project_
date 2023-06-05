# Technical_Project_
Pipeline for clustering and predicting patient outcomes

# Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [License](#license)


## Introduction<a name="introduction"></a>

This project addresses the limitations of existing Early Warning Score (EWS) systems in the ICU, leveraging the data-rich environment and the MIMIC database. The goal is to create a personalized patient risk prediction system by clustering ICU patients based on the similarities in the temporal progression of their vital measurements.

Unsupervised machine learning techniques are utislised, specifically Dynamic Time Warping (DTW) distance, to quantify the similarity between vital trajectories and form meaningful patient clusters. These clusters serve as the basis for patient characterization and risk group assignment, enabling personalized predictions.

Supervised machine learning methods are employed to produce explainable predictions and forecasting of patient deterioration, focusing on key outcomes such as in-hospital mortality and cardiac arrest.

To strike a balance between model complexity and interpretability, Explainable Boosting Machines (EBMs) are employed for transparent yet high-performing predictions.

Through this project, the aim is to advance patient risk prediction in the ICU, ultimately improving patient outcomes and healthcare decision-making.

## Installation<a name="installation"></a>

Steps for installation go here.

## Usage<a name="usage"></a>

DTW: performs independent or depedent dynamic time warping on mutlidimensional time series data,
producing a final distance matrix that is used for clustering.

Clustering: implements UMAP dimensionality reduction and HDBSCAN clustering to cluster a distance matrix.
Three cluster validity indexes are implemented to internally validate the results.

Explainable Boosting Machines: implements explainble boosting machines prediction models.


## Data<a name="data"></a>

The data is not uploaded since access to the MIMIC-IV 2.0 data is restricted for credentialed users for which CITI Data or Specimens Only Research is required and a data use agreement is signed. The researcher must complete a course in protecting human research participants. Researchers can apply for access to the data by following the instructions on this page: https://physionet.org/content/mimiciv/2.0/#files-panel.

The tables used from MIMIC-IV 2.0 are as follows:

From hosp module:
- patients 
- diagnoses_icd

From icu module:
- chartevents
- d_items
- 

## License<a name="license"></a>

Please note that the MIMIC database citation mentioned above is for reference purposes only and should be appropriately cited according to the database's guidelines.


