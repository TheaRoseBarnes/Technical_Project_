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


Sure! Here's the "Usage" section for your GitHub README, explaining the process of patient subtyping using vital measurements:

Usage

1.Feature_transformation.py
The Feature_transformation.py script utilizes the vitals_groupby_30min table from the SQL tables file to preprocess the raw vital measurement data. This table contains vital measurement features aggregated over 30-minute intervals for each patient stay. The script applies various transformations depending on the distribution of each feature. The output of this script is scaled_df, which contains the transformed data ready for dynamic time warping.
2.DTW.py
The DTW.py script takes scaled_df as input, representing the preprocessed vital measurement data for patients. It performs either independent or dependent dynamic time warping (DTW) on the data, considering the unique temporal patterns of patients' vital measurement trajectories. The result is the average_dtw_matrix, an nxn distance matrix, where n is the number of unique patient stays. Each value in the matrix represents the average optimal DTW alignment distance between the vital measurement trajectories of n patients.
3.Clustering.py
The Clustering.py script receives the average_dtw_matrix as input. It performs dimensionality reduction and clustering on the distance matrix to identify patient subtypes based on similar vital measurement patterns. The script implements UMAP dimensionality reduction and HDBSCAN clustering and internal cluster validation methods to ensure robust and meaningful clustering results.

This entire process forms the basis of patient subtyping using vital measurements. By leveraging dynamic time warping and clustering techniques, it enables the identification of distinct patient groups with similar vital measurement trajectories. These patient subtypes can provide valuable insights for personalized healthcare and targeted treatment strategies.



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


