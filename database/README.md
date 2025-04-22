# /database Folder Structure

The `/database` folder contains the raw datasets used for the DataFusion-App-Python project. It is divided into two subfolders: **Adult** and **Chronic**. Each subfolder contains the respective dataset and its associated metadata.

## Adult

This dataset is used to predict whether an individual's annual income exceeds $50K per year based on census data. It is also known as the "Census Income" dataset and is typically used for classification tasks.

### Dataset Characteristics

- **Type:** Multivariate
- **Subject Area:** Social Science
- **Associated Task:** Classification
- **Feature Types:** Categorical, Integer

### Dataset Details

- **Instances:** 48,842
- **Features:** 14

### Additional Information

The dataset was extracted by Barry Becker from the 1994 Census database. The extraction process applied the following conditions to obtain a set of reasonably clean records:

- **AAGE > 16**
- **AGI > 100**
- **AFNLWGT > 1**
- **HRSWK > 0**

The primary prediction task is to determine whether a person's income exceeds $50,000 per year based on various demographic and employment factors. The dataset includes both numerical and categorical features, making it an excellent choice for classification models and data preprocessing techniques such as encoding and feature scaling.

### Features:

- **Numerical Features:**
  - `age`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`
- **Categorical Features:**
  - `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`
- **Target Variable:**
  - `income` (binary: >50K or <=50K)

### Missing Values

- **Present:** Yes  
  Some rows in the dataset contain missing values, typically represented as "?" or similar placeholders. These need to be handled during the data cleaning process (e.g., removal or imputation).

For further details, visit the [UCI Adult Dataset page](https://archive.ics.uci.edu/dataset/2/adult).

## Chronic

The Chronic Kidney Disease dataset is used for diagnosing whether a patient has chronic kidney disease (CKD). It is primarily a classification dataset where the goal is to predict if a patient has CKD based on various medical parameters.

### Dataset Characteristics

- **Type:** Multivariate
- **Subject Area:** Medical Science
- **Associated Task:** Classification
- **Feature Types:** Categorical, Integer, Float

### Dataset Details

- **Instances:** 400
- **Features:** 24

### Additional Information

The dataset contains medical data collected from patients with various health indicators. It aims to predict whether a person suffers from chronic kidney disease (CKD) based on multiple health attributes. This dataset includes continuous variables representing medical conditions and categorical variables for specific diagnosis indicators.

### Features:

- **Numerical Features:**
  - `age`, `blood-pressure`, `specific-gravity`, `albumin`, `sugar`, `red-blood-cells`, `pus-cell`, `polynuclear`, `pedal-edema`, `anaemia`
- **Categorical Features:**
  - `class` (binary: `ckd` or `notckd`), indicating whether the patient has chronic kidney disease or not.

### Missing Values

- **Present:** Yes  
  This dataset also contains missing values in some attributes, which need to be addressed through imputation or exclusion during data preprocessing.

For further details, visit the [UCI Chronic Kidney Disease Dataset page](https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease).

---

Each dataset is stored in its respective folder (`Adult` or `Chronic`), and preprocessing operations (such as handling missing values, encoding categorical data, etc.) can be performed directly on these files as part of the ETL pipeline in the DataFusion-App-Python application.
