# DataFusion-App-Python

DataFusion-App-Python is a comprehensive, Python-based GUI application developed for data analysis, processing, and machine learning tasks. The project integrates powerful data manipulation techniques with a user-friendly interface, making it an excellent tool for both educational and professional purposes. This application is designed to process and analyze data from two distinct datasets: the UCI Adult dataset and the UCI Chronic Kidney Disease dataset, enabling users to explore and uncover insights across a variety of domains.

## Overview

At its core, DataFusion-App-Python provides a unified platform for working with real-world datasets, offering a variety of data processing, statistical analysis, and machine learning features. The application allows users to:

- **Extract** data from CSV files, including the UCI Adult and Chronic Kidney Disease datasets.
- **Transform** the data by performing various data cleaning, normalization, and feature engineering tasks.
- **Analyze** the data through statistical summaries, correlation analysis, and visualizations.
- **Apply** machine learning techniques such as classification, clustering, and association rule mining.

The user-friendly GUI, built using PySimpleGUI, ensures that even users with limited programming experience can perform complex data operations effortlessly.

## Datasets

### UCI Adult (Census Income) Dataset

The UCI Adult dataset contains demographic and economic information, including both numerical and categorical features. This dataset is ideal for demonstrating data preprocessing techniques and machine learning applications, particularly classification tasks.

- **Numerical Features:**

  - `age`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`
    These numerical attributes allow users to compute key statistical measures, apply scaling techniques, and analyze distributions.

- **Categorical Features:**

  - `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`
    These categorical attributes offer opportunities for encoding and visualization, providing insights into the relationships between different demographic factors.

- **Target Variable:**
  - `income` (binary: >50K or <=50K)
    This serves as the target for classification tasks, offering a clear challenge for predicting income levels based on demographic and employment factors.

### UCI Chronic Kidney Disease Dataset

The UCI Chronic Kidney Disease dataset contains medical data for diagnosing chronic kidney disease (CKD). This dataset presents a mix of continuous and categorical data related to various health parameters, making it suitable for both data analysis and machine learning tasks.

- **Numerical Features:**

  - `age`, `blood-pressure`, `specific-gravity`, `albumin`, `sugar`, `red-blood-cells`, `pus-cell`, `polynuclear`, `pedal-edema`, `anaemia`, `class`
    These features are crucial for analyzing medical conditions, allowing users to perform statistical analysis, normalization, and transformation.

- **Categorical Features:**
  - `class` (categorical: 'ckd' or 'notckd')
    This is the target variable for classification tasks, allowing for the prediction of kidney disease based on medical indicators.

## Features

- **Interactive GUI:**
  A clean, intuitive interface built with PySimpleGUI that allows users to perform complex data operations without needing to write any code. The GUI offers seamless access to all functionalities and makes it easy to visualize results.

- **Comprehensive Data Processing Pipeline:**

  - **Extraction:** Effortlessly load data from CSV files, including both the UCI Adult and Chronic Kidney Disease datasets.
  - **Transformation:** Clean and transform data by applying a variety of techniques such as data imputation, feature scaling, encoding categorical variables, and extracting subsets based on user-defined criteria.
  - **Loading:** Prepare the transformed data for analysis or save it to new files for further use.

- **Statistical Analysis & Data Processing:**

  - Compute essential statistical metrics (min, max, median, standard deviation, mode) for both numerical and categorical features.
  - Perform correlation analysis between numerical features using Pearson, Kendall, and Spearman methods.
  - Extract specific data subsets and handle missing values through user-defined strategies (e.g., removing rows/columns or replacing with mean/median values).
  - Visualize data distributions, relationships, and correlations using various chart types such as histograms, scatter plots, and heatmaps.

- **Machine Learning Capabilities:**

  - **Classification:** Implement machine learning models (e.g., decision trees, k-NN, and logistic regression) to predict income levels from the Adult dataset or diagnose chronic kidney disease from the medical dataset.
  - **Clustering:** Use clustering techniques like k-means to identify patterns or group similar data points.
  - **Association Rules:** Apply association rule mining (e.g., Apriori) to discover hidden relationships between variables in the datasets.

- **Modular Architecture:**
  The application is designed with a modular architecture, where each feature is encapsulated in its own module. This ensures maintainability and extensibility of the codebase, allowing for easy updates and future improvements.

- **Educational Focus:**
  DataFusion-App-Python serves as an educational tool for exploring various data processing and machine learning techniques. It provides an interactive environment to learn about data wrangling, statistical analysis, and machine learning while working with real-world datasets.

## Project Structure

```
DataFusion-App-Python/
├── README.md                # Project overview and instructions
├── LICENSE                  # Licensing information
├── database/
│   └── adult
│   └── chronic
├── docs/
│   └── description.docx     # Detailed project description
├── src/
│   └── requirements.txt     # List of project dependencies
│   └── main.py              # Entry point for the GUI application
```

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/dawidolko/DataFusion-App-Python.git
   cd DataFusion-App-Python
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Start the application by running:

```bash
python src/main.py
```

The GUI will launch, allowing you to:

- Load the UCI Adult or Chronic Kidney Disease dataset.
- Perform data processing tasks such as cleaning, transformation, and normalization.
- Visualize statistical summaries and relationships between variables.
- Implement machine learning models for classification, clustering, and association rule mining.

## Contributing

Contributions, bug reports, and suggestions are welcome! If you’d like to improve this project or add new features, please open an issue or submit a pull request. Let’s work together to keep the code as clean and efficient as possible.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, feel free to open an issue on GitHub or reach out via email at [dawidolko@outlook.com].
