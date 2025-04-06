# AdultETL-App-Python

AdultETL-App-Python is a versatile, Python-based GUI application designed for data warehousing projects. Developed as part of an academic course on data warehousing, this project bridges theory and practice by showcasing a comprehensive ETL (Extract, Transform, Load) pipeline integrated with advanced data processing and machine learning techniques. The tool is built to process and analyze real-world data from the UCI Adult (Census Income) dataset, making it a robust educational platform for both students and professionals.

## Overview

At its core, AdultETL-App-Python transforms raw data into actionable insights. The application provides functionality to:

- **Extract** data from CSV files – including the UCI Adult dataset.
- **Transform** data by applying a wide range of processing techniques such as data cleaning, normalization, standardization, and custom feature engineering.
- **Load** the processed data into a data warehouse environment for further analysis.

This tool addresses key aspects of data processing:

- Statistical analysis (computing min, max, median, standard deviation, mode),
- Correlation analysis among numerical features,
- Extraction of data subsets,
- Handling missing values and removal of irrelevant columns,
- Visualization of data distributions and relationships through basic plots,
- Implementation of machine learning tasks including classification and clustering.

All operations are accessible through an intuitive GUI built with popular Python libraries (e.g., PySimpleGUI), ensuring ease of use even for those with limited coding experience.

## Dataset: UCI Adult (Census Income) Dataset

The UCI Adult dataset is an ideal choice for this project due to its rich mix of continuous and categorical data:

- **Numerical Features:**

  - `age`, `education-num`, `capital-gain`, `capital-loss`, `hours-per-week`  
    These allow you to compute statistical measures and apply scaling/standardization techniques.

- **Categorical Features:**

  - `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, `native-country`  
    These enable various data transformations such as encoding, cleaning (e.g., replacing “?” values), and visualization via bar charts.

- **Target Variable:**
  - `income` (binary: >50K or <=50K)  
    This serves as the basis for classification tasks and provides a clear problem statement for implementing machine learning algorithms.

The dataset poses realistic challenges, including missing values and varying scales, which make it perfect for demonstrating robust ETL processes and advanced data processing operations.

## Features

- **Interactive GUI:**  
  A clean and intuitive interface that allows you to perform complex ETL processes without diving deep into code.

- **Comprehensive ETL Pipeline:**

  - **Extraction:** Seamlessly import CSV data, including the UCI Adult dataset.
  - **Transformation:** Execute data cleaning, normalization, standardization, and custom processing techniques.
  - **Loading:** Efficiently integrate the processed data into your data warehouse.

- **Statistical Analysis & Data Processing:**

  - Compute key statistical metrics (min, max, median, standard deviation, mode).
  - Analyze correlations between continuous features.
  - Extract specific subsets of data based on user-defined criteria.
  - Handle missing data through removal or imputation.
  - Add new features via feature engineering.

- **Machine Learning Capabilities:**

  - Implement classification (e.g., decision trees, k-NN) to predict income levels.
  - Apply clustering techniques (e.g., k-means) to uncover hidden patterns.
  - Explore association rules to discover relationships between different attributes.

- **Modular Architecture:**  
  Each functionality is encapsulated in its own module, ensuring that the code is maintainable and extensible.

- **Educational Focus:**  
  This project not only demonstrates the practical aspects of data warehousing and ETL processes but also serves as a foundation for further exploration in data analysis and machine learning.

## Project Structure

```

AdultETL-App-Python/
├── README.md # Project overview and instructions
├── LICENSE # Licensing information
├── requirements.txt # List of project dependencies
├── docs/
│ └── description.docx # Detailed project description
├── src/
│ └── main.py # Entry point for the GUI application

```

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/AdultETL-App-Python.git
   cd AdultETL-App-Python
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

The GUI will launch, allowing you to:

- Load the UCI Adult CSV dataset.
- Execute various ETL operations including data cleaning, transformation, and loading.
- Visualize statistical summaries, correlations, and outputs from machine learning models.

## Contributing

Contributions, bug reports, and suggestions are welcome!
If you’d like to improve this project or add new features, please open an issue or submit a pull request. Let’s work together to keep the code as clean as our data should be.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, feel free to open an issue on GitHub or reach out via email at [dawid_olko@outlook.com].
