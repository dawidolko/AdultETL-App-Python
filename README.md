# DataTransmuter-App-Python

DataTransmuter-App-Python is a versatile, Python-based GUI application designed for data warehousing projects. Developed as part of an academic course, this project bridges the gap between theory and practice by showcasing a full-featured ETL (Extract, Transform, Load) pipeline in a user-friendly environment. Whether you’re a student or a professional, this tool offers an educational yet practical insight into modern data processing techniques.

## Overview

At its core, DataTransmuter-App-Python is all about transforming raw data into insightful information. The application lets you extract data from various sources, perform essential transformations like normalization and standardization, and then load the processed data into your data warehouse. All these operations are accessible via an intuitive GUI built with popular Python libraries such as PySimpleGUI.

## Features

- **Interactive GUI:** A clean and intuitive interface that makes it easy to run complex ETL processes without diving deep into code.
- **Comprehensive ETL Pipeline:** 
  - **Extraction:** Seamlessly import data from different sources.
  - **Transformation:** Apply normalization, standardization, and custom processing techniques.
  - **Loading:** Efficiently load the processed data into your data warehouse.
- **Modular Architecture:** Each functionality is encapsulated in its own module, making it simple to maintain and extend.
- **Educational Focus:** Designed as an academic project, this tool not only demonstrates the practical aspects of data warehousing but also serves as a foundation for further exploration.
- **Extensibility:** The codebase is built to be flexible, allowing you to plug in additional features or optimizations as needed.

## Project Structure

``` 
DataTransmuter-App-Python/
├── README.md             # Project overview and instructions
├── LICENSE               # Licensing information
├── docs/
```

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/DataTransmuter-App-Python.git
   cd DataTransmuter-App-Python
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
The GUI will launch, allowing you to select data sources, execute ETL operations, and visualize your data processing pipeline with ease.

## Contributing

Contributions, bug reports, and suggestions are welcome!  
If you’d like to improve this project or add new features, please open an issue or submit a pull request. Let’s work together to keep the code as clean as our data should be.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, feel free to open an issue on GitHub or reach out via email at [dawid_olko@outlook.com].

---

Happy coding, and may your data always be clean!
