import PySimpleGUI as sg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from scipy.stats import kurtosis, skew

# ------------------ Global GUI Settings ------------------ #

sg.set_options(font=("Helvetica", 12))
df = None
figure_canvas_agg = None

# ------------------ Helper Functions ------------------ #

def load_dataset(file_name):
    """
    Loads the selected dataset from a file.
    """
    if file_name == "Adult Dataset":
        col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                     'marital-status', 'occupation', 'relationship', 'race',
                     'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                     'native-country', 'income']
        try:
            df = pd.read_csv('../database/adult/adult.data', header=None, names=col_names, skipinitialspace=True)
        except Exception as e:
            df = pd.read_csv('adult.data', header=None, names=col_names, skipinitialspace=True)

        df.replace('?', np.nan, inplace=True)

        print("Columns and Data Types before conversion (Adult Dataset):")
        print(df.dtypes)

        numeric_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
        for col in numeric_cols:
            print(f"Converting {col} to numeric...")
            df[col] = pd.to_numeric(df[col], errors='coerce')

        print("Columns and Data Types after conversion (Adult Dataset):")
        print(df.dtypes)

        categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                            'race', 'sex', 'native-country', 'income']
        for col in categorical_cols:
            df[col] = df[col].astype('category')

    elif file_name == "Kidney Disease Dataset":
        try:
            df = pd.read_csv('../database/chronic/kidney_disease.csv')
        except Exception as e:
            raise ValueError(f"Error loading kidney disease dataset: {e}")

        df.replace('?', np.nan, inplace=True)

        categorical_cols = ['fbs', 'restecg', 'exang', 'slope', 'thal', 'classification']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')

    return df


def compute_statistics(df, dataset_type):
    """
    Computes statistics for both numeric and categorical columns:
    - For numeric columns: min, max, mean, median, std, mode, variance, skewness, kurtosis.
    - For categorical columns: count, mode.
    """
    stats_data = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"Detected numeric columns: {numeric_cols}")

    for col in numeric_cols:
        col_data = df[col].dropna()
        print(f"Computing statistics for {col}...")

        if len(col_data) == 0:
            stats_data.append([col, None, None, None, None, None, None, None, None, None])
        else:
            mode_val = col_data.mode()
            mode_val = mode_val.iloc[0] if not mode_val.empty else None

            stats_data.append([col,
                               col_data.min(),
                               col_data.max(),
                               round(col_data.mean(), 2),
                               round(col_data.median(), 2),
                               round(col_data.std(), 2),
                               mode_val,
                               round(col_data.var(), 2),  # Variance
                               round(skew(col_data), 2),  # Skewness
                               round(kurtosis(col_data), 2)  # Kurtosis
                               ])

    if dataset_type == "Adult Dataset":
        categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                            'race', 'sex', 'native-country', 'income']
        for col in categorical_cols:
            if col in df.columns:
                col_data = df[col].dropna()
                print(f"Computing statistics for categorical column {col}...")
                if len(col_data) == 0:
                    stats_data.append([col, None, None, None, None, None, None])
                else:
                    stats_data.append([col,
                                       col_data.value_counts().to_dict(),
                                       col_data.mode().iloc[0] if not col_data.mode().empty else None,
                                       ])
    else:
        categorical_cols = df.select_dtypes(include=['category', object]).columns
        for col in categorical_cols:
            col_data = df[col].dropna()
            print(f"Computing statistics for categorical column {col}...")
            if len(col_data) == 0:
                stats_data.append([col, None, None, None, None, None, None])
            else:
                stats_data.append([col,
                                   col_data.value_counts().to_dict(),
                                   col_data.mode().iloc[0] if not col_data.mode().empty else None,
                                   ])

    return stats_data

def compute_correlation(df):
    """
    Computes the correlation matrix for numeric columns using both Pearson and Spearman methods.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return None, None

    pearson_corr = df[numeric_cols].corr(method='pearson')

    spearman_corr = df[numeric_cols].corr(method='spearman')

    return pearson_corr, spearman_corr

import seaborn as sns
import matplotlib.pyplot as plt

def generate_plot(df, column, chart_type):
    """
    Generates a plot based on the selected column and chart type.
    - "Histogram": histogram for numeric or bar chart for categorical.
    - "Boxplot": for numeric columns.
    - "Bar Chart": for categorical columns (or binned for numeric).
    - "Line Plot": line plot for numeric columns.
    - "Pie Chart": pie chart for categorical data.
    - "Heatmap": heatmap for correlation matrix.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    if column not in df.columns:
        ax.text(0.5, 0.5, f'Column "{column}" not found.', ha='center', va='center')
        return fig

    if chart_type == "Histogram":
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column].dropna().plot(kind='hist', bins=20, ax=ax, color='skyblue', edgecolor='black')
            ax.set_title(f'Histogram of {column}')
            ax.set_xlabel(column)
            ax.set_ylabel('Frequency')
        else:
            df[column].dropna().value_counts().plot(kind='bar', ax=ax, color='orange', edgecolor='black')
            ax.set_title(f'Bar Chart of {column}')
            ax.set_xlabel(column)
            ax.set_ylabel('Count')

    elif chart_type == "Boxplot":
        if pd.api.types.is_numeric_dtype(df[column]):
            sns.boxplot(x=df[column].dropna(), ax=ax)
            ax.set_title(f'Boxplot of {column}')
        else:
            ax.text(0.5, 0.5, f'Boxplot not applicable for categorical column "{column}".', ha='center', va='center')

    elif chart_type == "Bar Chart":
        if not pd.api.types.is_numeric_dtype(df[column]):
            df[column].dropna().value_counts().plot(kind='bar', ax=ax, color='green', edgecolor='black')
            ax.set_title(f'Bar Chart of {column}')
            ax.set_xlabel(column)
            ax.set_ylabel('Count')
        else:
            bins = 10
            counts, bin_edges = np.histogram(df[column].dropna(), bins=bins)
            ax.bar(range(bins), counts, width=0.8, color='purple', edgecolor='black')
            ax.set_title(f'Bar Chart (Binned) of {column}')
            ax.set_xlabel(column)
            ax.set_ylabel('Frequency')

    elif chart_type == "Line Plot":
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column].dropna().plot(kind='line', ax=ax, color='blue')
            ax.set_title(f'Line Plot of {column}')
            ax.set_xlabel(column)
            ax.set_ylabel('Value')
        else:
            ax.text(0.5, 0.5, f'Line Plot not applicable for categorical column "{column}".', ha='center', va='center')

    elif chart_type == "Pie Chart":
        if not pd.api.types.is_numeric_dtype(df[column]):
            df[column].dropna().value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=sns.color_palette("Set3", len(df[column].dropna().unique())))
            ax.set_title(f'Pie Chart of {column}')
            ax.set_ylabel('')
        else:
            ax.text(0.5, 0.5, f'Pie Chart not applicable for numeric column "{column}".', ha='center', va='center')

    elif chart_type == "Heatmap":
        # Only generate heatmap for numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            ax.text(0.5, 0.5, f'Heatmap not possible. No numeric columns available.', ha='center', va='center')
        else:
            corr_matrix = numeric_df.corr()  # Calculate correlation matrix for numeric columns
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, linewidths=0.5)
            ax.set_title('Correlation Heatmap')

    plt.tight_layout()
    return fig

def draw_figure(canvas, figure):
    """
    Draws a matplotlib figure on the provided PySimpleGUI Canvas.
    """
    for child in canvas.winfo_children():
        child.destroy()
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def generate_plot(df, column, chart_type):
    """
    Generates a plot based on the selected column and chart type.
    - "Histogram": histogram for numeric or bar chart for categorical.
    - "Boxplot": for numeric columns.
    - "Bar Chart": for categorical columns (or binned for numeric).
    - "Line Plot": line plot for numeric columns.
    - "Pie Chart": pie chart for categorical data.
    - "Heatmap": heatmap for correlation matrix.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    if column not in df.columns:
        ax.text(0.5, 0.5, f'Column "{column}" not found.', ha='center', va='center')
        return fig

    if chart_type == "Histogram":
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column].dropna().plot(kind='hist', bins=20, ax=ax, color='skyblue', edgecolor='black')
            ax.set_title(f'Histogram of {column}')
            ax.set_xlabel(column)
            ax.set_ylabel('Frequency')
        else:
            df[column].dropna().value_counts().plot(kind='bar', ax=ax, color='orange', edgecolor='black')
            ax.set_title(f'Bar Chart of {column}')
            ax.set_xlabel(column)
            ax.set_ylabel('Count')

    elif chart_type == "Boxplot":
        if pd.api.types.is_numeric_dtype(df[column]):
            sns.boxplot(x=df[column].dropna(), ax=ax)
            ax.set_title(f'Boxplot of {column}')
        else:
            ax.text(0.5, 0.5, f'Boxplot not applicable for categorical column "{column}".', ha='center', va='center')

    elif chart_type == "Bar Chart":
        if not pd.api.types.is_numeric_dtype(df[column]):
            df[column].dropna().value_counts().plot(kind='bar', ax=ax, color='green', edgecolor='black')
            ax.set_title(f'Bar Chart of {column}')
            ax.set_xlabel(column)
            ax.set_ylabel('Count')
        else:
            bins = 10
            counts, bin_edges = np.histogram(df[column].dropna(), bins=bins)
            ax.bar(range(bins), counts, width=0.8, color='purple', edgecolor='black')
            ax.set_title(f'Bar Chart (Binned) of {column}')
            ax.set_xlabel(column)
            ax.set_ylabel('Frequency')

    elif chart_type == "Line Plot":
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column].dropna().plot(kind='line', ax=ax, color='blue')
            ax.set_title(f'Line Plot of {column}')
            ax.set_xlabel(column)
            ax.set_ylabel('Value')
        else:
            ax.text(0.5, 0.5, f'Line Plot not applicable for categorical column "{column}".', ha='center', va='center')

    elif chart_type == "Pie Chart":
        if not pd.api.types.is_numeric_dtype(df[column]):
            df[column].dropna().value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%',
                                                    colors=sns.color_palette("Set3", len(df[column].dropna().unique())))
            ax.set_title(f'Pie Chart of {column}')
            ax.set_ylabel('')
        else:
            ax.text(0.5, 0.5, f'Pie Chart not applicable for numeric column "{column}".', ha='center', va='center')

    elif chart_type == "Heatmap":
        corr_matrix = df.corr()  # Calculate correlation matrix for numeric columns
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, linewidths=0.5)
        ax.set_title('Correlation Heatmap')

    plt.tight_layout()
    return fig


def extract_subtable(df, row_indices=None, col_indices=None, keep=False):
    """
    Extract a subtable from the dataframe based on the rows and columns provided by the user.

    Parameters:
    - row_indices: List of row indices to keep/remove (can be integers or column names).
    - col_indices: List of column indices or column names to keep/remove.
    - keep: Boolean flag indicating whether to keep (True) or remove (False) the specified rows/columns.

    Returns:
    - A DataFrame with the extracted subtable, or None if invalid.
    """
    if df is None:
        return None

    if row_indices is not None:
        if isinstance(row_indices[0], str):
            row_indices = [df.index.get_loc(r) if r in df.index else r for r in row_indices]
        invalid_rows = [r for r in row_indices if not isinstance(r, int) or r not in df.index]
        if invalid_rows:
            sg.popup(f"Invalid rows: {invalid_rows}. Please enter valid row indices.")
            return None

    if col_indices is not None:
        if isinstance(col_indices[0], str):
            col_indices = [df.columns.get_loc(c) if c in df.columns else c for c in col_indices]
        invalid_cols = [c for c in col_indices if not isinstance(c, int) or c not in range(len(df.columns))]
        if invalid_cols:
            sg.popup(f"Invalid columns: {invalid_cols}. Please enter valid column indices or names.")
            return None

    if row_indices is not None:
        if keep:
            df = df.iloc[row_indices, :]
        else:
            df = df.drop(df.index[row_indices])

    if col_indices is not None:
        if keep:
            df = df.iloc[:, col_indices]
        else:
            df = df.drop(df.columns[col_indices], axis=1)

    return df

def remove_columns(df, cols_to_remove):
    """
    Removes specified columns from the DataFrame.
    """
    for c in cols_to_remove:
        if c in df.columns:
            df = df.drop(columns=c)
    return df

def replace_values(df, column, old_value, new_value):
    """
    Replace specific value in the column with the new value.
    Handles both regular columns and categorical columns.
    """
    if column not in df.columns:
        sg.popup(f"Column '{column}' does not exist!")
        return df

    if isinstance(df[column].dtype, pd.CategoricalDtype):
        if old_value in df[column].cat.categories:
            if new_value not in df[column].cat.categories:
                df[column] = df[column].cat.set_categories(list(df[column].cat.categories) + [new_value])
            df[column] = df[column].replace(old_value, new_value)
        else:
            sg.popup(f"Old value '{old_value}' not found in categorical column '{column}'.")
            return df
    else:
        df[column] = df[column].replace(old_value, new_value)

    return df

def replace_all_values(df, column, new_value):
    """
    Replace all values in the column with the new value.
    Handles both regular columns and categorical columns.
    """
    if column not in df.columns:
        sg.popup(f"Column '{column}' does not exist!")
        return df

    # For categorical columns, rename all categories to the new value
    if isinstance(df[column].dtype, pd.CategoricalDtype):
        # Check if new_value already exists in categories
        if new_value not in df[column].cat.categories:
            # Add the new value as a category
            df[column] = df[column].cat.set_categories(list(df[column].cat.categories) + [new_value])
        # Set all values in the column to new_value
        df[column] = new_value
    else:
        # For non-categorical columns, replace all values
        df[column] = new_value

    return df


def handle_missing_values(df, strategy):
    """
    Handles missing values based on the strategy:
    - 'remove': drops rows with missing values.
    - 'fill_mean': fills numeric columns with the mean.
    - 'fill_median': fills numeric columns with the median.
    - 'fill_mode': fills numeric columns with the mode.
    """
    if strategy == 'remove':
        df = df.dropna()
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if strategy == 'fill_mean':
                df.loc[:, col] = df[col].fillna(df[col].mean())
            elif strategy == 'fill_median':
                df.loc[:, col] = df[col].fillna(df[col].median())
            elif strategy == 'fill_mode':
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df.loc[:, col] = df[col].fillna(mode_val.iloc[0])
    return df

def remove_duplicates(df):
    """
    Removes duplicate rows from the dataframe.
    """
    return df.drop_duplicates()

def one_hot_encoding(df, column):
    """
    Perform One-Hot Encoding on the specified column.
    """
    return pd.get_dummies(df, columns=[column], drop_first=True)

def binary_encoding(df, column):
    """
    Perform Binary Encoding on the specified column using category codes.
    """
    df[column] = df[column].astype('category')
    df[column] = df[column].cat.codes
    return df

def target_encoding(df, column, target):
    """
    Perform Target Encoding on the specified column using the target column.
    """
    encoding_map = df.groupby(column)[target].mean()
    df[column] = df[column].map(encoding_map)
    return df


def scale_columns(df, cols, method='standard'):
    """
    Scales the specified numeric columns using StandardScaler or MinMaxScaler.
    """
    if df is None or df.empty:
        raise ValueError("The dataframe is empty. Please load a valid dataset.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    chosen_cols = [c for c in cols if c in numeric_cols]

    if not chosen_cols:
        raise ValueError("No valid numeric columns selected.")

    scaler = StandardScaler() if method == 'standard' else MinMaxScaler()

    scaled_values = scaler.fit_transform(df[chosen_cols])

    df.loc[:, chosen_cols] = scaled_values.astype('float64')

    return df

def add_symbolic_column(df):
    """
    Adds a symbolic column 'age_category' based on the 'age' column.
    Divides age into intervals.
    """
    if 'age' in df.columns and pd.api.types.is_numeric_dtype(df['age']):
        df['age_category'] = pd.cut(
            df['age'],
            bins=[0, 25, 45, 65, 150],
            labels=['Young', 'Adult', 'Senior', 'Elder'],
            right=False
        )
    return df


def logistic_regression(df):
    """
    Runs Logistic Regression to predict 'income' (<=50K or >50K) based on numeric features.
    Returns the accuracy on the test set.
    """
    if 'income' not in df.columns:
        return None, "No 'income' column found."
    local_df = df.dropna(subset=['income']).copy()
    local_df = local_df[local_df['income'].isin(['<=50K', '>50K'])]
    encoder = LabelEncoder()
    local_df['income_encoded'] = encoder.fit_transform(local_df['income'])
    numeric_cols = local_df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols if c not in ['income_encoded']]
    if not feature_cols:
        return None, "No numeric features available for classification."
    X = local_df[feature_cols]
    y = local_df['income_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    try:
        clf.fit(X_train, y_train)
    except Exception as e:
        return None, f"Error in Logistic Regression fit: {e}"
    accuracy = clf.score(X_test, y_test)
    return accuracy, None


def kmeans_clustering(df, n_clusters=2):
    """
    Runs K-Means clustering on numeric columns.
    Returns the cluster labels.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 1:
        return None, "No numeric columns available for clustering."
    local_df = df.dropna(subset=numeric_cols).copy()
    X = local_df[numeric_cols]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    try:
        labels = kmeans.fit_predict(X)
    except Exception as e:
        return None, f"Error in KMeans: {e}"
    return labels, None


# ------------------ Layout Definition ------------------ #

# Tab 0: Logo & Creators Info
tab0_layout = [
    [sg.Text("", size=(1, 1))],
    [sg.Column([
        [sg.Text("Subject: Data Warehousing", font=("Helvetica", 16), justification='center')],
        [sg.Image(filename="assets/logo.png", key="-LOGO-", size=(250, 250))],
        [sg.Text("Created by:", font=("Helvetica", 16), justification='center')],
        [sg.Text("Dawid Olko", font=("Helvetica", 14), justification='center')],
        [sg.Text("Piotr Smoła", font=("Helvetica", 14), justification='center')],
        [sg.Button("Go to data", key="-ENTER-", font=("Helvetica", 14), size=(20, 1), pad=((0, 0), (20, 0)))]
    ], justification='center', element_justification='center')],
    [sg.Text("", size=(1, 1))],
]

# Tab 1: Data & Statistics
tab1_layout = [
    [sg.Text("Select Dataset to Load:")],
    [sg.Combo(["Adult Dataset", "Kidney Disease Dataset"], default_value="Adult Dataset", key="-SELECT_FILE-"),
     sg.Button("Load Data", key="-LOAD-")],
    [sg.Multiline(size=(100, 6), key="-DATA_INFO-", disabled=True)],
    [sg.Button("Compute Stats", key="-STATS-"), sg.Button("Correlation", key="-CORR-")],

    [sg.Text("Numeric Columns Statistics:")],
    [sg.Table(values=[], headings=["Column", "Min", "Max", "Mean", "Median", "Std", "Mode", "Variance", "Skewness", "Kurtosis"],
              key="-NUMERIC_STATS-", auto_size_columns=True, justification='center', expand_x=True, expand_y=True)],

    [sg.Text("Categorical Columns Statistics:")],
    [sg.Table(values=[], headings=["Column", "Value Counts", "Mode"],
              key="-CATEGORICAL_STATS-", auto_size_columns=True, justification='center', expand_x=True, expand_y=True)],

    [sg.Text("Correlation Results:")],
    [sg.Multiline(size=(100, 10), key="-CORR_OUT-", disabled=True)]
]

# Tab 2: Extract Subtable
tab2_layout = [
    [sg.Text("Subtable Extraction: Select whether to Remove or Keep Rows/Columns")],

    [sg.Radio("Remove Rows/Columns", "EXTRACTION_TYPE", default=True, key="-REMOVE_EXTRACT-"),
     sg.Radio("Keep Rows/Columns", "EXTRACTION_TYPE", key="-KEEP_EXTRACT-")],

    [sg.Text("Rows (comma separated indices or names to remove/keep):"),
     sg.InputText(key="-ROW_INPUT-", size=(30, 1))],

    [sg.Text("Columns (comma separated indices or names to remove/keep):"),
     sg.InputText(key="-COL_INPUT-", size=(30, 1))],

    [sg.Button("Extract Subtable", key="-EXTRACT_BTN-")],

    [sg.Multiline(size=(1200, 8), key="-EXTRACT_OUT-", disabled=True)],

    [sg.Text("Replacement: Enter the column name and values to replace.")],

    [sg.Text("Select Column to Replace:"),
     sg.InputText(key="-REPLACE_COL-", size=(30, 1))],

    [sg.Text("Current Value to Replace:"),
     sg.InputText(key="-OLD_VAL-", size=(30, 1))],

    [sg.Text("New Value:"),
     sg.InputText(key="-NEW_VAL-", size=(30, 1))],

    [sg.Button("Replace Values in Column", key="-REPLACE_BTN-")],

    [sg.Text("Optionally, replace all values in a column with a specific value.")],

    [sg.Text("Replace All in Column:"),
     sg.InputText(key="-ALL_REPLACE_COL-", size=(30, 1))],

    [sg.Text("New Value to Replace All with:"),
     sg.InputText(key="-ALL_NEW_VAL-", size=(30, 1))],

    [sg.Button("Replace All Values", key="-REPLACE_ALL_BTN-")]
]

# Tab 3: Scaling & Visualization
tab3_layout = [
    [sg.Text("Columns to Scale (comma separated):"), sg.InputText(key="-SCALE_COLS-", size=(30, 1))],
    [sg.Radio("StandardScaler", "SCALER", default=True, key="-STD_SCALER-"),
     sg.Radio("MinMaxScaler", "SCALER", key="-MINMAX_SCALER-")],
    [sg.Button("Apply Scaling", key="-APPLY_SCALING-")],

    [sg.Text("Scaled Data Preview:")],
    [sg.Table(values=[], headings=[], key="-SCALED_DATA-", auto_size_columns=True, justification='center',
              expand_x=True, expand_y=True)],

    [sg.HorizontalSeparator()],

    [sg.Text("Select Column to Plot:"), sg.Combo([], key="-PLOT_SELECT-", size=(20, 1))],

    [sg.Text("Select Chart Type:")],
    [sg.Radio("Histogram", "CHART", default=True, key="-CHART_HIST-"),
     sg.Radio("Boxplot", "CHART", key="-CHART_BOX-"),
     sg.Radio("Bar Chart", "CHART", key="-CHART_BAR-"),
     sg.Radio("Line Plot", "CHART", key="-CHART_LINE-"),
     sg.Radio("Pie Chart", "CHART", key="-CHART_PIE-")],

    [sg.Button("Generate Plot", key="-PLOT_BTN-")],
    [sg.Canvas(key="-CANVAS-", size=(600, 400), expand_x=True, expand_y=True)]
]

# Tab 4: Data Cleaning & Transformation
tab4_layout = [
    [sg.Text("Handling Missing Values:")],
    [sg.Combo(["remove", "fill_mean", "fill_median", "fill_mode"], default_value="remove", key="-MISSING_STRATEGY-"),
     sg.Button("Apply Missing Values Handling", key="-APPLY_MISSING-")],

    [sg.Text("Remove Duplicate Rows:"), sg.Button("Remove Duplicates", key="-REMOVE_DUPLICATES-")],

    [sg.Text("Encode Categorical Column:")],
    [sg.Text("Select Column to Encode:"),
     sg.Combo([], key="-ENCODE_COL-", size=(20, 1))],
    [sg.Radio("One-Hot Encoding", "ENCODE_TYPE", default=True, key="-ONE_HOT-"),
     sg.Radio("Binary Encoding", "ENCODE_TYPE", key="-BINARY_ENCODE-"),
     sg.Radio("Target Encoding", "ENCODE_TYPE", key="-TARGET_ENCODE-")],

    [sg.Text("Select Target Column (for Target Encoding):"),
     sg.Combo([], key="-TARGET_COL-", size=(20, 1))],

    [sg.Button("Apply Encoding", key="-APPLY_ENCODING-")],

    [sg.Text("Data Preview (After Cleaning and Encoding):")],
    [sg.Multiline(size=(80, 10), key="-CLEANED_DATA-", disabled=True)]
]

background_color = '#64778d'
selected_background_color = '#4CAF50'
selected_text_color = 'white'

layout = [
    [sg.Column([
        [sg.TabGroup([[
            sg.Tab("Creators & Info", tab0_layout, expand_x=True, expand_y=True, background_color=background_color),
            sg.Tab("Data & Stats", tab1_layout, expand_x=True, expand_y=True, background_color=background_color),
            sg.Tab("Replacement & Subtable", tab2_layout, expand_x=True, expand_y=True, background_color=background_color),
            sg.Tab("Scaling & Visualization", tab3_layout, expand_x=True, expand_y=True, background_color=background_color),
            sg.Tab("Data Cleaning & Transformation", tab4_layout, expand_x=True, expand_y=True, background_color=background_color)
        ]], tab_location='top', font=("Helvetica", 14, "bold"), expand_x=True, expand_y=True, key='-TABGROUP-', size=(1200, 800))]
    ], scrollable=True, vertical_scroll_only=True, expand_x=True, expand_y=True)]
]

window = sg.Window("DataFusion - Project", layout, resizable=True, finalize=True, element_justification='center')

# ------------------ Event Loop ------------------ #
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break

    if event == "-ENTER-":
        window['-TABGROUP-'].Widget.select(1)

    if event == "-LOAD-":
        try:
            selected_file = values["-SELECT_FILE-"]
            df = load_dataset(selected_file)
            window["-DATA_INFO-"].update("Dataset loaded with columns: " + ", ".join(df.columns))
            window["-PLOT_SELECT-"].update(values=list(df.columns), value=list(df.columns)[0])
            sg.popup("Dataset loaded successfully!", keep_on_top=True)
        except Exception as e:
            sg.popup(f"Error loading dataset: {e}", keep_on_top=True)

    elif event == "-STATS-":
        if df is None:
            sg.popup("Please load the dataset first.")
        else:
            dataset_type = values["-SELECT_FILE-"]

            stats_data = compute_statistics(df, dataset_type)

            numeric_stats = [stat for stat in stats_data if isinstance(stat[1], (int, float))]
            categorical_stats = [stat for stat in stats_data if isinstance(stat[1], dict)]  # For categorical stats

            window["-NUMERIC_STATS-"].update(values=numeric_stats)
            window["-CATEGORICAL_STATS-"].update(values=categorical_stats)

    elif event == "-CORR-":
        if df is None:
            sg.popup("Please load the dataset first.")
        else:
            pearson_corr, spearman_corr = compute_correlation(df)

            if pearson_corr is None or spearman_corr is None:
                    window["-CORR_OUT-"].update("Not enough numeric columns for correlation.")
            else:
                pearson_str = f"Pearson Correlation:\n{pearson_corr.to_string()}\n\n"
                spearman_str = f"Spearman Correlation:\n{spearman_corr.to_string()}"

                window["-CORR_OUT-"].update(pearson_str + spearman_str)

    # Tab 2: Cleaning & Subtable
    elif event == "-EXTRACT_BTN-":
        if df is None:
            sg.popup("Please load the dataset first.")
        else:
            try:
                row_input = values["-ROW_INPUT-"].strip()
                col_input = values["-COL_INPUT-"].strip()

                if row_input:
                    row_indices = [int(i) if i.isdigit() else i for i in row_input.split(",")]
                else:
                    row_indices = None

                if col_input:
                    col_indices = [int(i) if i.isdigit() else i for i in col_input.split(",")]
                else:
                    col_indices = None

                keep = values["-KEEP_EXTRACT-"]

                sub_df = extract_subtable(df, row_indices, col_indices, keep)

                if sub_df is None or sub_df.empty:
                    window["-EXTRACT_OUT-"].update("Invalid range or empty subtable.")
                else:
                    window["-EXTRACT_OUT-"].update(sub_df.to_string())
            except Exception as e:
                sg.popup(f"Error extracting subtable: {e}")

    elif event == "-REPLACE_BTN-":
        if df is None:
            sg.popup("Please load the dataset first.")
        else:
            col_to_replace = values["-REPLACE_COL-"]
            old_value = values["-OLD_VAL-"]
            new_value = values["-NEW_VAL-"]

            if not col_to_replace or not old_value or not new_value:
                sg.popup("Please fill in all fields for column, old value, and new value.")
            else:
                try:
                    df = replace_values(df, col_to_replace, old_value, new_value)
                    window["-EXTRACT_OUT-"].update(
                        f"Replaced '{old_value}' with '{new_value}' in column '{col_to_replace}'")
                except Exception as e:
                    sg.popup(f"Error replacing values: {e}")

    elif event == "-REPLACE_ALL_BTN-":
        if df is None:
            sg.popup("Please load the dataset first.")
        else:
            col_to_replace_all = values["-ALL_REPLACE_COL-"]
            new_value_all = values["-ALL_NEW_VAL-"]

            if not col_to_replace_all or not new_value_all:
                sg.popup("Please fill in all fields for column and new value.")
            else:
                try:
                    df = replace_all_values(df, col_to_replace_all, new_value_all)
                    window["-EXTRACT_OUT-"].update(
                        f"Replaced all values in column '{col_to_replace_all}' with '{new_value_all}'")
                except Exception as e:
                    sg.popup(f"Error replacing all values: {e}")

    # Tab 3: Scaling & Visualization
    elif event == "-APPLY_SCALING-":
        if df is None:
            sg.popup("Please load the dataset first.")
        else:
            cols_str = values["-SCALE_COLS-"]
            cols_list = [c.strip() for c in cols_str.split(",") if c.strip() != ""]  # Clean the input

            method = "standard" if values["-STD_SCALER-"] else "minmax"

            try:
                df = scale_columns(df, cols_list, method=method)

                scaled_data = df[cols_list].head(10).values.tolist()
                window["-SCALED_DATA-"].update(values=scaled_data, headings=cols_list)

                sg.popup(f"Applied {method} scaling to columns: {cols_list}")

                window["-PLOT_SELECT-"].update(values=list(df.columns), value=list(df.columns)[0])
            except Exception as e:
                sg.popup(f"Error scaling columns: {e}")

    elif event == "-PLOT_BTN-":
        if df is None:
            sg.popup("Please load the dataset first.")
        else:
            col_to_plot = values["-PLOT_SELECT-"]

            if values["-CHART_HIST-"]:
                chart_type = "Histogram"
            elif values["-CHART_BOX-"]:
                chart_type = "Boxplot"
            elif values["-CHART_BAR-"]:
                chart_type = "Bar Chart"
            elif values["-CHART_LINE-"]:
                chart_type = "Line Plot"
            elif values["-CHART_PIE-"]:
                chart_type = "Pie Chart"
            else:
                chart_type = "Histogram"

            fig = generate_plot(df, col_to_plot, chart_type)

            if figure_canvas_agg:
                figure_canvas_agg.get_tk_widget().forget()

            figure_canvas_agg = draw_figure(window["-CANVAS-"].TKCanvas, fig)

    # Tab 4: ML & Clustering
    elif event == "-APPLY_MISSING-":
        if df is None:
            sg.popup("Please load the dataset first.")
        else:
            strategy = values["-MISSING_STRATEGY-"]
            try:
                df = handle_missing_values(df, strategy)
                sg.popup(f"Missing values handled with strategy: {strategy}")
                window["-CLEANED_DATA-"].update(df.head().to_string())
            except Exception as e:
                sg.popup(f"Error handling missing values: {e}")

    elif event == "-REMOVE_DUPLICATES-":
        if df is None:
            sg.popup("Please load the dataset first.")
        else:
            try:
                df = remove_duplicates(df)
                sg.popup("Duplicate rows removed successfully.")
                window["-CLEANED_DATA-"].update(df.head().to_string())
            except Exception as e:
                sg.popup(f"Error removing duplicates: {e}")

    elif event == "-APPLY_ENCODING-":
        if df is None:
            sg.popup("Please load the dataset first.")
        else:
            column = values["-ENCODE_COL-"]
            if values["-ONE_HOT-"]:
                df = one_hot_encoding(df, column)
                sg.popup(f"One-Hot Encoding applied to column: {column}")
            elif values["-BINARY_ENCODE-"]:
                df = binary_encoding(df, column)
                sg.popup(f"Binary Encoding applied to column: {column}")
            elif values["-TARGET_ENCODE-"]:
                target_column = values["-TARGET_COL-"]
                if target_column == "":
                    sg.popup("Please select a target column for Target Encoding.")
                else:
                    df = target_encoding(df, column, target_column)
                    sg.popup(f"Target Encoding applied to column: {column} using target column: {target_column}")
            window["-CLEANED_DATA-"].update(df.head().to_string())

            window.close()