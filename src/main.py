import threading

import PySimpleGUI as sg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

# ------------------ Global GUI Settings ------------------ #
sg.set_options(font=("Helvetica", 12))

# ------------------ Helper Functions ------------------ #

def load_adult_dataset():
    """
    Loads the Adult dataset from a local file.
    Replaces '?' with NaN.
    """
    col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                 'marital-status', 'occupation', 'relationship', 'race',
                 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                 'native-country', 'income']
    try:
        df = pd.read_csv('../database/adult/adult.data', header=None, names=col_names, skipinitialspace=True)
    except Exception as e:
        df = pd.read_csv('adult.data', header=None, names=col_names, skipinitialspace=True)
    df.replace('?', np.nan, inplace=True)
    return df

def compute_statistics(df):
    """
    Computes basic statistics (min, max, mean, median, std, mode) for numeric columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats_data = []
    for col in numeric_cols:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            stats_data.append([col, None, None, None, None, None, None])
        else:
            stats_data.append([
                col,
                col_data.min(),
                col_data.max(),
                round(col_data.mean(), 2),
                round(col_data.median(), 2),
                round(col_data.std(), 2),
                col_data.mode().iloc[0] if not col_data.mode().empty else None
            ])
    return stats_data

def compute_correlation(df):
    """
    Returns the correlation matrix for numeric columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return None
    corr_matrix = df[numeric_cols].corr()
    return corr_matrix

def generate_plot(df, column, chart_type):
    """
    Generates a plot based on the selected column and chart type.
    - "Histogram": histogram for numeric or bar chart for categorical.
    - "Boxplot": for numeric columns.
    - "Bar Chart": for categorical columns (or binned for numeric).
    """
    fig, ax = plt.subplots(figsize=(6,4))
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
            ax.boxplot(df[column].dropna())
            ax.set_title(f'Boxplot of {column}')
            ax.set_xticks([1])
            ax.set_xticklabels([column])
        else:
            ax.text(0.5, 0.5, f'Boxplot not applicable for categorical column "{column}".', ha='center', va='center')
    elif chart_type == "Bar Chart":
        if not pd.api.types.is_numeric_dtype(df[column]):
            df[column].dropna().value_counts().plot(kind='bar', ax=ax, color='green', edgecolor='black')
            ax.set_title(f'Bar Chart of {column}')
            ax.set_xlabel(column)
            ax.set_ylabel('Count')
        else:
            # For numeric, create binned bar chart
            bins = 10
            counts, bin_edges = np.histogram(df[column].dropna(), bins=bins)
            ax.bar(range(bins), counts, width=0.8, color='purple', edgecolor='black')
            ax.set_title(f'Bar Chart (Binned) of {column}')
            ax.set_xlabel(column)
            ax.set_ylabel('Frequency')
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

def extract_subtable(df, row_start, row_end, col_start, col_end):
    """
    Returns a subtable using iloc with the given row and column ranges.
    """
    try:
        rs = int(row_start)
        re = int(row_end)
        cs = int(col_start)
        ce = int(col_end)
        return df.iloc[rs:re, cs:ce]
    except:
        return None

def remove_columns(df, cols_to_remove):
    """
    Removes specified columns from the DataFrame.
    """
    for c in cols_to_remove:
        if c in df.columns:
            df = df.drop(columns=c)
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

def scale_columns(df, cols, method='standard'):
    """
    Scales the specified numeric columns using StandardScaler or MinMaxScaler.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    chosen_cols = [c for c in cols if c in numeric_cols]
    if not chosen_cols:
        return df
    scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
    df.loc[:, chosen_cols] = scaler.fit_transform(df[chosen_cols])
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

# Tab 1: Data & Statistics
tab1_layout = [
    [sg.Text("Load Adult Dataset:")],
    [sg.Button("Load Data", key="-LOAD-"), sg.Button("Show Head", key="-SHOW_HEAD-")],
    [sg.Multiline(size=(100, 6), key="-DATA_INFO-", disabled=True)],
    [sg.Button("Compute Stats", key="-STATS-"), sg.Button("Correlation", key="-CORR-")],
    [sg.Table(values=[], headings=["Column", "Min", "Max", "Mean", "Median", "Std", "Mode"],
              key="-STATS_TABLE-", auto_size_columns=True, justification='center', expand_x=True, expand_y=True)],
    [sg.Multiline(size=(100, 5), key="-CORR_OUT-", disabled=True)]
]

# Tab 2: Cleaning & Subtable
tab2_layout = [
    [sg.Text("Missing Values Strategy:"),
     sg.Combo(["remove", "fill_mean", "fill_median", "fill_mode"], default_value="remove", key="-MISSING_STRATEGY-"),
     sg.Button("Apply", key="-APPLY_MISSING-")],
    [sg.Text("Remove Columns (comma separated):"), sg.InputText(key="-REMOVE_COLS-", size=(30,1)),
     sg.Button("Remove", key="-REMOVE_BTN-")],
    [sg.Text("Extract Subtable: RowStart, RowEnd, ColStart, ColEnd")],
    [sg.InputText("0", size=(5,1), key="-ROW_START-"),
     sg.InputText("5", size=(5,1), key="-ROW_END-"),
     sg.InputText("0", size=(5,1), key="-COL_START-"),
     sg.InputText("5", size=(5,1), key="-COL_END-"),
     sg.Button("Extract", key="-EXTRACT_BTN-")],
    [sg.Multiline(size=(100, 8), key="-EXTRACT_OUT-", disabled=True)],
    [sg.Text("Add Symbolic Column (e.g. age_category)"), sg.Button("Add Column", key="-ADD_COL-")]
]

# Tab 3: Scaling & Visualization
tab3_layout = [
    [sg.Text("Columns to Scale (comma separated):"), sg.InputText(key="-SCALE_COLS-", size=(30,1))],
    [sg.Radio("StandardScaler", "SCALER", default=True, key="-STD_SCALER-"),
     sg.Radio("MinMaxScaler", "SCALER", key="-MINMAX_SCALER-")],
    [sg.Button("Apply Scaling", key="-APPLY_SCALING-")],
    [sg.HorizontalSeparator()],
    [sg.Text("Select Column to Plot:"),
     sg.Combo([], key="-PLOT_SELECT-", size=(20,1))],
    [sg.Text("Select Chart Type:")],
    [sg.Radio("Histogram", "CHART", default=True, key="-CHART_HIST-"),
     sg.Radio("Boxplot", "CHART", key="-CHART_BOX-"),
     sg.Radio("Bar Chart", "CHART", key="-CHART_BAR-")],
    [sg.Button("Generate Plot", key="-PLOT_BTN-")],
    [sg.Canvas(key="-CANVAS-", size=(600, 400), expand_x=True, expand_y=True)]
]

# Tab 4: ML & Clustering
tab4_layout = [
    [sg.Text("Logistic Regression on 'income':"), sg.Button("Run Logistic Regression", key="-LOGREG-")],
    [sg.Multiline(size=(80, 3), key="-LOGREG_OUT-", disabled=True)],
    [sg.HorizontalSeparator()],
    [sg.Text("K-Means Clustering (# of clusters):"), sg.InputText("2", key="-N_CLUSTERS-", size=(5,1)),
     sg.Button("Run K-Means", key="-KMEANS-")],
    [sg.Multiline(size=(80, 3), key="-KMEANS_OUT-", disabled=True)]
]

# Main Layout with TabGroup (tabs on top with increased font)
layout = [
    [sg.TabGroup([[
         sg.Tab("Data & Stats", tab1_layout, expand_x=True, expand_y=True),
         sg.Tab("Cleaning & Subtable", tab2_layout, expand_x=True, expand_y=True),
         sg.Tab("Scaling & Visualization", tab3_layout, expand_x=True, expand_y=True),
         sg.Tab("ML & Clustering", tab4_layout, expand_x=True, expand_y=True)
    ]], tab_location='top', font=("Helvetica", 14, "bold"), expand_x=True, expand_y=True)]
]

window = sg.Window("DataFusion - Project", layout, resizable=True, finalize=True, size=(800, 600))

# ------------------ Global Variables ------------------ #
df_adult = None
figure_canvas_agg = None

# ------------------ Event Loop ------------------ #
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break

    # Tab 1: Data & Statistics
    if event == "-LOAD-":
        try:
            df_adult = load_adult_dataset()
            window["-DATA_INFO-"].update("Dataset loaded with columns: " + ", ".join(df_adult.columns))
            # Update plot column combo with available columns
            window["-PLOT_SELECT-"].update(values=list(df_adult.columns), value='age')
            sg.popup("Dataset loaded successfully!", keep_on_top=True)
        except Exception as e:
            sg.popup(f"Error loading dataset: {e}", keep_on_top=True)

    elif event == "-SHOW_HEAD-":
        if df_adult is None:
            sg.popup("Please load the dataset first.")
        else:
            head_str = df_adult.head(5).to_string()
            window["-DATA_INFO-"].update(head_str)

    elif event == "-STATS-":
        if df_adult is None:
            sg.popup("Please load the dataset first.")
        else:
            stats_data = compute_statistics(df_adult)
            window["-STATS_TABLE-"].update(values=stats_data)

    elif event == "-CORR-":
        if df_adult is None:
            sg.popup("Please load the dataset first.")
        else:
            corr_matrix = compute_correlation(df_adult)
            if corr_matrix is None:
                window["-CORR_OUT-"].update("Not enough numeric columns for correlation.")
            else:
                window["-CORR_OUT-"].update(corr_matrix.to_string())

    # Tab 2: Cleaning & Subtable
    elif event == "-APPLY_MISSING-":
        if df_adult is None:
            sg.popup("Please load the dataset first.")
        else:
            strategy = values["-MISSING_STRATEGY-"]
            try:
                df_adult = handle_missing_values(df_adult, strategy)
                sg.popup(f"Missing values handled with strategy: {strategy}")
            except Exception as e:
                sg.popup(f"Error handling missing values: {e}")

    elif event == "-REMOVE_BTN-":
        if df_adult is None:
            sg.popup("Please load the dataset first.")
        else:
            cols_str = values["-REMOVE_COLS-"]
            cols_list = [c.strip() for c in cols_str.split(",") if c.strip() != ""]
            try:
                df_adult = remove_columns(df_adult, cols_list)
                sg.popup(f"Removed columns: {cols_list}")
                # Also update the plot combo in case columns changed
                window["-PLOT_SELECT-"].update(values=list(df_adult.columns))
            except Exception as e:
                sg.popup(f"Error removing columns: {e}")

    elif event == "-EXTRACT_BTN-":
        if df_adult is None:
            sg.popup("Please load the dataset first.")
        else:
            row_start = values["-ROW_START-"]
            row_end = values["-ROW_END-"]
            col_start = values["-COL_START-"]
            col_end = values["-COL_END-"]
            sub_df = extract_subtable(df_adult, row_start, row_end, col_start, col_end)
            if sub_df is None or sub_df.empty:
                window["-EXTRACT_OUT-"].update("Invalid range or empty subtable.")
            else:
                window["-EXTRACT_OUT-"].update(sub_df.to_string())

    elif event == "-ADD_COL-":
        if df_adult is None:
            sg.popup("Please load the dataset first.")
        else:
            try:
                df_adult = add_symbolic_column(df_adult)
                sg.popup("Symbolic column (age_category) added!")
                window["-PLOT_SELECT-"].update(values=list(df_adult.columns))
            except Exception as e:
                sg.popup(f"Error adding column: {e}")

    # Tab 3: Scaling & Visualization
    elif event == "-APPLY_SCALING-":
        if df_adult is None:
            sg.popup("Please load the dataset first.")
        else:
            cols_str = values["-SCALE_COLS-"]
            cols_list = [c.strip() for c in cols_str.split(",") if c.strip() != ""]
            method = "standard" if values["-STD_SCALER-"] else "minmax"
            try:
                df_adult = scale_columns(df_adult, cols_list, method=method)
                sg.popup(f"Applied {method} scaling to columns: {cols_list}")
            except Exception as e:
                sg.popup(f"Error scaling columns: {e}")

    elif event == "-PLOT_BTN-":
        if df_adult is None:
            sg.popup("Please load the dataset first.")
        else:
            col_to_plot = values["-PLOT_SELECT-"]
            # Determine chart type
            if values["-CHART_HIST-"]:
                chart_type = "Histogram"
            elif values["-CHART_BOX-"]:
                chart_type = "Boxplot"
            elif values["-CHART_BAR-"]:
                chart_type = "Bar Chart"
            else:
                chart_type = "Histogram"
            fig = generate_plot(df_adult, col_to_plot, chart_type)
            if figure_canvas_agg:
                figure_canvas_agg.get_tk_widget().forget()
            figure_canvas_agg = draw_figure(window["-CANVAS-"].TKCanvas, fig)

    # Tab 4: ML & Clustering
    elif event == "-LOGREG-":
        if df_adult is None:
            sg.popup("Please load the dataset first.")
        else:
            acc, err = logistic_regression(df_adult)
            if err:
                window["-LOGREG_OUT-"].update(f"Error: {err}")
            else:
                window["-LOGREG_OUT-"].update(f"Logistic Regression accuracy: {round(acc, 4)}")

    elif event == "-KMEANS-":
        if df_adult is None:
            sg.popup("Please load the dataset first.")
        else:
            n_clusters_str = values["-N_CLUSTERS-"]
            try:
                n_clusters = int(n_clusters_str)
                labels, err = kmeans_clustering(df_adult, n_clusters=n_clusters)
                if err:
                    window["-KMEANS_OUT-"].update(f"Error: {err}")
                else:
                    window["-KMEANS_OUT-"].update(f"K-Means assigned cluster labels: {labels[:30]}...\n(Showing first 30)")
            except Exception as e:
                window["-KMEANS_OUT-"].update(f"Error: {e}")

window.close()
