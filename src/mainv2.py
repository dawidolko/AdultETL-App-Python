# import PySimpleGUI as sg
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.cluster import KMeans
# from scipy.stats import kurtosis, skew
#
# # ------------------ Global GUI Settings ------------------ #
#
# # Set theme for a modern look
# sg.theme('LightGrey1')  # More modern theme
#
# sg.set_options(font=("Helvetica", 12))
# df = None
# figure_canvas_agg = None
#
#
# # ------------------ Helper Functions ------------------ #
#
# def load_dataset(file_name):
#     """
#     Loads the selected dataset from a file.
#     """
#     if file_name == "Adult Dataset":
#         col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
#                      'marital-status', 'occupation', 'relationship', 'race',
#                      'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
#                      'native-country', 'income']
#         try:
#             df = pd.read_csv('../database/adult/adult.data', header=None, names=col_names, skipinitialspace=True)
#         except Exception as e:
#             try:
#                 df = pd.read_csv('adult.data', header=None, names=col_names, skipinitialspace=True)
#             except Exception as e2:
#                 raise ValueError(f"Failed to load Adult Dataset: {e2}")
#
#         df.replace('?', np.nan, inplace=True)
#
#         # Force convert numeric columns - critical step
#         numeric_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
#         for col in numeric_cols:
#             print(f"Converting {col} to numeric during dataset load...")
#             df[col] = pd.to_numeric(df[col], errors='coerce')
#
#         # Print data types after conversion
#         print("Column data types after loading Adult Dataset:")
#         print(df.dtypes)
#
#         # Set remaining columns as categorical
#         categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
#                             'race', 'sex', 'native-country', 'income']
#         for col in categorical_cols:
#             df[col] = df[col].astype('category')
#
#     elif file_name == "Kidney Disease Dataset":
#         try:
#             df = pd.read_csv('../database/chronic/kidney_disease.csv')
#         except Exception as e:
#             raise ValueError(f"Error loading kidney disease dataset: {e}")
#
#         df.replace('?', np.nan, inplace=True)
#
#         categorical_cols = ['fbs', 'restecg', 'exang', 'slope', 'thal', 'classification']
#         for col in categorical_cols:
#             if col in df.columns:
#                 df[col] = df[col].astype('category')
#
#     return df
#
#
# def compute_statistics(df, dataset_type):
#     """
#     Computes statistics for both numeric and categorical columns:
#     - For numeric columns: min, max, mean, median, std, mode, variance, skewness, kurtosis.
#     - For categorical columns: count, mode.
#     """
#     stats_data = []
#
#     # Explicitly identify numeric columns for Adult Dataset
#     if dataset_type == "Adult Dataset":
#         # Force these columns to be numeric
#         numeric_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
#
#         # Convert each column to numeric explicitly and handle errors
#         for col in numeric_cols:
#             if col in df.columns:
#                 print(f"Converting {col} to numeric for stats calculation...")
#                 # Make a copy to avoid SettingWithCopyWarning
#                 df = df.copy()
#                 df[col] = pd.to_numeric(df[col], errors='coerce')
#
#         # Verify the conversion worked
#         print("Numeric column types after conversion:")
#         for col in numeric_cols:
#             if col in df.columns:
#                 print(f"{col}: {df[col].dtype}")
#     else:
#         # For other datasets, detect numeric columns
#         numeric_cols = df.select_dtypes(include=[np.number]).columns
#
#     print(f"Computing statistics for numeric columns: {numeric_cols}")
#
#     # Process numeric columns
#     for col in numeric_cols:
#         if col not in df.columns:
#             print(f"Warning: Column {col} not found in dataframe")
#             continue
#
#         col_data = df[col].dropna()
#         print(f"Computing statistics for {col}, found {len(col_data)} non-null values")
#
#         if len(col_data) == 0:
#             stats_data.append([col, None, None, None, None, None, None, None, None, None])
#         else:
#             try:
#                 # Convert to numeric again to ensure calculations work
#                 col_data = pd.to_numeric(col_data, errors='coerce')
#                 col_data = col_data.dropna()  # Remove any NaN values after conversion
#
#                 mode_val = col_data.mode()
#                 mode_val = mode_val.iloc[0] if not mode_val.empty else None
#
#                 # Calculate statistics safely
#                 stats_data.append([col,
#                                    float(col_data.min()),
#                                    float(col_data.max()),
#                                    round(float(col_data.mean()), 2),
#                                    round(float(col_data.median()), 2),
#                                    round(float(col_data.std()), 2),
#                                    float(mode_val) if mode_val is not None else None,
#                                    round(float(col_data.var()), 2),
#                                    round(float(skew(col_data)), 2),
#                                    round(float(kurtosis(col_data)), 2)
#                                    ])
#                 print(f"Successfully calculated statistics for {col}")
#             except Exception as e:
#                 print(f"Error calculating statistics for {col}: {e}")
#                 stats_data.append([col, None, None, None, None, None, None, None, None, None])
#
#     # Process categorical columns
#     if dataset_type == "Adult Dataset":
#         categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
#                             'race', 'sex', 'native-country', 'income']
#     else:
#         categorical_cols = df.select_dtypes(include=['category', object]).columns
#
#     for col in categorical_cols:
#         if col in df.columns:
#             col_data = df[col].dropna()
#             print(f"Computing statistics for categorical column {col}...")
#             if len(col_data) == 0:
#                 stats_data.append([col, None, None])
#             else:
#                 stats_data.append([col,
#                                    col_data.value_counts().to_dict(),
#                                    col_data.mode().iloc[0] if not col_data.mode().empty else None,
#                                    ])
#
#     return stats_data
#
#
# def compute_correlation(df):
#     """
#     Computes the correlation matrix for numeric columns using both Pearson and Spearman methods.
#     """
#     numeric_cols = df.select_dtypes(include=[np.number]).columns
#     if len(numeric_cols) < 2:
#         return None, None
#
#     pearson_corr = df[numeric_cols].corr(method='pearson')
#     spearman_corr = df[numeric_cols].corr(method='spearman')
#
#     return pearson_corr, spearman_corr
#
#
# def generate_plot(df, column, chart_type):
#     """
#     Generates a plot based on the selected column and chart type.
#     """
#     fig, ax = plt.subplots(figsize=(6, 4))
#
#     if column not in df.columns:
#         ax.text(0.5, 0.5, f'Column "{column}" not found.', ha='center', va='center')
#         return fig
#
#     if chart_type == "Histogram":
#         if pd.api.types.is_numeric_dtype(df[column]):
#             df[column].dropna().plot(kind='hist', bins=20, ax=ax, color='skyblue', edgecolor='black')
#             ax.set_title(f'Histogram of {column}')
#             ax.set_xlabel(column)
#             ax.set_ylabel('Frequency')
#         else:
#             df[column].dropna().value_counts().plot(kind='bar', ax=ax, color='orange', edgecolor='black')
#             ax.set_title(f'Bar Chart of {column}')
#             ax.set_xlabel(column)
#             ax.set_ylabel('Count')
#
#     elif chart_type == "Boxplot":
#         if pd.api.types.is_numeric_dtype(df[column]):
#             sns.boxplot(x=df[column].dropna(), ax=ax)
#             ax.set_title(f'Boxplot of {column}')
#         else:
#             ax.text(0.5, 0.5, f'Boxplot not applicable for categorical column "{column}".', ha='center', va='center')
#
#     elif chart_type == "Bar Chart":
#         if not pd.api.types.is_numeric_dtype(df[column]):
#             df[column].dropna().value_counts().plot(kind='bar', ax=ax, color='green', edgecolor='black')
#             ax.set_title(f'Bar Chart of {column}')
#             ax.set_xlabel(column)
#             ax.set_ylabel('Count')
#         else:
#             bins = 10
#             counts, bin_edges = np.histogram(df[column].dropna(), bins=bins)
#             ax.bar(range(bins), counts, width=0.8, color='purple', edgecolor='black')
#             ax.set_title(f'Bar Chart (Binned) of {column}')
#             ax.set_xlabel(column)
#             ax.set_ylabel('Frequency')
#
#     elif chart_type == "Line Plot":
#         if pd.api.types.is_numeric_dtype(df[column]):
#             df[column].dropna().plot(kind='line', ax=ax, color='blue')
#             ax.set_title(f'Line Plot of {column}')
#             ax.set_xlabel(column)
#             ax.set_ylabel('Value')
#         else:
#             ax.text(0.5, 0.5, f'Line Plot not applicable for categorical column "{column}".', ha='center', va='center')
#
#     elif chart_type == "Pie Chart":
#         if not pd.api.types.is_numeric_dtype(df[column]):
#             df[column].dropna().value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%',
#                                                     colors=sns.color_palette("Set3", len(df[column].dropna().unique())))
#             ax.set_title(f'Pie Chart of {column}')
#             ax.set_ylabel('')
#         else:
#             ax.text(0.5, 0.5, f'Pie Chart not applicable for numeric column "{column}".', ha='center', va='center')
#
#     elif chart_type == "Heatmap":
#         # Only generate heatmap for numeric columns
#         numeric_df = df.select_dtypes(include=[np.number])
#         if numeric_df.empty:
#             ax.text(0.5, 0.5, f'Heatmap not possible. No numeric columns available.', ha='center', va='center')
#         else:
#             corr_matrix = numeric_df.corr()  # Calculate correlation matrix for numeric columns
#             sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, linewidths=0.5)
#             ax.set_title('Correlation Heatmap')
#
#     plt.tight_layout()
#     return fig
#
#
# def draw_figure(canvas, figure):
#     """
#     Draws a matplotlib figure on the provided PySimpleGUI Canvas.
#     """
#     for child in canvas.winfo_children():
#         child.destroy()
#     figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
#     figure_canvas_agg.draw()
#     figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
#     return figure_canvas_agg
#
#
# def extract_subtable(df, row_indices=None, col_indices=None, keep=False):
#     """
#     Extract a subtable from the dataframe based on the rows and columns provided by the user.
#     """
#     if df is None:
#         return None
#
#     # Create a copy to avoid modifying the original
#     result_df = df.copy()
#
#     # Add debug prints
#     print(f"Original dataframe has {len(df)} rows and {len(df.columns)} columns")
#     print(f"Keep mode: {keep}")
#
#     if row_indices is not None:
#         # Try to convert string indices to integers
#         try:
#             # Print raw input for debugging
#             print(f"Raw row indices: {row_indices}")
#
#             # Convert strings to integers if possible
#             processed_indices = []
#             for r in row_indices:
#                 if isinstance(r, str) and r.isdigit():
#                     processed_indices.append(int(r))
#                 else:
#                     processed_indices.append(r)
#
#             row_indices = processed_indices
#             print(f"Processed row indices: {row_indices}")
#
#             # Check if indices are valid
#             valid_indices = [idx for idx in row_indices if isinstance(idx, int) and 0 <= idx < len(result_df)]
#             print(f"Valid row indices: {valid_indices}")
#
#             if not valid_indices:
#                 print("No valid row indices found!")
#                 sg.popup("No valid row indices found. Please enter values between 0 and " + str(len(result_df) - 1))
#                 return None
#
#             if keep:
#                 # Keep only specified rows
#                 print(f"Keeping rows at indices: {valid_indices}")
#                 result_df = result_df.iloc[valid_indices, :]
#             else:
#                 # Remove specified rows
#                 print(f"Removing rows at indices: {valid_indices}")
#                 result_df = result_df.drop(result_df.index[valid_indices])
#
#             print(f"After row operation, dataframe has {len(result_df)} rows")
#
#         except Exception as e:
#             print(f"Error processing row indices: {e}")
#             sg.popup(f"Error processing row indices: {e}")
#             return None
#
#     if col_indices is not None:
#         # Similar processing for column indices...
#         try:
#             print(f"Raw column indices: {col_indices}")
#
#             # Process column indices
#             col_names = []
#             for c in col_indices:
#                 if isinstance(c, int) and 0 <= c < len(result_df.columns):
#                     col_names.append(result_df.columns[c])
#                 elif isinstance(c, str) and c in result_df.columns:
#                     col_names.append(c)
#
#             print(f"Valid column names: {col_names}")
#
#             if not col_names:
#                 print("No valid column indices found!")
#                 sg.popup("No valid column indices or names found.")
#                 return None
#
#             if keep:
#                 # Keep only specified columns
#                 print(f"Keeping columns: {col_names}")
#                 result_df = result_df[col_names]
#             else:
#                 # Remove specified columns
#                 print(f"Removing columns: {col_names}")
#                 result_df = result_df.drop(columns=col_names)
#
#             print(f"After column operation, dataframe has {len(result_df.columns)} columns")
#
#         except Exception as e:
#             print(f"Error processing column indices: {e}")
#             sg.popup(f"Error processing column indices: {e}")
#             return None
#
#     print(f"Final dataframe has {len(result_df)} rows and {len(result_df.columns)} columns")
#     return result_df
#
#
# def remove_columns(df, cols_to_remove):
#     """
#     Removes specified columns from the DataFrame.
#     """
#     for c in cols_to_remove:
#         if c in df.columns:
#             df = df.drop(columns=c)
#     return df
#
#
# def replace_values(df, column, old_value, new_value):
#     """
#     Replace specific value in the column with the new value.
#     Handles both regular columns and categorical columns.
#     """
#     if column not in df.columns:
#         sg.popup(f"Column '{column}' does not exist!")
#         return df
#
#     # Create a copy to avoid modifying the original DataFrame by reference
#     df = df.copy()
#
#     # Check column data type
#     if pd.api.types.is_numeric_dtype(df[column]):
#         # For numeric columns, convert values to the appropriate type
#         try:
#             old_value_numeric = float(old_value) if '.' in old_value else int(old_value)
#             new_value_numeric = float(new_value) if '.' in new_value else int(new_value)
#
#             # Replace values with consideration for floating point tolerance
#             if isinstance(old_value_numeric, float):
#                 mask = np.isclose(df[column], old_value_numeric)
#                 df.loc[mask, column] = new_value_numeric
#             else:
#                 df.loc[df[column] == old_value_numeric, column] = new_value_numeric
#
#             # Print for debugging
#             print(f"Replaced '{old_value_numeric}' with '{new_value_numeric}' in column '{column}'")
#
#         except ValueError:
#             sg.popup(f"Value conversion error for numeric column '{column}'")
#             return df
#
#     elif isinstance(df[column].dtype, pd.CategoricalDtype):
#         # For categorical columns
#         if old_value in df[column].cat.categories:
#             if new_value not in df[column].cat.categories:
#                 df[column] = df[column].cat.add_categories([new_value])
#             df[column] = df[column].replace(old_value, new_value)
#         else:
#             sg.popup(f"Old value '{old_value}' not found in categorical column '{column}'.")
#     else:
#         # For other column types (string, etc.)
#         df[column] = df[column].replace(old_value, new_value)
#
#     return df
#
#
# def replace_all_values(df, column, new_value):
#     """
#     Replace all values in the column with the new value.
#     Handles both regular columns and categorical columns.
#     """
#     if column not in df.columns:
#         sg.popup(f"Column '{column}' does not exist!")
#         return df
#
#     # For categorical columns, rename all categories to the new value
#     if isinstance(df[column].dtype, pd.CategoricalDtype):
#         # Check if new_value already exists in categories
#         if new_value not in df[column].cat.categories:
#             # Add the new value as a category
#             df[column] = df[column].cat.set_categories(list(df[column].cat.categories) + [new_value])
#         # Set all values in the column to new_value
#         df[column] = new_value
#     else:
#         # For non-categorical columns, replace all values
#         df[column] = new_value
#
#     return df
#
#
# def handle_missing_values(df, strategy):
#     """
#     Handles missing values based on the strategy:
#     - 'remove': drops rows with missing values.
#     - 'fill_mean': fills numeric columns with the mean.
#     - 'fill_median': fills numeric columns with the median.
#     - 'fill_mode': fills numeric columns with the mode.
#     """
#     if strategy == 'remove':
#         df = df.dropna()
#     else:
#         numeric_cols = df.select_dtypes(include=[np.number]).columns
#         for col in numeric_cols:
#             if strategy == 'fill_mean':
#                 df.loc[:, col] = df[col].fillna(df[col].mean())
#             elif strategy == 'fill_median':
#                 df.loc[:, col] = df[col].fillna(df[col].median())
#             elif strategy == 'fill_mode':
#                 mode_val = df[col].mode()
#                 if not mode_val.empty:
#                     df.loc[:, col] = df[col].fillna(mode_val.iloc[0])
#     return df
#
#
# def remove_duplicates(df):
#     """
#     Removes duplicate rows from the dataframe.
#     """
#     return df.drop_duplicates()
#
#
# def one_hot_encoding(df, column):
#     """
#     Perform One-Hot Encoding on the specified column.
#     """
#     if column not in df.columns:
#         raise ValueError(f"Column '{column}' not found in the dataframe")
#
#     # Create dummy variables and join with original df
#     dummies = pd.get_dummies(df[column], prefix=column, drop_first=True)
#     result_df = pd.concat([df.drop(columns=[column]), dummies], axis=1)
#     return result_df
#
#
# def binary_encoding(df, column):
#     """
#     Perform Binary Encoding on the specified column using category codes.
#     """
#     if column not in df.columns:
#         raise ValueError(f"Column '{column}' not found in the dataframe")
#
#     # Convert to category if it's not already
#     df = df.copy()
#     df[column] = df[column].astype('category')
#     df[f"{column}_encoded"] = df[column].cat.codes
#     return df
#
#
# def target_encoding(df, column, target):
#     """
#     Perform Target Encoding on the specified column using the target column.
#     """
#     if column not in df.columns or target not in df.columns:
#         raise ValueError(f"Column '{column}' or target '{target}' not found in the dataframe")
#
#     # Calculate mean target value for each category
#     df = df.copy()
#     encoding_map = df.groupby(column)[target].mean().to_dict()
#     df[f"{column}_target_encoded"] = df[column].map(encoding_map)
#     return result_df
#
#
# def scale_columns(df, cols, method='standard'):
#     """
#     Scales the specified numeric columns using StandardScaler or MinMaxScaler.
#     """
#     if df is None or df.empty:
#         raise ValueError("The dataframe is empty. Please load a valid dataset.")
#
#     numeric_cols = df.select_dtypes(include=[np.number]).columns
#     chosen_cols = [c for c in cols if c in numeric_cols]
#
#     if not chosen_cols:
#         raise ValueError("No valid numeric columns selected. Please select from: " + ", ".join(numeric_cols))
#
#     # Create a copy to avoid modifying the original data
#     df_copy = df.copy()
#
#     # Handle any NaN values before scaling
#     for col in chosen_cols:
#         df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
#
#     # Perform scaling
#     scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
#
#     try:
#         scaled_values = scaler.fit_transform(df_copy[chosen_cols])
#         df_copy.loc[:, chosen_cols] = scaled_values
#         return df_copy
#     except Exception as e:
#         raise ValueError(f"Scaling error: {str(e)}")
#
#
# def add_symbolic_column(df):
#     """
#     Adds a symbolic column 'age_category' based on the 'age' column.
#     Divides age into intervals.
#     """
#     if 'age' in df.columns and pd.api.types.is_numeric_dtype(df['age']):
#         df['age_category'] = pd.cut(
#             df['age'],
#             bins=[0, 25, 45, 65, 150],
#             labels=['Young', 'Adult', 'Senior', 'Elder'],
#             right=False
#         )
#     return df
#
#
# def logistic_regression(df):
#     """
#     Runs Logistic Regression to predict 'income' (<=50K or >50K) based on numeric features.
#     Returns the accuracy on the test set.
#     """
#     if 'income' not in df.columns:
#         return None, "No 'income' column found."
#     local_df = df.dropna(subset=['income']).copy()
#     local_df = local_df[local_df['income'].isin(['<=50K', '>50K'])]
#     encoder = LabelEncoder()
#     local_df['income_encoded'] = encoder.fit_transform(local_df['income'])
#     numeric_cols = local_df.select_dtypes(include=[np.number]).columns
#     feature_cols = [c for c in numeric_cols if c not in ['income_encoded']]
#     if not feature_cols:
#         return None, "No numeric features available for classification."
#     X = local_df[feature_cols]
#     y = local_df['income_encoded']
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     clf = LogisticRegression(max_iter=1000)
#     try:
#         clf.fit(X_train, y_train)
#     except Exception as e:
#         return None, f"Error in Logistic Regression fit: {e}"
#     accuracy = clf.score(X_test, y_test)
#     return accuracy, None
#
#
# def kmeans_clustering(df, n_clusters=2):
#     """
#     Runs K-Means clustering on numeric columns.
#     Returns the cluster labels.
#     """
#     numeric_cols = df.select_dtypes(include=[np.number]).columns
#     if len(numeric_cols) < 1:
#         return None, "No numeric columns available for clustering."
#     local_df = df.dropna(subset=numeric_cols).copy()
#     X = local_df[numeric_cols]
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#     try:
#         labels = kmeans.fit_predict(X)
#     except Exception as e:
#         return None, f"Error in KMeans: {e}"
#     return labels, None
#
#
# # Convert dataframe to table data
# def df_to_table_data(df, max_rows=10):
#     """
#     Convert a dataframe to a format suitable for sg.Table
#     Returns headers and data rows
#     """
#     if df is None or df.empty:
#         return [], []
#
#     headers = list(df.columns)
#
#     # Get the first max_rows rows
#     display_df = df.head(max_rows)
#
#     # Convert to list of lists
#     data = display_df.values.tolist()
#
#     # Convert any non-string values to strings for display
#     for i, row in enumerate(data):
#         for j, val in enumerate(row):
#             if pd.isna(val):
#                 data[i][j] = 'NaN'
#             elif isinstance(val, (dict, list)):
#                 data[i][j] = str(val)
#             else:
#                 data[i][j] = str(val)
#
#     return headers, data
#
#
# # ------------------ Layout Definition ------------------ #
#
# # Tab 0: Logo & Creators Info
# tab0_layout = [
#     [sg.Text("", size=(1, 1))],
#     [sg.Column([
#         [sg.Text("Subject: Data Warehousing", font=("Helvetica", 20, "bold"), justification='center', pad=(0, 20))],
#         [sg.Image(filename="assets/logo.png", key="-LOGO-", size=(250, 250))],
#         [sg.Text("Created by:", font=("Helvetica", 16), justification='center', pad=(0, 10))],
#         [sg.Text("Dawid Olko", font=("Helvetica", 14), justification='center')],
#         [sg.Text("Piotr Smoła", font=("Helvetica", 14), justification='center')],
#         [sg.Button("Go to data", key="-ENTER-", font=("Helvetica", 14), size=(20, 1), pad=((0, 0), (20, 0)),
#                    button_color=('white', '#4CAF50'))]
#     ], justification='center', element_justification='center')],
#     [sg.Text("", size=(1, 1))],
# ]
#
# # Tab 1: Data & Statistics
# tab1_layout = [
#     [sg.Frame("Dataset Selection", [
#         [sg.Combo(["Adult Dataset", "Kidney Disease Dataset"], default_value="Adult Dataset", key="-SELECT_FILE-",
#                   size=(30, 1)),
#          sg.Button("Load Data", key="-LOAD-", button_color=('white', '#4CAF50'))]
#     ])],
#
#     [sg.Frame("Dataset Information", [
#         [sg.Table(values=[], headings=["Column", "Data Type", "Non-Null Count", "Memory Usage"],
#                   key="-DATA_INFO_TABLE-", auto_size_columns=True, justification='center',
#                   num_rows=6, expand_x=True)]
#     ])],
#
#     [sg.Button("Compute Stats", key="-STATS-", button_color=('white', '#2196F3')),
#      sg.Button("Correlation", key="-CORR-", button_color=('white', '#2196F3'))],
#
#     [sg.Frame("Numeric Columns Statistics", [
#         [sg.Table(values=[],
#                   headings=["Column", "Min", "Max", "Mean", "Median", "Std", "Mode", "Variance", "Skewness",
#                             "Kurtosis"],
#                   key="-NUMERIC_STATS-", auto_size_columns=True, justification='center',
#                   expand_x=True, num_rows=6)]
#     ])],
#
#     [sg.Frame("Categorical Columns Statistics", [
#         [sg.Table(values=[], headings=["Column", "Value Counts", "Mode"],
#                   key="-CATEGORICAL_STATS-", auto_size_columns=True, justification='center',
#                   expand_x=True, num_rows=6)]
#     ])],
#
#     [sg.Frame("Correlation Results", [
#         [sg.Table(values=[], headings=[], key="-CORR_TABLE-", auto_size_columns=True, num_rows=10,
#                   justification='center', expand_x=True, expand_y=True)]
#     ])]
# ]
#
# # Tab 2: Extract Subtable
# tab2_layout = [
#     [sg.Frame("Subtable Extraction", [
#         [sg.Radio("Remove Rows/Columns", "EXTRACTION_TYPE", default=True, key="-REMOVE_EXTRACT-"),
#          sg.Radio("Keep Rows/Columns", "EXTRACTION_TYPE", key="-KEEP_EXTRACT-")],
#
#         [sg.Text("Rows (comma separated indices or names):", size=(30, 1)),
#          sg.InputText(key="-ROW_INPUT-", size=(30, 1))],
#
#         [sg.Text("Columns (comma separated indices or names):", size=(30, 1)),
#          sg.InputText(key="-COL_INPUT-", size=(30, 1))],
#
#         [sg.Button("Extract Subtable", key="-EXTRACT_BTN-", button_color=('white', '#2196F3'))]
#     ])],
#
#     [sg.Frame("Subtable Preview", [
#         [sg.Table(values=[], headings=[], key="-EXTRACT_TABLE-", auto_size_columns=True,
#                   num_rows=10, justification='center', expand_x=True)]
#     ])],
#
#     [sg.Frame("Value Replacement", [
#         # Dokończenie kodu w miejscu przerwania - kontynuacja Tab 2 layout
#         [sg.Frame("Value Replacement", [
#             [sg.Text("Column:"), sg.Combo([], key="-REPLACE_COLUMN-", size=(20, 1))],
#             [sg.Radio("Replace Specific Value", "REPLACE_TYPE", default=True, key="-REPLACE_SPECIFIC-"),
#              sg.Radio("Replace All Values", "REPLACE_TYPE", key="-REPLACE_ALL-")],
#             [sg.Text("Old Value:"), sg.InputText(key="-OLD_VALUE-", size=(15, 1)),
#              sg.Text("New Value:"), sg.InputText(key="-NEW_VALUE-", size=(15, 1))],
#             [sg.Button("Replace", key="-REPLACE_BTN-", button_color=('white', '#FF9800'))]
#         ])],
#
#         [sg.Button("Apply Changes", key="-APPLY_EXTRACT-", button_color=('white', '#4CAF50')),
#          sg.Button("Reset Changes", key="-RESET_EXTRACT-", button_color=('white', '#F44336'))]
#     ])]
# ]
#
# # Tab 3: Data Cleaning
# tab3_layout = [
#     [sg.Frame("Missing Values", [
#         [sg.Text("Strategy for handling missing values:")],
#         [sg.Radio("Remove rows with missing values", "MISSING", default=True, key="-MISSING_REMOVE-"),
#          sg.Radio("Fill with mean", "MISSING", key="-MISSING_MEAN-")],
#         [sg.Radio("Fill with median", "MISSING", key="-MISSING_MEDIAN-"),
#          sg.Radio("Fill with mode", "MISSING", key="-MISSING_MODE-")],
#         [sg.Button("Handle Missing Values", key="-HANDLE_MISSING-", button_color=('white', '#2196F3'))]
#     ])],
#
#     [sg.Frame("Duplicate Rows", [
#         [sg.Text("Remove duplicate rows from the dataset:")],
#         [sg.Button("Remove Duplicates", key="-REMOVE_DUPLICATES-", button_color=('white', '#FF9800'))]
#     ])],
#
#     [sg.Frame("Feature Engineering", [
#         [sg.Text("Add symbolic column 'age_category' based on 'age':")],
#         [sg.Button("Add Age Category", key="-ADD_AGE_CAT-", button_color=('white', '#9C27B0'))]
#     ])],
#
#     [sg.Frame("Scaling", [
#         [sg.Text("Select columns to scale:"),
#          sg.Listbox(values=[], select_mode='multiple', key="-SCALE_COLS-", size=(30, 5))],
#         [sg.Radio("Standard Scaling (zero mean, unit variance)", "SCALING", default=True, key="-STD_SCALING-"),
#          sg.Radio("MinMax Scaling (0 to 1 range)", "SCALING", key="-MINMAX_SCALING-")],
#         [sg.Button("Scale Data", key="-SCALE_BTN-", button_color=('white', '#00BCD4'))]
#     ])],
#
#     [sg.Frame("Preview Cleaned Data", [
#         [sg.Table(values=[], headings=[], key="-CLEAN_TABLE-", auto_size_columns=True,
#                   num_rows=10, justification='center', expand_x=True)]
#     ])],
#
#     [sg.Button("Apply Changes", key="-APPLY_CLEAN-", button_color=('white', '#4CAF50')),
#      sg.Button("Reset Changes", key="-RESET_CLEAN-", button_color=('white', '#F44336'))]
# ]
#
# # Tab 4: Data Encoding
# tab4_layout = [
#     [sg.Frame("Encoding Techniques", [
#         [sg.Text("Select column to encode:"),
#          sg.Combo([], key="-ENCODE_COLUMN-", size=(30, 1))],
#         [sg.Radio("One-Hot Encoding", "ENCODING", default=True, key="-ONE_HOT-"),
#          sg.Radio("Binary Encoding", "ENCODING", key="-BINARY-"),
#          sg.Radio("Target Encoding", "ENCODING", key="-TARGET-")],
#         [sg.Text("Target column (for Target Encoding):"),
#          sg.Combo([], key="-TARGET_COLUMN-", size=(25, 1), disabled=True)],
#         [sg.Button("Apply Encoding", key="-ENCODE_BTN-", button_color=('white', '#2196F3'))]
#     ])],
#
#     [sg.Frame("Preview Encoded Data", [
#         [sg.Table(values=[], headings=[], key="-ENCODE_TABLE-", auto_size_columns=True,
#                   num_rows=10, justification='center', expand_x=True)]
#     ])],
#
#     [sg.Button("Apply Changes", key="-APPLY_ENCODE-", button_color=('white', '#4CAF50')),
#      sg.Button("Reset Changes", key="-RESET_ENCODE-", button_color=('white', '#F44336'))]
# ]
#
# # Tab 5: Visualization
# tab5_layout = [
#     [sg.Frame("Visualization Options", [
#         [sg.Text("Select Column:"),
#          sg.Combo([], key="-VIS_COLUMN-", size=(20, 1)),
#          sg.Text("Chart Type:"),
#          sg.Combo(["Histogram", "Boxplot", "Bar Chart", "Line Plot", "Pie Chart", "Heatmap"],
#                   default_value="Histogram", key="-CHART_TYPE-", size=(15, 1))],
#         [sg.Button("Generate Plot", key="-PLOT-", button_color=('white', '#2196F3'))]
#     ])],
#
#     [sg.Frame("Plot", [
#         [sg.Canvas(key="-CANVAS-", size=(500, 400))]
#     ])]
# ]
#
# # Tab 6: Machine Learning
# tab6_layout = [
#     [sg.Frame("Classification", [
#         [sg.Text("Run logistic regression to predict income class:")],
#         [sg.Button("Run Logistic Regression", key="-LOGISTIC-", button_color=('white', '#FF9800'))],
#         [sg.Text("Accuracy: ", size=(10, 1)), sg.Text("", key="-ACCURACY-", size=(30, 1))]
#     ])],
#
#     [sg.Frame("Clustering", [
#         [sg.Text("Run K-Means clustering on numeric columns:")],
#         [sg.Text("Number of clusters:"),
#          sg.Spin([i for i in range(2, 11)], initial_value=2, key="-N_CLUSTERS-", size=(5, 1))],
#         [sg.Button("Run K-Means", key="-KMEANS-", button_color=('white', '#9C27B0'))],
#         [sg.Text("Results:", size=(10, 1))],
#         [sg.Multiline(size=(60, 5), key="-CLUSTER_RESULTS-", disabled=True)]
#     ])]
# ]
#
# # Define the main layout with tabs
# layout = [
#     [sg.TabGroup([
#         [sg.Tab("Home", tab0_layout, key="-TAB0-"),
#          sg.Tab("Data & Statistics", tab1_layout, key="-TAB1-", disabled=True),
#          sg.Tab("Extract & Replace", tab2_layout, key="-TAB2-", disabled=True),
#          sg.Tab("Data Cleaning", tab3_layout, key="-TAB3-", disabled=True),
#          sg.Tab("Encoding", tab4_layout, key="-TAB4-", disabled=True),
#          sg.Tab("Visualization", tab5_layout, key="-TAB5-", disabled=True),
#          sg.Tab("Machine Learning", tab6_layout, key="-TAB6-", disabled=True)]
#     ], key="-TABGROUP-", expand_x=True, expand_y=True)]
# ]
#
# # ------------------ Main Application ------------------ #
#
#
# def main():
#     # Create the window
#     window = sg.Window("Data Analysis App", layout, resizable=True, size=(1000, 700))
#
#     # Initialize variables
#     df = None
#     original_df = None
#     extraction_df = None
#     cleaned_df = None
#     encoded_df = None
#     figure_canvas_agg = None
#
#     # Event loop
#     while True:
#         event, values = window.read()
#
#         # Close the window
#         if event == sg.WINDOW_CLOSED:
#             break
#
#         # --- Tab 0: Home ---
#         if event == "-ENTER-":
#             # Enable all other tabs when entering the app
#             for i in range(1, 7):
#                 window[f"-TAB{i}-"].update(disabled=False)
#             window["-TABGROUP-"].set_focus(1)  # Switch to the first tab
#
#         # --- Tab 1: Data & Statistics ---
#         elif event == "-LOAD-":
#             try:
#                 file_name = values["-SELECT_FILE-"]
#                 df = load_dataset(file_name)
#                 original_df = df.copy()
#
#                 # Update data info table
#                 info_data = []
#                 for col in df.columns:
#                     col_type = str(df[col].dtype)
#                     non_null = df[col].count()
#                     memory = df[col].memory_usage(deep=True) / 1024  # KB
#                     info_data.append([col, col_type, non_null, f"{memory:.2f} KB"])
#
#                 window["-DATA_INFO_TABLE-"].update(values=info_data)
#
#                 # Fill column options for other tabs
#                 column_list = list(df.columns)
#                 window["-REPLACE_COLUMN-"].update(values=column_list)
#                 window["-ENCODE_COLUMN-"].update(values=column_list)
#                 window["-TARGET_COLUMN-"].update(values=column_list)
#                 window["-VIS_COLUMN-"].update(values=column_list)
#
#                 # Update scaling columns listbox
#                 numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
#                 window["-SCALE_COLS-"].update(values=numeric_cols)
#
#                 sg.popup(f"Dataset loaded successfully: {len(df)} rows, {len(df.columns)} columns")
#
#             except Exception as e:
#                 sg.popup_error(f"Error loading dataset: {str(e)}")
#
#         elif event == "-STATS-":
#             if df is not None:
#                 try:
#                     dataset_type = values["-SELECT_FILE-"]
#                     stats = compute_statistics(df, dataset_type)
#
#                     # Separate numeric and categorical stats
#                     numeric_stats = []
#                     cat_stats = []
#
#                     for stat in stats:
#                         if len(stat) > 3:  # Numeric stats have more columns
#                             numeric_stats.append(stat)
#                         else:
#                             cat_stats.append(stat)
#
#                     window["-NUMERIC_STATS-"].update(values=numeric_stats)
#                     window["-CATEGORICAL_STATS-"].update(values=cat_stats)
#
#                 except Exception as e:
#                     sg.popup_error(f"Error computing statistics: {str(e)}")
#             else:
#                 sg.popup("Please load a dataset first.")
#
#         elif event == "-CORR-":
#             if df is not None:
#                 try:
#                     pearson_corr, spearman_corr = compute_correlation(df)
#                     if pearson_corr is not None:
#                         # Create a formatted table of correlation values
#                         headers = ["Column"] + list(pearson_corr.columns)
#                         data = []
#
#                         for col in pearson_corr.index:
#                             row = [col]
#                             for other_col in pearson_corr.columns:
#                                 row.append(f"{pearson_corr.loc[col, other_col]:.3f}")
#                             data.append(row)
#
#                         window["-CORR_TABLE-"].update(values=data, headings=headers)
#                     else:
#                         sg.popup("Not enough numeric columns for correlation analysis.")
#                 except Exception as e:
#                     sg.popup_error(f"Error computing correlation: {str(e)}")
#             else:
#                 sg.popup("Please load a dataset first.")
#
#         # --- Tab 2: Extract & Replace ---
#         elif event == "-EXTRACT_BTN-":
#             if df is not None:
#                 try:
#                     # Get row and column inputs
#                     row_input = values["-ROW_INPUT-"].strip()
#                     col_input = values["-COL_INPUT-"].strip()
#
#                     row_indices = None
#                     col_indices = None
#
#                     if row_input:
#                         row_indices = [r.strip() for r in row_input.split(',')]
#                         # Try to convert numeric strings to integers
#                         row_indices = [int(r) if r.isdigit() else r for r in row_indices]
#
#                     if col_input:
#                         col_indices = [c.strip() for c in col_input.split(',')]
#                         # Try to convert numeric strings to integers
#                         col_indices = [int(c) if c.isdigit() else c for c in col_indices]
#
#                     keep = values["-KEEP_EXTRACT-"]
#
#                     extraction_df = extract_subtable(df, row_indices, col_indices, keep)
#
#                     if extraction_df is not None:
#                         # Update the table with the extracted data
#                         headers, data = df_to_table_data(extraction_df)
#                         window["-EXTRACT_TABLE-"].update(values=data, headings=headers)
#                         sg.popup(
#                             f"Extraction successful: {len(extraction_df)} rows, {len(extraction_df.columns)} columns")
#
#                 except Exception as e:
#                     sg.popup_error(f"Error extracting subtable: {str(e)}")
#             else:
#                 sg.popup("Please load a dataset first.")
#
#         elif event == "-REPLACE_BTN-":
#             if df is not None and extraction_df is not None:
#                 try:
#                     column = values["-REPLACE_COLUMN-"]
#                     if not column:
#                         sg.popup("Please select a column for replacement.")
#                         continue
#
#                     new_value = values["-NEW_VALUE-"]
#
#                     if values["-REPLACE_SPECIFIC-"]:
#                         old_value = values["-OLD_VALUE-"]
#                         if not old_value:
#                             sg.popup("Please enter the old value to replace.")
#                             continue
#
#                         extraction_df = replace_values(extraction_df, column, old_value, new_value)
#                     else:  # Replace all values
#                         extraction_df = replace_all_values(extraction_df, column, new_value)
#
#                     # Update the table with the modified data
#                     headers, data = df_to_table_data(extraction_df)
#                     window["-EXTRACT_TABLE-"].update(values=data, headings=headers)
#                     sg.popup("Replacement completed!")
#
#                 except Exception as e:
#                     sg.popup_error(f"Error replacing values: {str(e)}")
#             else:
#                 sg.popup("Please extract a subtable first.")
#
#         elif event == "-APPLY_EXTRACT-":
#             if extraction_df is not None:
#                 df = extraction_df.copy()
#                 sg.popup("Changes applied to main dataset!")
#             else:
#                 sg.popup("No changes to apply.")
#
#         elif event == "-RESET_EXTRACT-":
#             if original_df is not None:
#                 extraction_df = None
#                 window["-EXTRACT_TABLE-"].update(values=[], headings=[])
#                 sg.popup("Extraction reset!")
#             else:
#                 sg.popup("No original data to reset to.")
#
#         # --- Tab 3: Data Cleaning ---
#         elif event == "-HANDLE_MISSING-":
#             if df is not None:
#                 try:
#                     strategy = None
#                     if values["-MISSING_REMOVE-"]:
#                         strategy = "remove"
#                     elif values["-MISSING_MEAN-"]:
#                         strategy = "fill_mean"
#                     elif values["-MISSING_MEDIAN-"]:
#                         strategy = "fill_median"
#                     elif values["-MISSING_MODE-"]:
#                         strategy = "fill_mode"
#
#                     cleaned_df = handle_missing_values(df.copy(), strategy)
#
#                     # Update the preview table
#                     headers, data = df_to_table_data(cleaned_df)
#                     window["-CLEAN_TABLE-"].update(values=data, headings=headers)
#
#                     # Show message with counts
#                     orig_null = df.isna().sum().sum()
#                     clean_null = cleaned_df.isna().sum().sum()
#                     sg.popup(f"Missing values handled: {orig_null - clean_null} values processed.\n"
#                              f"Original NaN count: {orig_null}, New NaN count: {clean_null}")
#
#                 except Exception as e:
#                     sg.popup_error(f"Error handling missing values: {str(e)}")
#             else:
#                 sg.popup("Please load a dataset first.")
#
#         elif event == "-REMOVE_DUPLICATES-":
#             if df is not None:
#                 try:
#                     if cleaned_df is None:
#                         cleaned_df = df.copy()
#
#                     original_rows = len(cleaned_df)
#                     cleaned_df = remove_duplicates(cleaned_df)
#                     removed_rows = original_rows - len(cleaned_df)
#
#                     # Update the preview table
#                     headers, data = df_to_table_data(cleaned_df)
#                     window["-CLEAN_TABLE-"].update(values=data, headings=headers)
#
#                     sg.popup(f"Duplicate rows removed: {removed_rows} rows")
#
#                 except Exception as e:
#                     sg.popup_error(f"Error removing duplicates: {str(e)}")
#             else:
#                 sg.popup("Please load a dataset first.")
#
#         elif event == "-ADD_AGE_CAT-":
#             if df is not None:
#                 try:
#                     if cleaned_df is None:
#                         cleaned_df = df.copy()
#
#                     if 'age' in cleaned_df.columns:
#                         cleaned_df = add_symbolic_column(cleaned_df)
#
#                         # Update the preview table
#                         headers, data = df_to_table_data(cleaned_df)
#                         window["-CLEAN_TABLE-"].update(values=data, headings=headers)
#
#                         sg.popup("Age category column added!")
#                     else:
#                         sg.popup("No 'age' column found in the dataset.")
#
#                 except Exception as e:
#                     sg.popup_error(f"Error adding age category: {str(e)}")
#             else:
#                 sg.popup("Please load a dataset first.")
#
#         elif event == "-SCALE_BTN-":
#             if df is not None:
#                 try:
#                     selected_cols = values["-SCALE_COLS-"]
#                     if not selected_cols:
#                         sg.popup("Please select columns to scale.")
#                         continue
#
#                     method = "standard" if values["-STD_SCALING-"] else "minmax"
#
#                     if cleaned_df is None:
#                         cleaned_df = df.copy()
#
#                     cleaned_df = scale_columns(cleaned_df, selected_cols, method)
#
#                     # Update the preview table
#                     headers, data = df_to_table_data(cleaned_df)
#                     window["-CLEAN_TABLE-"].update(values=data, headings=headers)
#
#                     sg.popup(f"Scaling applied to {len(selected_cols)} columns using {method} scaling!")
#
#                 except Exception as e:
#                     sg.popup_error(f"Error scaling data: {str(e)}")
#             else:
#                 sg.popup("Please load a dataset first.")
#
#         elif event == "-APPLY_CLEAN-":
#             if cleaned_df is not None:
#                 df = cleaned_df.copy()
#                 sg.popup("Cleaning changes applied to main dataset!")
#             else:
#                 sg.popup("No cleaning changes to apply.")
#
#         elif event == "-RESET_CLEAN-":
#             if original_df is not None:
#                 cleaned_df = None
#                 window["-CLEAN_TABLE-"].update(values=[], headings=[])
#                 sg.popup("Cleaning reset!")
#             else:
#                 sg.popup("No original data to reset to.")
#
#         # --- Tab 4: Data Encoding ---
#         elif event == "-TARGET-":  # Enable target column when Target Encoding is selected
#             window["-TARGET_COLUMN-"].update(disabled=False)
#
#         elif event in ["-ONE_HOT-", "-BINARY-"]:  # Disable target column for other encoding methods
#             window["-TARGET_COLUMN-"].update(disabled=True)
#
#         elif event == "-ENCODE_BTN-":
#             if df is not None:
#                 try:
#                     column = values["-ENCODE_COLUMN-"]
#                     if not column:
#                         sg.popup("Please select a column to encode.")
#                         continue
#
#                     # Create encoded DataFrame
#                     if encoded_df is None:
#                         encoded_df = df.copy()
#
#                     # Apply the selected encoding method
#                     if values["-ONE_HOT-"]:
#                         encoded_df = one_hot_encoding(encoded_df, column)
#                         encoding_type = "One-Hot"
#                     elif values["-BINARY-"]:
#                         encoded_df = binary_encoding(encoded_df, column)
#                         encoding_type = "Binary"
#                     elif values["-TARGET-"]:
#                         target = values["-TARGET_COLUMN-"]
#                         if not target:
#                             sg.popup("Please select a target column for Target Encoding.")
#                             continue
#                         encoded_df = target_encoding(encoded_df, column, target)
#                         encoding_type = "Target"
#
#                     # Update the preview table
#                     headers, data = df_to_table_data(encoded_df)
#                     window["-ENCODE_TABLE-"].update(values=data, headings=headers)
#
#                     sg.popup(f"{encoding_type} Encoding applied to column '{column}'!")
#
#                 except Exception as e:
#                     sg.popup_error(f"Error encoding data: {str(e)}")
#             else:
#                 sg.popup("Please load a dataset first.")
#
#         elif event == "-APPLY_ENCODE-":
#             if encoded_df is not None:
#                 df = encoded_df.copy()
#                 sg.popup("Encoding changes applied to main dataset!")
#             else:
#                 sg.popup("No encoding changes to apply.")
#
#         elif event == "-RESET_ENCODE-":
#             if original_df is not None:
#                 encoded_df = None
#                 window["-ENCODE_TABLE-"].update(values=[], headings=[])
#                 sg.popup("Encoding reset!")
#             else:
#                 sg.popup("No original data to reset to.")
#
#         # --- Tab 5: Visualization ---
#         elif event == "-PLOT-":
#             if df is not None:
#                 try:
#                     column = values["-VIS_COLUMN-"]
#                     chart_type = values["-CHART_TYPE-"]
#
#                     # Generate the plot
#                     fig = generate_plot(df, column, chart_type)
#
#                     # Draw the figure on the canvas
#                     if figure_canvas_agg:
#                         figure_canvas_agg.get_tk_widget().forget()
#                     figure_canvas_agg = draw_figure(window["-CANVAS-"].TKCanvas, fig)
#
#                 except Exception as e:
#                     sg.popup_error(f"Error generating plot: {str(e)}")
#             else:
#                 sg.popup("Please load a dataset first.")
#
#         # --- Tab 6: Machine Learning ---
#         elif event == "-LOGISTIC-":
#             if df is not None:
#                 try:
#                     accuracy, error = logistic_regression(df)
#                     if accuracy is not None:
#                         window["-ACCURACY-"].update(f"{accuracy:.4f} ({accuracy * 100:.2f}%)")
#                     else:
#                         window["-ACCURACY-"].update(f"Error: {error}")
#
#                 except Exception as e:
#                     sg.popup_error(f"Error running logistic regression: {str(e)}")
#             else:
#                 sg.popup("Please load a dataset first.")
#
#         elif event == "-KMEANS-":
#             if df is not None:
#                 try:
#                     n_clusters = int(values["-N_CLUSTERS-"])
#                     labels, error = kmeans_clustering(df, n_clusters)
#
#                     if labels is not None:
#                         # Count samples in each cluster
#                         cluster_counts = {i: sum(labels == i) for i in range(n_clusters)}
#                         result_text = "K-Means Clustering Results:\n\n"
#                         for cluster, count in cluster_counts.items():
#                             result_text += f"Cluster {cluster}: {count} samples ({count / len(labels) * 100:.1f}%)\n"
#
#                         window["-CLUSTER_RESULTS-"].update(result_text)
#                     else:
#                         window["-CLUSTER_RESULTS-"].update(f"Error: {error}")
#
#                 except Exception as e:
#                     sg.popup_error(f"Error running K-Means clustering: {str(e)}")
#             else:
#                 sg.popup("Please load a dataset first.")
#
#     # Close the window
#     window.close()
#
#
# if __name__ == "__main__":
#     main()