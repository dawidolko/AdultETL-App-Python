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
import os
import tkinter as tk
from tkinter import filedialog

# ------------------ Global GUI Settings ------------------ #

sg.set_options(font=("Helvetica", 12))
df = None
original_df = None  # Dodane do przechowywania oryginalnej wersji danych
figure_canvas_agg = None


# ------------------ Helper Functions ------------------ #

def browse_for_csv_file():
    file_path = filedialog.askopenfilename(
        title="Wybierz plik CSV",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    return file_path if file_path else None


def load_dataset(file_path=None, is_predefined=True):
    global original_df

    if is_predefined:
        if file_path == "Adult Dataset":
            col_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                         'marital-status', 'occupation', 'relationship', 'race',
                         'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                         'native-country', 'income']
            try:
                df = pd.read_csv('../database/adult/adult.data', header=None, names=col_names, skipinitialspace=True)
            except Exception as e:
                try:
                    df = pd.read_csv('adult.data', header=None, names=col_names, skipinitialspace=True)
                except Exception as e2:
                    raise ValueError(f"Failed to load Adult Dataset: {e2}")

            df.replace('?', np.nan, inplace=True)

            numeric_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
            for col in numeric_cols:
                print(f"Converting {col} to numeric during dataset load...")
                df[col] = pd.to_numeric(df[col], errors='coerce')

            print("Column data types after loading Adult Dataset:")
            print(df.dtypes)

            categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                                'race', 'sex', 'native-country', 'income']
            for col in categorical_cols:
                df[col] = df[col].astype('category')

        elif file_path == "Kidney Disease Dataset":
            try:
                df = pd.read_csv('../database/chronic/kidney_disease.csv')
            except Exception as e:
                try:
                    df = pd.read_csv('kidney_disease.csv')
                except Exception as e2:
                    raise ValueError(f"Error loading kidney disease dataset: {e2}")

            df.replace('?', np.nan, inplace=True)

            categorical_cols = ['fbs', 'restecg', 'exang', 'slope', 'thal', 'classification']
            for col in categorical_cols:
                if col in df.columns:
                    df[col] = df[col].astype('category')


    else:
        try:
            with open(file_path, 'r') as f:
                first_lines = [next(f) for _ in range(5)]

            first_row = first_lines[0].strip().split(',')
            second_row = first_lines[1].strip().split(',') if len(first_lines) > 1 else []

            has_header = True

            first_row_numeric = [is_float(val.strip()) for val in first_row]

            if all(first_row_numeric) and len(second_row) > 0:
                has_header = False
                print("Detected: First row contains only numbers - treating as data")

            if has_header and any(first_row_numeric) and not all(first_row_numeric) and len(second_row) > 0:
                second_row_numeric = [is_float(val.strip()) for val in second_row]

                if any(second_row_numeric) and not all(second_row_numeric):
                    if sum(first_row_numeric) / len(first_row_numeric) > 0.5:
                        has_header = False
                        print("Detected: First row has mixed types similar to data rows - treating as data")

            common_header_terms = ["id", "name", "value", "date", "time", "category", "type", "description",
                                   "price", "cost", "age", "title", "address", "email", "phone"]

            header_term_matches = sum(
                1 for val in first_row if any(term in val.lower() for term in common_header_terms))

            if header_term_matches > 0 and header_term_matches >= len(
                    first_row) / 3:
                has_header = True
                print(
                    f"Detected: First row contains likely column names ({header_term_matches} matches) - treating as header")

            if has_header:
                df = pd.read_csv(file_path, skipinitialspace=True)
                print("Loading CSV with header")
            else:
                df = pd.read_csv(file_path, header=None, skipinitialspace=True)
                print("Loading CSV without header")

                import string
                alphabet = list(string.ascii_lowercase)
                column_names = []
                for i in range(len(df.columns)):
                    if i < 26:
                        column_names.append(f"col_{alphabet[i]}")
                    else:
                        first_letter = alphabet[(i // 26) - 1]
                        second_letter = alphabet[i % 26]
                        column_names.append(f"col_{first_letter}{second_letter}")
                df.columns = column_names

            # Zastąp wartości "?" wartościami NaN
            df.replace('?', np.nan, inplace=True)

            for col in df.columns:
                try:
                    numeric_values = pd.to_numeric(df[col], errors='coerce')
                    if numeric_values.notna().sum() >= len(df[col]) * 0.7:
                        df[col] = numeric_values
                except:
                    pass

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise ValueError(f"Error loading custom CSV file: {e}")

        # Zachowaj oryginalną wersję DataFrame
    original_df = df.copy(deep=True)
    return df


def is_float(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False


def save_dataframe_to_csv(df, include_index=False, include_header=True, index_label="id"):
    if df is None or df.empty:
        sg.popup("No data to save! Please load or process data first.")
        return False
    try:
        file_path = sg.popup_get_file('Save data as CSV file', save_as=True,
                                      file_types=(("CSV Files", "*.csv"),),
                                      default_extension=".csv")

        if not file_path:
            return False

        if not file_path.lower().endswith('.csv'):
            file_path += '.csv'

        df.to_csv(file_path, index=include_index, header=include_header,
                  index_label=index_label if include_index else None)

        sg.popup(f"Data successfully saved to {file_path}")
        return True

    except Exception as e:
        sg.popup(f"Error saving data: {e}")
        return False


def create_save_options_window(title="Save Options"):
    save_layout = [
        [sg.Checkbox("Include Row Indices", default=False, key=f"-SAVE_INCLUDE_INDEX-{title}")],
        [sg.Checkbox("Include Column Headers", default=True, key=f"-SAVE_INCLUDE_HEADER-{title}")],
        [sg.Button("Save", key="-CONFIRM_SAVE-"), sg.Button("Cancel", key="-CANCEL_SAVE-")]
    ]

    return sg.Window(title, save_layout, modal=True, finalize=True)


def restore_original_data():
    global df, original_df

    if original_df is None:
        sg.popup("No original data available to restore!")
        return None

    df = original_df.copy(deep=True)
    return df


def save_plot_as_image(fig, default_filename="plot.png"):
    try:
        file_path = sg.popup_get_file('Save plot as image', save_as=True,
                                      file_types=(("PNG Files", "*.png"),
                                                  ("JPEG Files", "*.jpg"),
                                                  ("PDF Files", "*.pdf"),
                                                  ("SVG Files", "*.svg"),
                                                  ("All Files", "*.*")),
                                      default_extension=".png",
                                      default_path=default_filename)

        if not file_path:  # User cancelled
            return False

        if not any(file_path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.pdf', '.svg']):
            file_path += '.png'

        fig.savefig(file_path, dpi=300, bbox_inches='tight')
        return True

    except Exception as e:
        sg.popup(f"Error saving plot: {e}")
        return False


def compute_statistics(df, dataset_type):
    stats_data = []

    if dataset_type == "Adult Dataset":
        numeric_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

        for col in numeric_cols:
            if col in df.columns:
                print(f"Converting {col} to numeric for stats calculation...")
                df = df.copy()
                df[col] = pd.to_numeric(df[col], errors='coerce')

        print("Numeric column types after conversion:")
        for col in numeric_cols:
            if col in df.columns:
                print(f"{col}: {df[col].dtype}")
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns

    print(f"Computing statistics for numeric columns: {numeric_cols}")

    for col in numeric_cols:
        if col not in df.columns:
            print(f"Warning: Column {col} not found in dataframe")
            continue

        col_data = df[col].dropna()
        print(f"Computing statistics for {col}, found {len(col_data)} non-null values")

        if len(col_data) == 0:
            stats_data.append([col, None, None, None, None, None, None, None, None, None])
        else:
            try:
                col_data = pd.to_numeric(col_data, errors='coerce')
                col_data = col_data.dropna()

                mode_val = col_data.mode()
                mode_val = mode_val.iloc[0] if not mode_val.empty else None

                stats_data.append([col,
                                   float(col_data.min()),
                                   float(col_data.max()),
                                   round(float(col_data.mean()), 2),
                                   round(float(col_data.median()), 2),
                                   round(float(col_data.std()), 2),
                                   float(mode_val) if mode_val is not None else None,
                                   round(float(col_data.var()), 2),
                                   round(float(skew(col_data)), 2),
                                   round(float(kurtosis(col_data)), 2)
                                   ])
                print(f"Successfully calculated statistics for {col}")
            except Exception as e:
                print(f"Error calculating statistics for {col}: {e}")
                stats_data.append([col, None, None, None, None, None, None, None, None, None])

    if dataset_type == "Adult Dataset":
        categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship',
                            'race', 'sex', 'native-country', 'income']
    else:
        categorical_cols = df.select_dtypes(include=['category', object]).columns

    for col in categorical_cols:
        if col in df.columns:
            col_data = df[col].dropna()
            print(f"Computing statistics for categorical column {col}...")
            if len(col_data) == 0:
                stats_data.append([col, None, None])
            else:
                stats_data.append([col,
                                   col_data.value_counts().to_dict(),
                                   col_data.mode().iloc[0] if not col_data.mode().empty else None,
                                   ])

    return stats_data


def compute_correlation(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return None, None

    pearson_corr = df[numeric_cols].corr(method='pearson')
    spearman_corr = df[numeric_cols].corr(method='spearman')

    return pearson_corr, spearman_corr


def generate_plot(df, column, chart_type):
    fig, ax = plt.subplots(figsize=(8, 6))

    if column not in df.columns:
        ax.text(0.5, 0.5, f'Column "{column}" not found.', ha='center', va='center')
        return fig

    plt.subplots_adjust(top=0.85)

    if chart_type == "Histogram":
        if pd.api.types.is_numeric_dtype(df[column]):
            values = df[column].dropna()
            n, bins, patches = ax.hist(values, bins=20, color='skyblue', edgecolor='black')

            y_max = max(n) * 1.1
            ax.set_ylim(0, y_max)

            for i in range(len(n)):
                if n[i] > 0:
                    x_pos = (bins[i] + bins[i + 1]) / 2
                    y_pos = n[i] + (y_max * 0.02)

                    ax.text(x_pos, y_pos, f'{int(n[i])}',
                            ha='center', va='bottom',
                            fontsize=9, fontweight='bold',
                            bbox=dict(facecolor='white', edgecolor='black', alpha=0.7,
                                      boxstyle='round,pad=0.2'))

            ax.set_title(f'Histogram of {column}')
            ax.set_xlabel(column)
            ax.set_ylabel('Frequency')

            stats_text = f"Min: {values.min():.2f}, Max: {values.max():.2f}, Avg: {values.mean():.2f}"
            ax.text(0.5, 0.97, stats_text, transform=ax.transAxes, ha='center',
                    bbox=dict(facecolor='white', alpha=0.8))

        else:
            counts = df[column].dropna().value_counts()
            bars = counts.plot(kind='bar', ax=ax, color='orange', edgecolor='black')

            y_max = max(counts) * 1.1
            ax.set_ylim(0, y_max)

            for i, v in enumerate(counts):
                ax.text(i, v + (y_max * 0.02), f'{v}',
                        ha='center', va='bottom',
                        fontsize=9, fontweight='bold',
                        bbox=dict(facecolor='white', edgecolor='black', alpha=0.7,
                                  boxstyle='round,pad=0.2'))

            ax.set_title(f'Bar Chart of {column}')
            ax.set_xlabel(column)
            ax.set_ylabel('Count')


    elif chart_type == "Boxplot":
        if pd.api.types.is_numeric_dtype(df[column]):
            values = df[column].dropna()
            # Tworzenie boxplota
            boxplot = ax.boxplot(values, patch_artist=True)

            for box in boxplot['boxes']:
                box.set(facecolor='skyblue')
            # Dodanie statystyk
            quartiles = values.quantile([0.25, 0.5, 0.75])
            iqr = quartiles[0.75] - quartiles[0.25]
            whisker_min = values[values >= quartiles[0.25] - 1.5 * iqr].min()
            whisker_max = values[values <= quartiles[0.75] + 1.5 * iqr].max()

            # Przypisanie statystyk do poszczególnych elementów boxplota
            positions = [1]

            ax.annotate(f'{whisker_min:.2f}',
                        xy=(positions[0], whisker_min),
                        xytext=(0, -15),
                        textcoords='offset points',
                        ha='center',
                        va='bottom',
                        bbox=dict(facecolor='white', edgecolor='black', alpha=0.7))

            ax.annotate(f'{quartiles[0.25]:.2f}',
                        xy=(positions[0], quartiles[0.25]),
                        xytext=(-20, 0),
                        textcoords='offset points',
                        ha='center',
                        va='bottom',
                        bbox=dict(facecolor='white', edgecolor='black', alpha=0.7))

            ax.annotate(f'{quartiles[0.5]:.2f}',
                        xy=(positions[0], quartiles[0.5]),
                        xytext=(0, 0),
                        textcoords='offset points',
                        ha='center',
                        va='bottom',
                        bbox=dict(facecolor='white', edgecolor='black', alpha=0.7))

            ax.annotate(f'{quartiles[0.75]:.2f}',
                        xy=(positions[0], quartiles[0.75]),
                        xytext=(20, 0),
                        textcoords='offset points',
                        ha='center',
                        va='bottom',
                        bbox=dict(facecolor='white', edgecolor='black', alpha=0.7))

            ax.annotate(f'{whisker_max:.2f}',
                        xy=(positions[0], whisker_max),
                        xytext=(0, 15),
                        textcoords='offset points',
                        ha='center',
                        va='top',
                        bbox=dict(facecolor='white', edgecolor='black', alpha=0.7))

            # Dodanie legendy z wyjaśnieniem
            ax.text(1.3, values.mean(),
                    f"Min: {whisker_min:.2f}\nQ1: {quartiles[0.25]:.2f}\nMedian: {quartiles[0.5]:.2f}\nQ3: {quartiles[0.75]:.2f}\nMax: {whisker_max:.2f}",
                    va='center', bbox=dict(facecolor='lightyellow', alpha=0.8))
            ax.set_title(f'Boxplot of {column}')

            # Ustaw margines prawy dla mieszczenia legendy
            plt.subplots_adjust(right=0.75)

        else:
            ax.text(0.5, 0.5, f'Boxplot not applicable for categorical column "{column}".', ha='center', va='center')

    elif chart_type == "Bar Chart":
        if not pd.api.types.is_numeric_dtype(df[column]):
            # Dla kolumn kategorycznych
            counts = df[column].dropna().value_counts()
            bars = counts.plot(kind='bar', ax=ax, color='green', edgecolor='black')

            # Ustaw limit osi Y, aby etykiety się mieściły
            y_max = max(counts) * 1.1
            ax.set_ylim(0, y_max)

            # Dodanie etykiet nad słupkami
            for i, v in enumerate(counts):
                ax.text(i, v + (y_max * 0.02), f'{v}',
                        ha='center', va='bottom',
                        fontsize=9, fontweight='bold',
                        bbox=dict(facecolor='white', edgecolor='black', alpha=0.7,
                                  boxstyle='round,pad=0.2'))

            ax.set_title(f'Bar Chart of {column}')
            ax.set_xlabel(column)
            ax.set_ylabel('Count')
        else:
            # Dla kolumn numerycznych - dzielimy na przedziały
            bins = 10
            values = df[column].dropna()
            counts, bin_edges = np.histogram(values, bins=bins)
            width = (bin_edges[1] - bin_edges[0]) * 0.8
            center = (bin_edges[:-1] + bin_edges[1:]) / 2

            # Tworzenie wykresu słupkowego
            bars = ax.bar(center, counts, width=width, color='purple', edgecolor='black')

            # Ustaw limit osi Y, aby etykiety się mieściły
            y_max = max(counts) * 1.1 if len(counts) > 0 else 10
            ax.set_ylim(0, y_max)

            # Dodanie etykiet nad słupkami
            for i, v in enumerate(counts):
                if v > 0:  # Dodawaj etykietę tylko dla słupków z wartościami
                    ax.text(center[i], v + (y_max * 0.02), f'{v}',
                            ha='center', va='bottom',
                            fontsize=9, fontweight='bold',
                            bbox=dict(facecolor='white', edgecolor='black', alpha=0.7,
                                      boxstyle='round,pad=0.2'))

            ax.set_title(f'Bar Chart (Binned) of {column}')
            ax.set_xlabel(column)
            ax.set_ylabel('Frequency')

            # Dodanie etykiet na osi X dla przedziałów
            ax.set_xticks(center)
            ax.set_xticklabels([f'{bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}' for i in range(len(bin_edges) - 1)],
                               rotation=45, ha='right')


    elif chart_type == "Line Plot":
        if pd.api.types.is_numeric_dtype(df[column]):
            values = df[column].dropna().reset_index(drop=True)

            # Sortowanie wartości dla lepszej prezentacji
            values_sorted = values.sort_values().reset_index(drop=True)

            # Wykrywanie wartości odstających
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            # Zastosowanie linii dla wartości regularnych i markerów dla odstających
            regular_mask = (values_sorted >= lower_bound) & (values_sorted <= upper_bound)
            outlier_mask = ~regular_mask

            # Indeksy wartości regularnych jako lista
            regular_indices = np.where(regular_mask)[0]

            # Rysowanie linii dla wartości regularnych
            line = ax.plot(regular_indices, values_sorted[regular_mask],
                           color='blue', marker='o', markersize=5, linestyle='-', linewidth=1, label='Regular values')

            y_min, y_max = values_sorted.min(), values_sorted.max()
            y_range = y_max - y_min
            ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.2)

            if len(regular_indices) > 0:
                step = max(1, len(regular_indices) // 10)
                for i in range(0, len(regular_indices), step):

                    if i < len(regular_indices):
                        idx = regular_indices[i]
                        val = values_sorted[regular_mask].iloc[i]

                        ax.annotate(f'{val:.1f}', (idx, val),
                                    xytext=(0, 10), textcoords='offset points',
                                    bbox=dict(facecolor='white', edgecolor='black', alpha=0.7),
                                    ha='center', va='bottom', fontsize=8)

            # Indeksy wartości odstających jako lista
            outlier_indices = np.where(outlier_mask)[0]

            # Rysowanie punktów dla wartości odstających
            if len(outlier_indices) > 0:
                outliers = ax.scatter(outlier_indices, values_sorted[outlier_mask],
                                      color='red', marker='*', s=100, label='Outliers')

                # Dodanie etykiet dla wszystkich wartości odstających
                for i, idx in enumerate(outlier_indices):
                    val = values_sorted[outlier_mask].iloc[i]

                    ax.annotate(f'{val:.1f}', (idx, val),
                                xytext=(0, 10), textcoords='offset points',
                                bbox=dict(facecolor='yellow', edgecolor='black', alpha=0.9),
                                ha='center', va='bottom', fontsize=9, fontweight='bold')

            ax.set_title(f'Line Plot of {column}')
            ax.set_xlabel('Index')
            ax.set_ylabel(column)
            ax.legend()

            # Dodaj siatkę dla lepszej czytelności
            ax.grid(True, linestyle='--', alpha=0.7)

            # Dodaj informacje o statystykach
            stats_text = f"Min: {values.min():.2f}, Max: {values.max():.2f}, Avg: {values.mean():.2f}"
            ax.text(0.5, 0.02, stats_text, transform=ax.transAxes, ha='center',
                    bbox=dict(facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, f'Line Plot not applicable for categorical column "{column}".', ha='center', va='center')

    elif chart_type == "Pie Chart":
        if not pd.api.types.is_numeric_dtype(df[column]):
            counts = df[column].dropna().value_counts()

            # Ogranicz liczbę kategorii dla czytelności
            if len(counts) > 10:
                others = pd.Series([counts[10:].sum()], index=['Others'])
                counts = pd.concat([counts[:10], others])

            # Wygeneruj kolory
            colors = plt.cm.Set3(np.linspace(0, 1, len(counts)))

            # Tworzenie wykresu kołowego
            wedges, texts, autotexts = ax.pie(
                counts,
                labels=None,  # Bez etykiet w wykresie, dodamy legendę
                autopct='%1.1f%%',
                colors=colors,
                startangle=90,  # Rozpocznij od góry
                shadow=True,
                wedgeprops=dict(width=0.6, edgecolor='w')
            )

            # Stylizacja tekstów z procentami
            for autotext in autotexts:
                autotext.set_weight('bold')
                autotext.set_fontsize(9)
                autotext.set_bbox(dict(facecolor='white', edgecolor='black', alpha=0.7, boxstyle='round,pad=0.2'))

            # Dodaj legendę z wartościami dokładnymi
            labels = [f'{name}: {count} ({count / counts.sum() * 100:.1f}%)' for name, count in counts.items()]
            ax.legend(wedges, labels, title=column, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

            # Ustaw odpowiedni margines prawy, aby legenda się mieściła
            plt.subplots_adjust(right=0.7)

            # Dodaj tytuł
            ax.set_title(f'Pie Chart of {column}')
        else:
            ax.text(0.5, 0.5, f'Pie Chart not applicable for numeric column "{column}".', ha='center', va='center')

    plt.tight_layout()

    return fig


def draw_figure(canvas, figure):
    for child in canvas.winfo_children():
        child.destroy()
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg


def extract_subtable(df, row_indices=None, col_indices=None, keep=False):
    if df is None:
        return None

    result_df = df.copy()

    print(f"Original dataframe has {len(df)} rows and {len(df.columns)} columns")
    print(f"Keep mode: {keep}")

    if row_indices is not None:
        try:
            print(f"Raw row indices: {row_indices}")

            processed_indices = []
            for r in row_indices:
                if isinstance(r, str) and '-' in r:
                    try:
                        start, end = map(int, r.split('-'))
                        processed_indices.extend(range(start, end + 1))
                    except ValueError:
                        print(f"Invalid range format: {r}")
                elif isinstance(r, str) and r.isdigit():
                    processed_indices.append(int(r))
                elif isinstance(r, int):
                    processed_indices.append(r)
                else:
                    print(f"Ignoring invalid row index: {r}")

            row_indices = processed_indices
            print(f"Processed row indices: {row_indices}")

            valid_indices = [idx for idx in row_indices if isinstance(idx, int) and 0 <= idx < len(result_df)]
            print(f"Valid row indices: {valid_indices}")

            if not valid_indices:
                print("No valid row indices found!")
                sg.popup("No valid row indices found. Please enter values between 0 and " + str(len(result_df) - 1))
                return None

            if keep:
                print(f"Keeping rows at indices: {valid_indices}")
                result_df = result_df.iloc[valid_indices, :]
            else:
                # Remove specified rows
                print(f"Removing rows at indices: {valid_indices}")
                result_df = result_df.drop(result_df.index[valid_indices])

            print(f"After row operation, dataframe has {len(result_df)} rows")

        except Exception as e:
            print(f"Error processing row indices: {e}")
            sg.popup(f"Error processing row indices: {e}")
            return None

    if col_indices is not None:
        # Similar processing for column indices...
        try:
            print(f"Raw column indices: {col_indices}")

            # Process column indices
            col_names = []
            for c in col_indices:
                if isinstance(c, int) and 0 <= c < len(result_df.columns):
                    col_names.append(result_df.columns[c])
                elif isinstance(c, str) and c in result_df.columns:
                    col_names.append(c)
                elif isinstance(c, str) and '-' in c:
                    # Handle range format for columns if they are numeric
                    try:
                        start, end = map(int, c.split('-'))
                        # Add all column names in the range if valid
                        for i in range(start, end + 1):
                            if 0 <= i < len(result_df.columns):
                                col_names.append(result_df.columns[i])
                    except ValueError:
                        print(f"Invalid column range format: {c}")

            print(f"Valid column names: {col_names}")

            if not col_names:
                print("No valid column indices found!")
                sg.popup("No valid column indices or names found.")
                return None

            if keep:
                # Keep only specified columns
                print(f"Keeping columns: {col_names}")
                result_df = result_df[col_names]
            else:
                # Remove specified columns
                print(f"Removing columns: {col_names}")
                result_df = result_df.drop(columns=col_names)

            print(f"After column operation, dataframe has {len(result_df.columns)} columns")

        except Exception as e:
            print(f"Error processing column indices: {e}")
            sg.popup(f"Error processing column indices: {e}")
            return None

    print(f"Final dataframe has {len(result_df)} rows and {len(result_df.columns)} columns")
    return result_df


def remove_columns(df, cols_to_remove):
    for c in cols_to_remove:
        if c in df.columns:
            df = df.drop(columns=c)
    return df


def replace_values(df, column, old_value, new_value):
    if column not in df.columns:
        sg.popup(f"Column '{column}' does not exist!")
        return df

    # Create a copy to avoid modifying the original DataFrame by reference
    df = df.copy()

    # Check if old_value exists in the column
    if old_value not in df[column].values and not (pd.api.types.is_numeric_dtype(df[column]) and
                                                   str(old_value).replace('.', '', 1).isdigit()):
        sg.popup(f"Old value '{old_value}' not found in column '{column}'.")
        return df

    # First convert the value to the appropriate type based on the type of column data
    try:
        # For numeric columns, try to find closest match
        if pd.api.types.is_numeric_dtype(df[column]):
            try:
                old_value_numeric = float(old_value) if '.' in old_value else int(old_value)

                # Find exact matches or close matches for float values
                if isinstance(old_value_numeric, float):
                    mask = np.isclose(df[column], old_value_numeric)
                else:
                    mask = df[column] == old_value_numeric

                # If switching from numeric to non-numeric, convert the column first
                if not str(new_value).replace('.', '', 1).replace('-', '', 1).isdigit():
                    df[column] = df[column].astype(str)

                df.loc[mask, column] = new_value
                print(f"Replaced '{old_value_numeric}' with '{new_value}' in column '{column}'")
            except ValueError:
                # If we can't convert to numeric, treat as string replacement
                mask = df[column].astype(str) == str(old_value)
                df.loc[mask, column] = new_value
        elif isinstance(df[column].dtype, pd.CategoricalDtype):
            # For categorical columns - convert to string first for more flexible replacement
            df[column] = df[column].astype(str)
            mask = df[column] == str(old_value)
            df.loc[mask, column] = new_value
            # Convert back to category
            df[column] = df[column].astype('category')
        else:
            # For string/object columns
            mask = df[column].astype(str) == str(old_value)
            df.loc[mask, column] = new_value

        # Try to infer the right dtype after replacement
        if all(str(val).replace('.', '', 1).replace('-', '', 1).isdigit()
               for val in df[column].dropna().unique()):
            try:
                # If all values are numeric after replacement, convert to numeric
                df[column] = pd.to_numeric(df[column], errors='ignore')
            except:
                pass
    except Exception as e:
        sg.popup(f"Error replacing values: {e}")

    return df


def replace_all_values(df, column, new_value):
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
    return df.drop_duplicates()


def one_hot_encoding(df, column):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in the dataframe")

    # Create dummy variables and join with original df
    dummies = pd.get_dummies(df[column], prefix=column, drop_first=True)
    result_df = pd.concat([df.drop(columns=[column]), dummies], axis=1)
    return result_df


def binary_encoding(df, column):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in the dataframe")

    # Convert to category if it's not already
    df = df.copy()
    df[column] = df[column].astype('category')
    df[f"{column}_encoded"] = df[column].cat.codes
    return df


def target_encoding(df, column, target):
    if column not in df.columns or target not in df.columns:
        raise ValueError(f"Column '{column}' or target '{target}' not found in the dataframe")

    # Calculate mean target value for each category
    df = df.copy()
    encoding_map = df.groupby(column)[target].mean().to_dict()
    df[f"{column}_target_encoded"] = df[column].map(encoding_map)
    return df


def scale_columns(df, cols, method='standard'):
    if df is None or df.empty:
        raise ValueError("The dataframe is empty. Please load a valid dataset.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    chosen_cols = [c for c in cols if c in numeric_cols]

    if not chosen_cols:
        raise ValueError("No valid numeric columns selected. Please select from: " + ", ".join(numeric_cols))

    # Create a copy to avoid modifying the original data
    df_copy = df.copy()

    # Handle any NaN values before scaling
    for col in chosen_cols:
        df_copy[col] = df_copy[col].fillna(df_copy[col].mean())

    # Perform scaling
    scaler = StandardScaler() if method == 'standard' else MinMaxScaler()

    try:
        scaled_values = scaler.fit_transform(df_copy[chosen_cols])
        df_copy.loc[:, chosen_cols] = scaled_values
        return df_copy
    except Exception as e:
        raise ValueError(f"Scaling error: {str(e)}")


def add_symbolic_column(df):
    if 'age' in df.columns and pd.api.types.is_numeric_dtype(df['age']):
        df['age_category'] = pd.cut(
            df['age'],
            bins=[0, 25, 45, 65, 150],
            labels=['Young', 'Adult', 'Senior', 'Elder'],
            right=False
        )
    return df


# def logistic_regression(df):
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

# Tab 1: Data & Statistics - zmodyfikowana z rozwijalnymi listami i przyciskami do zapisu/przywrócenia
tab1_layout = [
    [sg.Frame("Load Data", [
        [sg.Radio("Predefined Dataset", "DATA_SOURCE", default=True, key="-PREDEFINED_DATASET-"),
         sg.Radio("Custom CSV File", "DATA_SOURCE", key="-CUSTOM_CSV-")],
        [sg.Text("Select Dataset:"), sg.Combo(["Adult Dataset", "Kidney Disease Dataset"],
                                              default_value="Adult Dataset", key="-SELECT_FILE-", size=(30, 1)),
         sg.Button("Browse...", key="-BROWSE_CSV-", visible=False),
         sg.Text("", key="-FILE_PATH-", size=(40, 1), visible=False)],
        [sg.Button("Load Data", key="-LOAD-"),
         sg.Button("Restore Original Data", key="-RESTORE_DATA-"),
         sg.Button("Save Current Data", key="-SAVE_DATA-")]
    ])],

    [sg.Multiline(size=(100, 6), key="-DATA_INFO-", disabled=True)],

    [sg.Frame("Data Statistics", [
        [sg.Button("Compute Stats", key="-STATS-"), sg.Button("Correlation", key="-CORR-")],

        [sg.Text("Numeric Columns Statistics:")],
        [sg.Table(values=[],
                  headings=["Column", "Min", "Max", "Mean", "Median", "Std", "Mode", "Variance", "Skewness",
                            "Kurtosis"],
                  key="-NUMERIC_STATS-", auto_size_columns=True, justification='center', expand_x=True, expand_y=True)],

        [sg.Text("Categorical Columns Statistics:")],
        [sg.Table(values=[], headings=["Column", "Value Counts", "Mode"],
                  key="-CATEGORICAL_STATS-", auto_size_columns=True, justification='center', expand_x=True,
                  expand_y=True)]
    ])],

    [sg.Frame("Correlation Results", [
        [sg.Multiline(size=(150, 10), key="-CORR_OUT-", disabled=True)]
    ])]
]

# Tab 2: Extract Subtable - zmodyfikowana z rozwijalnymi listami
tab2_layout = [
    [sg.Frame("Subtable Extraction", [
        [sg.Radio("Remove Rows/Columns", "EXTRACTION_TYPE", default=True, key="-REMOVE_EXTRACT-"),
         sg.Radio("Keep Rows/Columns", "EXTRACTION_TYPE", key="-KEEP_EXTRACT-")],

        [sg.Text("Rows (comma separated indices or ranges):"),
         sg.InputText(key="-ROW_INPUT-", size=(30, 1))],

        [sg.Text("Columns:"),
         sg.Combo([], key="-COL_SELECT-", size=(30, 1)),
         sg.Button("Add Column", key="-ADD_COL-"),
         sg.Button("Clear Columns", key="-CLEAR_COLS-")],

        # Dodaj tę linię, która była brakująca w oryginalnym kodzie
        [sg.Text("Selected Columns:"), sg.Input("", key="-SELECTED_COLS-", size=(50, 1), readonly=True)],

        [sg.Button("Extract Subtable", key="-EXTRACT_BTN-"),
         sg.Button("Save Subtable", key="-SAVE_SUBTABLE-")],

        [sg.Multiline(size=(150, 10), key="-EXTRACT_OUT-", disabled=True, horizontal_scroll=True)]
    ])],

    [sg.Frame("Value Replacement", [
        [sg.Text("Select Column:"),
         sg.Combo([], key="-REPLACE_COL-", size=(30, 1))],

        [sg.Text("Current Value:"),
         sg.Combo([], key="-OLD_VAL-", size=(30, 1)),
         sg.Button("Get Values", key="-GET_VALUES-")],

        [sg.Text("New Value:"),
         sg.InputText(key="-NEW_VAL-", size=(30, 1))],

        [sg.Button("Replace Values in Column", key="-REPLACE_BTN-"),
         sg.Button("Save After Replacement", key="-SAVE_REPLACED-")],

        [sg.Text("Replace All in Column:"),
         sg.Combo([], key="-ALL_REPLACE_COL-", size=(30, 1))],

        [sg.Text("New Value for All:"),
         sg.InputText(key="-ALL_NEW_VAL-", size=(30, 1))],

        [sg.Button("Replace All Values", key="-REPLACE_ALL_BTN-")]
    ])]
]

# Tab 3: Scaling & Visualization - zmodyfikowana z rozwijalnymi listami
tab3_layout = [
    [sg.Frame("Data Scaling", [
        [sg.Text("Columns to Scale:"),
         sg.Combo([], key="-SCALE_COL_SELECT-", size=(30, 1)),
         sg.Button("Add", key="-ADD_SCALE_COL-"),
         sg.Button("Clear", key="-CLEAR_SCALE_COLS-")],

        # Upewnij się, że ten element istnieje w układzie interfejsu
        [sg.Text("Selected Columns:"), sg.Input("", key="-SELECTED_SCALE_COLS-", size=(50, 1), readonly=True)],

        [sg.Radio("StandardScaler", "SCALER", default=True, key="-STD_SCALER-"),
         sg.Radio("MinMaxScaler", "SCALER", key="-MINMAX_SCALER-")],

        [sg.Button("Apply Scaling", key="-APPLY_SCALING-"),
         sg.Button("Save Scaled Data", key="-SAVE_SCALED-"),
         sg.Button("Restore Original", key="-RESTORE_SCALED-")],

        [sg.Multiline(size=(150, 8), key="-SCALED_DATA-", disabled=True)]
    ])],

    [sg.Frame("Data Visualization", [
        [sg.Text("Select Column to Plot:"),
         sg.Combo([], key="-PLOT_SELECT-", size=(30, 1))],

        [sg.Text("Select Chart Type:")],
        [sg.Radio("Histogram", "CHART", default=True, key="-CHART_HIST-"),
         sg.Radio("Boxplot", "CHART", key="-CHART_BOX-"),
         sg.Radio("Bar Chart", "CHART", key="-CHART_BAR-"),
         sg.Radio("Line Plot", "CHART", key="-CHART_LINE-"),
         sg.Radio("Pie Chart", "CHART", key="-CHART_PIE-")],

        [sg.Button("Generate Plot", key="-PLOT_BTN-"),
         sg.Button("Save Plot as Image", key="-SAVE_PLOT-")],

        [sg.Canvas(key="-CANVAS-", size=(600, 400), expand_x=True, expand_y=True)]
    ])]
]

# Tab 4: Data Cleaning & Transformation - zmodyfikowana z rozwijalnymi listami
tab4_layout = [
    [sg.Frame("Handling Missing Values", [
        [sg.Combo(["remove", "fill_mean", "fill_median", "fill_mode"], default_value="remove",
                  key="-MISSING_STRATEGY-"),
         sg.Button("Apply Missing Values Handling", key="-APPLY_MISSING-"),
         sg.Button("Save Cleaned Data", key="-SAVE_CLEANED-")]
    ])],

    [sg.Frame("Duplicates", [
        [sg.Button("Remove Duplicates", key="-REMOVE_DUPLICATES-"),
         sg.Button("Save After Duplicate Removal", key="-SAVE_DEDUP-")]
    ])],

    [sg.Frame("Encoding", [
        [sg.Text("Select Column to Encode:"),
         sg.Combo([], key="-ENCODE_COL-", size=(30, 1))],

        [sg.Radio("One-Hot Encoding", "ENCODE_TYPE", default=True, key="-ONE_HOT-"),
         sg.Radio("Binary Encoding", "ENCODE_TYPE", key="-BINARY_ENCODE-"),
         sg.Radio("Target Encoding", "ENCODE_TYPE", key="-TARGET_ENCODE-")],

        [sg.Text("Select Target Column (for Target Encoding):"),
         sg.Combo([], key="-TARGET_COL-", size=(30, 1))],

        [sg.Button("Apply Encoding", key="-APPLY_ENCODING-"),
         sg.Button("Save Encoded Data", key="-SAVE_ENCODED-")]
    ])],

    [sg.Frame("Data Preview", [
        [sg.Multiline(size=(150, 10), key="-CLEANED_DATA-", disabled=True)]
    ])]
]

# Opcje zapisu
save_options_layout = [
    [sg.Checkbox("Include Row Indices", default=False, key="-SAVE_INCLUDE_INDEX-")],
    [sg.Checkbox("Include Column Headers", default=True, key="-SAVE_INCLUDE_HEADER-")],
    [sg.Button("Save", key="-CONFIRM_SAVE-"), sg.Button("Cancel", key="-CANCEL_SAVE-")]
]

background_color = '#64778d'
selected_background_color = '#4CAF50'
selected_text_color = 'white'

layout = [
    [sg.Column([
        [sg.TabGroup([[
            sg.Tab("Creators & Info", tab0_layout, expand_x=True, expand_y=True, background_color=background_color),
            sg.Tab("Data & Stats", tab1_layout, expand_x=True, expand_y=True, background_color=background_color),
            sg.Tab("Replacement & Subtable", tab2_layout, expand_x=True, expand_y=True,
                   background_color=background_color),
            sg.Tab("Scaling & Visualization", tab3_layout, expand_x=True, expand_y=True,
                   background_color=background_color),
            sg.Tab("Data Cleaning & Transformation", tab4_layout, expand_x=True, expand_y=True,
                   background_color=background_color)
        ]], tab_location='top', font=("Helvetica", 14, "bold"), expand_x=True, expand_y=True, key='-TABGROUP-',
            size=(1200, 1200))]
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

    # Obsługa ładowania danych
    elif event == "-BROWSE_CSV-":
        file_path = sg.popup_get_file('Choose CSV file', file_types=(("CSV Files", "*.csv"),))
        if file_path:
            window["-FILE_PATH-"].update(file_path)
            window["-FILE_PATH-"].update(visible=True)

    elif event == "-PREDEFINED_DATASET-":
        window["-SELECT_FILE-"].update(visible=True)
        window["-BROWSE_CSV-"].update(visible=False)
        window["-FILE_PATH-"].update(visible=False)

    elif event == "-CUSTOM_CSV-":
        window["-SELECT_FILE-"].update(visible=False)
        window["-BROWSE_CSV-"].update(visible=True)


    elif event == "-LOAD-":
        try:
            if values["-PREDEFINED_DATASET-"]:
                selected_file = values["-SELECT_FILE-"]
                df = load_dataset(selected_file, is_predefined=True)
            else:
                # Gdy wybrano Custom CSV, sprawdź czy już wybrano plik
                file_path = values.get("-FILE_PATH-", "")

                # Jeśli nie ma jeszcze wybranego pliku, otwórz okno dialogowe wyboru pliku
                if not file_path:
                    file_path = sg.popup_get_file('Choose CSV file',
                                                  file_types=(("CSV Files", "*.csv"),))

                    # Aktualizuj etykietę z wybraną ścieżką
                    if file_path:
                        window["-FILE_PATH-"].update(file_path)
                        window["-FILE_PATH-"].update(visible=True)
                    else:
                        # Użytkownik anulował wybór pliku
                        continue

                # Załaduj wybrany plik CSV
                df = load_dataset(file_path, is_predefined=False)

            # Aktualizacja interfejsu
            window["-DATA_INFO-"].update(f"Dataset loaded with {len(df)} rows and {len(df.columns)} columns:\n" +
                                         ", ".join(df.columns))
            # Aktualizacja list rozwijanych ze wszystkimi kolumnami
            all_columns = list(df.columns)
            window["-PLOT_SELECT-"].update(values=all_columns, value=all_columns[0] if all_columns else "")
            window["-ENCODE_COL-"].update(values=all_columns)
            window["-TARGET_COL-"].update(values=all_columns)
            window["-REPLACE_COL-"].update(values=all_columns)
            window["-ALL_REPLACE_COL-"].update(values=all_columns)
            window["-COL_SELECT-"].update(values=all_columns)
            window["-SCALE_COL_SELECT-"].update(values=all_columns)
            sg.popup("Dataset loaded successfully!", keep_on_top=True)

        except Exception as e:
            sg.popup(f"Error loading dataset: {e}", keep_on_top=True)

    elif event == "-RESTORE_DATA-":
        if original_df is not None:
            df = restore_original_data()
            window["-DATA_INFO-"].update(f"Original data restored with {len(df)} rows and {len(df.columns)} columns.")
            sg.popup("Original data restored successfully!")
        else:
            sg.popup("No original data available to restore!")

    elif event == "-SAVE_DATA-":
        # Pokaż okno dialogowe z opcjami zapisu
        save_window = sg.Window("Save Options", save_options_layout)
        while True:
            save_event, save_values = save_window.read()
            if save_event in (sg.WIN_CLOSED, "-CANCEL_SAVE-"):
                break
            elif save_event == "-CONFIRM_SAVE-":
                include_index = save_values["-SAVE_INCLUDE_INDEX-"]
                include_header = save_values["-SAVE_INCLUDE_HEADER-"]
                save_dataframe_to_csv(df, include_index, include_header)
                break
        save_window.close()

    # Obsługa statystyk
    elif event == "-STATS-":
        if df is None:
            sg.popup("Please load the dataset first.")
        else:
            try:
                dataset_type = values["-SELECT_FILE-"] if values["-PREDEFINED_DATASET-"] else "Custom"
                print("Data types before statistics calculation:")
                print(df.dtypes)
                stats_data = compute_statistics(df, dataset_type)

                # Separate numeric and categorical stats
                numeric_stats = []
                categorical_stats = []
                for stat in stats_data:
                    # Check if this is a numeric stat (has at least 2 numeric values)
                    if len(stat) > 2 and isinstance(stat[1], (int, float)) and isinstance(stat[2], (int, float)):
                        numeric_stats.append(stat)
                    elif len(stat) <= 3:  # Categorical stats have 3 or fewer items
                        categorical_stats.append(stat)

                print(f"Found {len(numeric_stats)} numeric stats and {len(categorical_stats)} categorical stats")
                window["-NUMERIC_STATS-"].update(values=numeric_stats)
                window["-CATEGORICAL_STATS-"].update(values=categorical_stats)

                if not numeric_stats:
                    sg.popup("Warning: No numeric statistics were calculated. Check console for details.")
            except Exception as e:
                print(f"Error calculating statistics: {e}")
                import traceback

                traceback.print_exc()
                sg.popup(f"Error calculating statistics: {str(e)}")

    elif event == "-CORR-":
        if df is None:
            sg.popup("Please load the dataset first.")
        else:
            pearson_corr, spearman_corr = compute_correlation(df)
            if pearson_corr is None or spearman_corr is None:
                window["-CORR_OUT-"].update("Not enough numeric columns for correlation.")
            else:
                # Format correlation matrices for better readability
                def format_corr_matrix(corr_matrix, title):
                    result = f"\n{title}\n" + "=" * len(title) + "\n\n"
                    formatted_matrix = []
                    headers = [""] + list(corr_matrix.columns)
                    formatted_matrix.append("\t".join(str(h) for h in headers))
                    for idx, row in corr_matrix.iterrows():
                        formatted_row = [str(idx)]
                        for val in row:
                            formatted_row.append(f"{val:.3f}")
                        formatted_matrix.append("\t".join(formatted_row))
                    return result + "\n".join(formatted_matrix)


                pearson_formatted = format_corr_matrix(pearson_corr, "Pearson Correlation Matrix")
                spearman_formatted = format_corr_matrix(spearman_corr, "Spearman Correlation Matrix")
                window["-CORR_OUT-"].update(pearson_formatted + "\n\n" + spearman_formatted)

    # Obsługa subtable i operacji na wartościach
    # Poprawka dla obsługi zdarzenia "-ADD_COL-"
    elif event == "-ADD_COL-":
        if df is None:
            sg.popup("Please load the dataset first.")
            continue

        selected_col = values.get("-COL_SELECT-", "")
        if not selected_col:
            continue

        # Użyj metody get() aby uniknąć KeyError
        current_cols = values.get("-SELECTED_COLS-", "").strip()
        if current_cols:
            updated_cols = current_cols + ", " + selected_col
        else:
            updated_cols = selected_col

        # Sprawdź, czy element istnieje w oknie przed próbą aktualizacji
        if "-SELECTED_COLS-" in window.key_dict:
            window["-SELECTED_COLS-"].update(updated_cols)
        else:
            # Jeśli element nie istnieje, wyświetl komunikat dla diagnostyki
            print(f"Warning: Key '-SELECTED_COLS-' not found in window elements")
            sg.popup("UI error: Could not update selected columns list.")


    elif event == "-CLEAR_COLS-":
        if "-SELECTED_COLS-" in window.key_dict:
            window["-SELECTED_COLS-"].update("")
        else:
            print(f"Warning: Key '-SELECTED_COLS-' not found in window elements")
            sg.popup("UI error: Could not clear selected columns list.")

    elif event == "-EXTRACT_BTN-":
        if df is None:
            sg.popup("Please load the dataset first.")
        else:
            try:
                # Get input and remove whitespace
                row_input = values["-ROW_INPUT-"].strip()
                col_input = values["-SELECTED_COLS-"].strip()

                # Debug prints
                print(f"Row input: '{row_input}'")
                print(f"Column input: '{col_input}'")

                # Parse row indices
                row_indices = None
                if row_input:
                    # Split by comma and strip whitespace from each part
                    row_indices = [part.strip() for part in row_input.split(",") if part.strip()]
                    # Convert to integers if possible
                    row_indices = [int(i) if i.isdigit() else i for i in row_indices]
                    print(f"Parsed row indices: {row_indices}")

                # Parse column indices
                col_indices = None
                if col_input:
                    col_indices = [part.strip() for part in col_input.split(",") if part.strip()]
                    col_indices = [int(i) if i.isdigit() else i for i in col_indices]
                    print(f"Parsed column indices: {col_indices}")

                keep = values["-KEEP_EXTRACT-"]
                print(f"Keep mode: {keep}")

                # Extract subtable
                sub_df = extract_subtable(df, row_indices, col_indices, keep)

                if sub_df is None or sub_df.empty:
                    window["-EXTRACT_OUT-"].update("Invalid range or empty subtable.")
                else:
                    # Display result
                    display_df = sub_df.head(10)
                    result_text = display_df.to_string()
                    if len(sub_df) > 10:
                        result_text += f"\n\n[Showing first 10 of {len(sub_df)} rows]"
                    window["-EXTRACT_OUT-"].update(result_text)

                    # Aktualizacja globalnego DataFrame
                    df = sub_df
            except Exception as e:
                print(f"Error in subtable extraction: {e}")
                import traceback

                traceback.print_exc()
                sg.popup(f"Error extracting subtable: {e}")

    elif event == "-SAVE_SUBTABLE-":
        # Pokaż okno dialogowe z opcjami zapisu
        save_window = sg.Window("Save Options", save_options_layout)
        while True:
            save_event, save_values = save_window.read()
            if save_event in (sg.WIN_CLOSED, "-CANCEL_SAVE-"):
                break
            elif save_event == "-CONFIRM_SAVE-":
                include_index = save_values["-SAVE_INCLUDE_INDEX-"]
                include_header = save_values["-SAVE_INCLUDE_HEADER-"]
                save_dataframe_to_csv(df, include_index, include_header)
                break
        save_window.close()

    elif event == "-GET_VALUES-":
        if df is None:
            sg.popup("Please load the dataset first.")
            continue

        selected_col = values["-REPLACE_COL-"]
        if not selected_col or selected_col not in df.columns:
            sg.popup("Please select a valid column.")
            continue

        # Pobierz unikalne wartości z kolumny i zaktualizuj listę rozwijaną
        unique_values = df[selected_col].dropna().unique().tolist()
        unique_values = [str(val) for val in unique_values]  # Konwersja na stringi dla listy rozwijanej
        window["-OLD_VAL-"].update(values=unique_values, value=unique_values[0] if unique_values else "")

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
                    # Zachowaj kopię oryginalnego DataFrame
                    temp_df = df.copy()

                    # Zastosuj funkcję zastępowania
                    df = replace_values(df, col_to_replace, old_value, new_value)

                    # Sprawdź, czy nastąpiła zmiana
                    changes_made = not df[col_to_replace].equals(temp_df[col_to_replace])

                    if changes_made:
                        window["-EXTRACT_OUT-"].update(
                            f"Replaced '{old_value}' with '{new_value}' in column '{col_to_replace}'\n\n" +
                            df.head(10).to_string() +
                            (f"\n\n[Showing first 10 of {len(df)} rows]" if len(df) > 10 else "")
                        )

                        # Aktualizacja listy unikalnych wartości po zamianie
                        unique_values = df[col_to_replace].dropna().unique().tolist()
                        unique_values = [str(val) for val in unique_values]
                        window["-OLD_VAL-"].update(values=unique_values)
                    else:
                        window["-EXTRACT_OUT-"].update(
                            f"No values '{old_value}' found in column '{col_to_replace}' to replace."
                        )
                except Exception as e:
                    sg.popup(f"Error replacing values: {e}")


    elif event == "-SAVE_REPLACED-":
        # Użyj unikalnej nazwy dla okna, aby zapewnić unikalne klucze elementów
        save_window = create_save_options_window("Save After Replacement")
        while True:
            save_event, save_values = save_window.read()
            if save_event in (sg.WIN_CLOSED, "-CANCEL_SAVE-"):
                break
            elif save_event == "-CONFIRM_SAVE-":
                # Pobierz wartości z elementów o unikalnych kluczach
                include_index = save_values[f"-SAVE_INCLUDE_INDEX-Save After Replacement"]
                include_header = save_values[f"-SAVE_INCLUDE_HEADER-Save After Replacement"]
                save_dataframe_to_csv(df, include_index, include_header)
                break
        save_window.close()

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
                    # Keep a copy of the original dataframe for comparison
                    temp_df = df.copy()

                    # Apply the replace_all_values function
                    df = replace_all_values(df, col_to_replace_all, new_value_all)

                    # Check if the column exists in the dataframe
                    if col_to_replace_all in df.columns:
                        # Update the extract output area with confirmation and data preview
                        window["-EXTRACT_OUT-"].update(
                            f"Replaced all values in column '{col_to_replace_all}' with '{new_value_all}'\n\n" +
                            df.head(10).to_string() +
                            (f"\n\n[Showing first 10 of {len(df)} rows]" if len(df) > 10 else "")
                        )
                    else:
                        window["-EXTRACT_OUT-"].update(
                            f"Column '{col_to_replace_all}' does not exist in the dataframe."
                        )
                except Exception as e:
                    sg.popup(f"Error replacing all values: {e}")

    # Obsługa skalowania i wizualizacji
    elif event == "-ADD_SCALE_COL-":
        if df is None:
            sg.popup("Please load the dataset first.")
            continue

        selected_col = values.get("-SCALE_COL_SELECT-", "")
        if not selected_col:
            continue

        # Użyj metody get() aby uniknąć KeyError
        current_cols = values.get("-SELECTED_SCALE_COLS-", "").strip()
        if current_cols:
            updated_cols = current_cols + ", " + selected_col
        else:
            updated_cols = selected_col

        # Sprawdź, czy element istnieje w oknie przed próbą aktualizacji
        if "-SELECTED_SCALE_COLS-" in window.key_dict:
            window["-SELECTED_SCALE_COLS-"].update(updated_cols)
        else:
            print(f"Warning: Key '-SELECTED_SCALE_COLS-' not found in window elements")
            sg.popup("UI error: Could not update selected scale columns list.")


    elif event == "-CLEAR_SCALE_COLS-":
        if "-SELECTED_SCALE_COLS-" in window.key_dict:
            window["-SELECTED_SCALE_COLS-"].update("")
        else:
            print(f"Warning: Key '-SELECTED_SCALE_COLS-' not found in window elements")
            sg.popup("UI error: Could not clear selected scale columns list.")

    elif event == "-APPLY_SCALING-":
        if df is None:
            sg.popup("Please load the dataset first.")
        else:
            cols_str = values["-SELECTED_SCALE_COLS-"]
            cols_list = [c.strip() for c in cols_str.split(",") if c.strip() != ""]  # Clean the input

            if not cols_list:
                sg.popup("Please enter column names to scale, separated by commas.")
                continue

            method = "standard" if values["-STD_SCALER-"] else "minmax"

            try:
                # Create a copy to avoid modifying the original dataframe directly
                scaled_df = scale_columns(df, cols_list, method=method)

                # Update the original dataframe with the scaled values
                for col in cols_list:
                    if col in scaled_df.columns and col in df.columns:
                        df[col] = scaled_df[col]

                # Show preview of scaled data
                window["-SCALED_DATA-"].update(df[cols_list].head(10).to_string())

                # Show confirmation message
                sg.popup(f"Applied {method} scaling to columns: {', '.join(cols_list)}")
            except Exception as e:
                sg.popup(f"Error scaling columns: {str(e)}")

    elif event == "-SAVE_SCALED-":
        # Pokaż okno dialogowe z opcjami zapisu
        save_window = sg.Window("Save Options", save_options_layout)
        while True:
            save_event, save_values = save_window.read()
            if save_event in (sg.WIN_CLOSED, "-CANCEL_SAVE-"):
                break
            elif save_event == "-CONFIRM_SAVE-":
                include_index = save_values["-SAVE_INCLUDE_INDEX-"]
                include_header = save_values["-SAVE_INCLUDE_HEADER-"]
                save_dataframe_to_csv(df, include_index, include_header)
                break
        save_window.close()

    elif event == "-RESTORE_SCALED-":
        if original_df is not None:
            df = restore_original_data()
            window["-SCALED_DATA-"].update(df.head(10).to_string())
            sg.popup("Original data restored successfully!")
        else:
            sg.popup("No original data available to restore!")


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

            try:
                fig = generate_plot(df, col_to_plot, chart_type)
                if figure_canvas_agg:
                    figure_canvas_agg.get_tk_widget().forget()

                figure_canvas_agg = draw_figure(window["-CANVAS-"].TKCanvas, fig)

            except Exception as e:
                print(f"Error generating plot: {e}")
                import traceback

                traceback.print_exc()
                sg.popup(f"Error generating plot: {str(e)}")

    elif event == "-SAVE_PLOT-":
        if figure_canvas_agg is None:
            sg.popup("Please generate a plot first.")
            continue

        file_path = sg.popup_get_file('Save plot as image', save_as=True,
                                      file_types=(("PNG Files", "*.png"), ("All Files", "*.*")),
                                      default_extension=".png")
        if file_path:
            try:
                # Ensure file has .png extension
                if not file_path.lower().endswith('.png'):
                    file_path += '.png'

                # Get the current figure and save it
                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                sg.popup(f"Plot saved to {file_path}")
            except Exception as e:
                sg.popup(f"Error saving plot: {e}")

    # Obsługa czyszczenia danych
    elif event == "-APPLY_MISSING-":
        if df is None:
            sg.popup("Please load the dataset first.")
        else:
            strategy = values["-MISSING_STRATEGY-"]
            try:
                df = handle_missing_values(df, strategy)
                sg.popup(f"Missing values handled with strategy: {strategy}")
                window["-CLEANED_DATA-"].update(df.head(10).to_string())
            except Exception as e:
                sg.popup(f"Error handling missing values: {e}")

    elif event == "-SAVE_CLEANED-":
        # Pokaż okno dialogowe z opcjami zapisu
        save_window = sg.Window("Save Options", save_options_layout)
        while True:
            save_event, save_values = save_window.read()
            if save_event in (sg.WIN_CLOSED, "-CANCEL_SAVE-"):
                break
            elif save_event == "-CONFIRM_SAVE-":
                include_index = save_values["-SAVE_INCLUDE_INDEX-"]
                include_header = save_values["-SAVE_INCLUDE_HEADER-"]
                save_dataframe_to_csv(df, include_index, include_header)
                break
        save_window.close()

    elif event == "-REMOVE_DUPLICATES-":
        if df is None:
            sg.popup("Please load the dataset first.")
        else:
            try:
                original_len = len(df)
                df = remove_duplicates(df)
                removed_count = original_len - len(df)
                sg.popup(f"Duplicate rows removed successfully. Removed {removed_count} rows.")
                window["-CLEANED_DATA-"].update(df.head(10).to_string())
            except Exception as e:
                sg.popup(f"Error removing duplicates: {e}")

    elif event == "-SAVE_DEDUP-":
        # Pokaż okno dialogowe z opcjami zapisu
        save_window = sg.Window("Save Options", save_options_layout)
        while True:
            save_event, save_values = save_window.read()
            if save_event in (sg.WIN_CLOSED, "-CANCEL_SAVE-"):
                break
            elif save_event == "-CONFIRM_SAVE-":
                include_index = save_values["-SAVE_INCLUDE_INDEX-"]
                include_header = save_values["-SAVE_INCLUDE_HEADER-"]
                save_dataframe_to_csv(df, include_index, include_header)
                break
        save_window.close()

    elif event == "-APPLY_ENCODING-":
        if df is None:
            sg.popup("Please load the dataset first.")
        else:
            column = values["-ENCODE_COL-"]
            if not column:
                sg.popup("Please select a column to encode")
                continue

            try:
                if values["-ONE_HOT-"]:
                    df = one_hot_encoding(df, column)
                    sg.popup(f"One-Hot Encoding applied to column: {column}")
                elif values["-BINARY_ENCODE-"]:
                    df = binary_encoding(df, column)
                    sg.popup(f"Binary Encoding applied to column: {column}")
                elif values["-TARGET_ENCODE-"]:
                    target_column = values["-TARGET_COL-"]
                    if not target_column:
                        sg.popup("Please select a target column for Target Encoding.")
                    else:
                        df = target_encoding(df, column, target_column)
                        sg.popup(f"Target Encoding applied to column: {column} using target column: {target_column}")

                # Update UI elements that might display columns
                all_columns = list(df.columns)
                window["-PLOT_SELECT-"].update(values=all_columns)
                window["-ENCODE_COL-"].update(values=all_columns)
                window["-TARGET_COL-"].update(values=all_columns)
                window["-REPLACE_COL-"].update(values=all_columns)
                window["-ALL_REPLACE_COL-"].update(values=all_columns)
                window["-COL_SELECT-"].update(values=all_columns)
                window["-SCALE_COL_SELECT-"].update(values=all_columns)

                # Show preview of transformed data
                window["-CLEANED_DATA-"].update(df.head(10).to_string())
            except Exception as e:
                sg.popup(f"Error applying encoding: {str(e)}")

    elif event == "-SAVE_ENCODED-":
        # Pokaż okno dialogowe z opcjami zapisu
        save_window = sg.Window("Save Options", save_options_layout)
        while True:
            save_event, save_values = save_window.read()
            if save_event in (sg.WIN_CLOSED, "-CANCEL_SAVE-"):
                break
            elif save_event == "-CONFIRM_SAVE-":
                include_index = save_values["-SAVE_INCLUDE_INDEX-"]
                include_header = save_values["-SAVE_INCLUDE_HEADER-"]
                save_dataframe_to_csv(df, include_index, include_header)
                break
        save_window.close()
