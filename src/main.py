import PySimpleGUI as sg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import kurtosis, skew
from tkinter import filedialog
from PIL import Image, ImageTk

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800

sg.set_options(font=("Helvetica", 12))
df = None
original_df = None
figure_canvas_agg = None


def browse_for_csv_file():
    file_path = filedialog.askopenfilename(
        title="Wybierz plik CSV",
        filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
    )
    return file_path if file_path else None


def set_window_icon(window, icon_path):
    try:
        if not os.path.exists(icon_path):
            print(f"Plik ikony {icon_path} nie istnieje.")
            if os.path.exists("assets/logo.png"):
                icon_path = "assets/logo.png"
            elif os.path.exists("logo.png"):
                icon_path = "logo.png"
            else:
                print("Nie można znaleźć pliku ikony.")
                return

        root = window.TKroot

        icon_image = Image.open(icon_path)

        icon_photo = ImageTk.PhotoImage(icon_image)

        root.iconphoto(True, icon_photo)

    except Exception as e:
        print(f"Error icon: {e}")


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
                df[col] = pd.to_numeric(df[col], errors='coerce')

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
    unique_suffix = f"_{id(title)}_{title}"

    save_layout = [
        [sg.Checkbox("Include Row Indices", default=False, key=f"-SAVE_INCLUDE_INDEX{unique_suffix}")],
        [sg.Checkbox("Include Column Headers", default=True, key=f"-SAVE_INCLUDE_HEADER{unique_suffix}")],
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

        if not file_path:
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
                df = df.copy()
                df[col] = pd.to_numeric(df[col], errors='coerce')

        for col in numeric_cols:
            if col in df.columns:
                print(f"{col}: {df[col].dtype}")
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col not in df.columns:
            print(f"Warning: Column {col} not found in dataframe")
            continue

        col_data = df[col].dropna()

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
        print("Not enough numeric columns for correlation")
        return None, None

    try:
        df_clean = df[numeric_cols].dropna()

        pearson_corr = df_clean.corr(method='pearson')
        spearman_corr = df_clean.corr(method='spearman')

        if not pearson_corr.empty:
            first_col = pearson_corr.columns[0]

        return pearson_corr, spearman_corr
    except Exception as e:
        print(f"Error computing correlation: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def format_correlation_matrices(pearson_corr, spearman_corr):
    if pearson_corr is None or spearman_corr is None or pearson_corr.empty or spearman_corr.empty:
        return [], [], ["Column"]

    numeric_cols = list(pearson_corr.columns)

    pearson_data = []
    for idx, row in pearson_corr.iterrows():
        row_data = [idx]
        for col in numeric_cols:
            row_data.append(round(row[col], 3))
        pearson_data.append(row_data)

    spearman_data = []
    for idx, row in spearman_corr.iterrows():
        row_data = [idx]
        for col in numeric_cols:
            row_data.append(round(row[col], 3))
        spearman_data.append(row_data)

    headers = ["Column"] + numeric_cols

    return pearson_data, spearman_data, headers


def generate_plot(df, column, chart_type, options):
    show_values = options.get("show_values", True)
    show_grid = options.get("show_grid", True)
    label_size = options.get("label_size", 9)
    chart_title = options.get("chart_title", f"{chart_type} of {column}")
    color_theme = options.get("color_theme", "Default")

    if color_theme == "Pastel":
        colors = plt.cm.Pastel1.colors
    elif color_theme == "Dark":
        colors = plt.cm.Dark2.colors
    elif color_theme == "Colorblind":
        colors = plt.cm.tab10.colors
    elif color_theme == "Grayscale":
        colors = plt.cm.Greys(np.linspace(0.3, 0.9, 10))
    else:
        colors = plt.cm.Set3.colors

    fig, ax = plt.subplots(figsize=(8, 5))

    if column not in df.columns:
        ax.text(0.5, 0.5, f'Column "{column}" not found.', ha='center', va='center')
        return fig

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)

    if show_grid:
        ax.grid(True, linestyle='--', alpha=0.7)
    else:
        ax.grid(False)

    if chart_title:
        ax.set_title(chart_title, fontsize=12)
    else:
        ax.set_title(f'{chart_type} of {column}', fontsize=12)

    if chart_type == "Histogram":
        if pd.api.types.is_numeric_dtype(df[column]):
            values = df[column].dropna()
            n, bins, patches = ax.hist(values, bins=10, color=colors[0], edgecolor='black')

            if show_values:
                y_max = max(n) * 1.1
                ax.set_ylim(0, y_max)

                for i in range(len(n)):
                    if n[i] > 0:
                        x_pos = (bins[i] + bins[i + 1]) / 2
                        y_pos = n[i] + (y_max * 0.02)
                        ax.text(x_pos, y_pos, f'{int(n[i])}',
                                ha='center', va='bottom',
                                fontsize=label_size,
                                bbox=dict(facecolor='white', edgecolor='black', alpha=0.7, boxstyle='round,pad=0.2'))

            ax.set_xlabel(column, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)

            stats_text = f"Min: {values.min():.2f}, Max: {values.max():.2f}, Avg: {values.mean():.2f}"
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, ha='right', va='top',
                    fontsize=9, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'))

        else:
            counts = df[column].dropna().value_counts()
            if len(counts) > 8:
                counts = counts.sort_values(ascending=False).head(8)

            bars = counts.plot(kind='bar', ax=ax, color=colors[0], edgecolor='black')

            if show_values:
                y_max = max(counts) * 1.1
                ax.set_ylim(0, y_max)

                for i, v in enumerate(counts):
                    ax.text(i, v + (y_max * 0.02), f'{v}',
                            ha='center', va='bottom',
                            fontsize=label_size,
                            bbox=dict(facecolor='white', edgecolor='black', alpha=0.7, boxstyle='round,pad=0.2'))

            ax.set_xlabel(column, fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            plt.xticks(rotation=45, ha='right')

    elif chart_type == "Boxplot":
        if pd.api.types.is_numeric_dtype(df[column]):
            values = df[column].dropna()
            boxplot = ax.boxplot(values, patch_artist=True, widths=0.5)

            for box in boxplot['boxes']:
                box.set(facecolor=colors[0])

            if show_values:
                quartiles = values.quantile([0.25, 0.5, 0.75])
                iqr = quartiles[0.75] - quartiles[0.25]
                whisker_min = values[values >= quartiles[0.25] - 1.5 * iqr].min()
                whisker_max = values[values <= quartiles[0.75] + 1.5 * iqr].max()

                positions = [1]

                ax.annotate(f'{whisker_min:.2f}',
                            xy=(positions[0], whisker_min),
                            xytext=(-15, 0),
                            textcoords='offset points',
                            ha='right', va='center',
                            fontsize=label_size,
                            bbox=dict(facecolor='white', edgecolor='black', alpha=0.7, boxstyle='round,pad=0.2'))

                ax.annotate(f'{quartiles[0.25]:.2f}',
                            xy=(positions[0], quartiles[0.25]),
                            xytext=(-15, 0),
                            textcoords='offset points',
                            ha='right', va='center',
                            fontsize=label_size,
                            bbox=dict(facecolor='white', edgecolor='black', alpha=0.7, boxstyle='round,pad=0.2'))

                ax.annotate(f'{quartiles[0.5]:.2f}',
                            xy=(positions[0], quartiles[0.5]),
                            xytext=(15, 0),
                            textcoords='offset points',
                            ha='left', va='center',
                            fontsize=label_size,
                            bbox=dict(facecolor='white', edgecolor='black', alpha=0.7, boxstyle='round,pad=0.2'))

                ax.annotate(f'{quartiles[0.75]:.2f}',
                            xy=(positions[0], quartiles[0.75]),
                            xytext=(15, 0),
                            textcoords='offset points',
                            ha='left', va='center',
                            fontsize=label_size,
                            bbox=dict(facecolor='white', edgecolor='black', alpha=0.7, boxstyle='round,pad=0.2'))

                ax.annotate(f'{whisker_max:.2f}',
                            xy=(positions[0], whisker_max),
                            xytext=(15, 0),
                            textcoords='offset points',
                            ha='left', va='center',
                            fontsize=label_size,
                            bbox=dict(facecolor='white', edgecolor='black', alpha=0.7, boxstyle='round,pad=0.2'))

                leg_text = f"Min: {whisker_min:.2f}\nQ1: {quartiles[0.25]:.2f}\nMedian: {quartiles[0.5]:.2f}\nQ3: {quartiles[0.75]:.2f}\nMax: {whisker_max:.2f}"
                ax.text(1.25, values.mean(), leg_text, va='center', fontsize=10,
                        bbox=dict(facecolor='lightyellow', alpha=0.8, boxstyle='round,pad=0.2'))

            ax.set_ylabel(column, fontsize=10)
            ax.set_xticklabels([])

        else:
            ax.text(0.5, 0.5, f'Boxplot not applicable for categorical column "{column}".',
                    ha='center', va='center', fontsize=10)

    elif chart_type == "Bar Chart":
        if not pd.api.types.is_numeric_dtype(df[column]):
            counts = df[column].dropna().value_counts()

            if len(counts) > 8:
                counts = counts.sort_values(ascending=False).head(8)

            bars = counts.plot(kind='bar', ax=ax, color=colors, edgecolor='black')

            if show_values:
                y_max = max(counts) * 1.1
                ax.set_ylim(0, y_max)

                for i, v in enumerate(counts):
                    ax.text(i, v + (y_max * 0.02), f'{v}',
                            ha='center', va='bottom',
                            fontsize=label_size,
                            bbox=dict(facecolor='white', edgecolor='black', alpha=0.7, boxstyle='round,pad=0.2'))

            ax.set_xlabel(column, fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            plt.xticks(rotation=45, ha='right')
        else:
            bins = 8
            values = df[column].dropna()
            counts, bin_edges = np.histogram(values, bins=bins)
            width = (bin_edges[1] - bin_edges[0]) * 0.8
            center = (bin_edges[:-1] + bin_edges[1:]) / 2

            bars = ax.bar(center, counts, width=width, color=colors, edgecolor='black')

            if show_values:
                y_max = max(counts) * 1.1 if len(counts) > 0 else 10
                ax.set_ylim(0, y_max)

                for i, v in enumerate(counts):
                    if v > 0:
                        ax.text(center[i], v + (y_max * 0.02), f'{v}',
                                ha='center', va='bottom',
                                fontsize=label_size,
                                bbox=dict(facecolor='white', edgecolor='black', alpha=0.7, boxstyle='round,pad=0.2'))

            ax.set_xlabel(column, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)

            x_labels = [f'{bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}' for i in range(len(bin_edges) - 1)]
            ax.set_xticks(center)
            ax.set_xticklabels(x_labels, rotation=45, ha='right')

    elif chart_type == "Line Plot":
        if pd.api.types.is_numeric_dtype(df[column]):
            values = df[column].dropna().reset_index(drop=True)
            values_sorted = values.sort_values().reset_index(drop=True)

            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            regular_mask = (values_sorted >= lower_bound) & (values_sorted <= upper_bound)
            outlier_mask = ~regular_mask

            regular_indices = np.where(regular_mask)[0]

            line = ax.plot(regular_indices, values_sorted[regular_mask],
                           color='blue', marker='o', markersize=5, linestyle='-', linewidth=1, label='Regular values')

            y_min, y_max = values_sorted.min(), values_sorted.max()
            y_range = y_max - y_min
            ax.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.2)

            if show_values:
                step = max(1, len(regular_indices) // 8)
                for i in range(0, len(regular_indices), step):
                    if i < len(regular_indices):
                        idx = regular_indices[i]
                        val = values_sorted[regular_mask].iloc[i]
                        ax.annotate(f'{val:.1f}', (idx, val),
                                    xytext=(0, 8), textcoords='offset points',
                                    bbox=dict(facecolor='white', edgecolor='black', alpha=0.7,
                                              boxstyle='round,pad=0.2'),
                                    ha='center', va='bottom', fontsize=label_size)

            outlier_indices = np.where(outlier_mask)[0]
            if len(outlier_indices) > 0:
                outliers = ax.scatter(outlier_indices, values_sorted[outlier_mask],
                                      color='red', marker='*', s=100, label='Outliers')

                if show_values:
                    for i, idx in enumerate(outlier_indices):
                        val = values_sorted[outlier_mask].iloc[i]
                        ax.annotate(f'{val:.1f}', (idx, val),
                                    xytext=(0, 10), textcoords='offset points',
                                    bbox=dict(facecolor='yellow', edgecolor='black', alpha=0.9,
                                              boxstyle='round,pad=0.2'),
                                    ha='center', va='bottom', fontsize=label_size, fontweight='bold')

            ax.set_xlabel('Index', fontsize=10)
            ax.set_ylabel(column, fontsize=10)
            ax.legend()

            stats_text = f"Min: {values.min():.2f}, Max: {values.max():.2f}, Avg: {values.mean():.2f}"
            ax.text(0.5, 0.02, stats_text, transform=ax.transAxes, ha='center',
                    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2'), fontsize=9)
        else:
            ax.text(0.5, 0.5, f'Line Plot not applicable for categorical column "{column}".',
                    ha='center', va='center', fontsize=10)

    elif chart_type == "Pie Chart":
        if not pd.api.types.is_numeric_dtype(df[column]):
            counts = df[column].dropna().value_counts()

            if len(counts) > 6:
                others = pd.Series([counts[6:].sum()], index=['Others'])
                counts = pd.concat([counts[:6], others])

            pie_colors = colors[:len(counts)]

            wedges, texts, autotexts = ax.pie(
                counts,
                labels=None,
                autopct='%1.1f%%' if show_values else None,
                colors=pie_colors,
                startangle=90,
                shadow=True,
                wedgeprops=dict(width=0.5, edgecolor='w')
            )

            if show_values:
                for autotext in autotexts:
                    autotext.set_weight('bold')
                    autotext.set_fontsize(label_size)
                    autotext.set_bbox(dict(facecolor='white', edgecolor='black', alpha=0.7, boxstyle='round,pad=0.2'))

                labels = [f'{name}: {count} ({count / counts.sum() * 100:.1f}%)' for name, count in counts.items()]
                ax.legend(wedges, labels, title=column, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)

                plt.subplots_adjust(right=0.65)
        else:
            ax.text(0.5, 0.5, f'Pie Chart not applicable for numeric column "{column}".',
                    ha='center', va='center', fontsize=10)

    plt.tight_layout()

    return fig


def display_plot_in_new_window(fig, title="Plot Viewer"):
    plot_layout = [
        [sg.Canvas(key="-PLOT-CANVAS-", size=(700, 500))],
        [sg.Button("Save Image", key="-SAVE-PLOT-IMG-"), sg.Button("Close", key="-CLOSE-PLOT-")]
    ]

    plot_window = sg.Window(title, plot_layout, finalize=True, resizable=True, modal=True)

    figure_canvas_agg = FigureCanvasTkAgg(fig, plot_window["-PLOT-CANVAS-"].TKCanvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)

    while True:
        plot_event, plot_values = plot_window.read()
        if plot_event in (sg.WIN_CLOSED, "-CLOSE-PLOT-"):
            break
        elif plot_event == "-SAVE-PLOT-IMG-":
            file_path = sg.popup_get_file('Save plot as image', save_as=True,
                                          file_types=(("PNG Files", "*.png"), ("JPEG Files", "*.jpg"),
                                                      ("PDF Files", "*.pdf"), ("All Files", "*.*")),
                                          default_extension=".png")
            if file_path:
                try:
                    if not any(file_path.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.pdf']):
                        file_path += '.png'

                    fig.savefig(file_path, dpi=300, bbox_inches='tight')
                    sg.popup(f"Image saved to {file_path}")
                except Exception as e:
                    sg.popup(f"Error saving image: {e}")

    plot_window.close()
    plt.close(fig)


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
                print(f"Removing rows at indices: {valid_indices}")
                result_df = result_df.drop(result_df.index[valid_indices])

            print(f"After row operation, dataframe has {len(result_df)} rows")

        except Exception as e:
            print(f"Error processing row indices: {e}")
            sg.popup(f"Error processing row indices: {e}")
            return None

    if col_indices is not None:
        try:
            print(f"Raw column indices: {col_indices}")

            col_names = []
            for c in col_indices:
                if isinstance(c, int) and 0 <= c < len(result_df.columns):
                    col_names.append(result_df.columns[c])
                elif isinstance(c, str) and c in result_df.columns:
                    col_names.append(c)
                elif isinstance(c, str) and '-' in c:
                    try:
                        start, end = map(int, c.split('-'))
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
                print(f"Keeping columns: {col_names}")
                result_df = result_df[col_names]
            else:
                print(f"Removing columns: {col_names}")
                result_df = result_df.drop(columns=col_names)

            print(f"After column operation, dataframe has {len(result_df.columns)} columns")

        except Exception as e:
            print(f"Error processing column indices: {e}")
            sg.popup(f"Error processing column indices: {e}")
            return None

    return result_df


def update_table_headers(window, table_key, df, prefix_col="Index"):
    try:
        window[table_key].update(visible=False)

        headers = [prefix_col] + list(df.columns)

        window[table_key].ColumnHeadings = headers

        window[table_key].update(visible=True)

        return True
    except Exception as e:
        print(f"Error updating {table_key} headers: {e}")
        import traceback
        traceback.print_exc()
        return False


def dataframe_to_table_data(df, max_rows=None):
    if max_rows is not None:
        df = df.head(max_rows)

    table_data = []
    for i, row in df.iterrows():
        row_data = [str(i)]

        for val in row:
            if isinstance(val, (float, int)):
                row_data.append(str(round(val, 4)) if val % 1 != 0 else str(int(val)))
            else:
                row_data.append(str(val))

        table_data.append(row_data)

    return table_data


def remove_columns(df, cols_to_remove):
    for c in cols_to_remove:
        if c in df.columns:
            df = df.drop(columns=c)
    return df


def replace_values(df, column, old_value, new_value):
    if column not in df.columns:
        sg.popup(f"Column '{column}' does not exist!")
        return df

    df = df.copy()

    if old_value not in df[column].values and not (pd.api.types.is_numeric_dtype(df[column]) and
                                                   str(old_value).replace('.', '', 1).isdigit()):
        sg.popup(f"Old value '{old_value}' not found in column '{column}'.")
        return df

    try:
        if pd.api.types.is_numeric_dtype(df[column]):
            try:
                old_value_numeric = float(old_value) if '.' in old_value else int(old_value)

                if isinstance(old_value_numeric, float):
                    mask = np.isclose(df[column], old_value_numeric)
                else:
                    mask = df[column] == old_value_numeric

                if not str(new_value).replace('.', '', 1).replace('-', '', 1).isdigit():
                    df[column] = df[column].astype(str)

                df.loc[mask, column] = new_value
                print(f"Replaced '{old_value_numeric}' with '{new_value}' in column '{column}'")
            except ValueError:
                mask = df[column].astype(str) == str(old_value)
                df.loc[mask, column] = new_value
        elif isinstance(df[column].dtype, pd.CategoricalDtype):
            df[column] = df[column].astype(str)
            mask = df[column] == str(old_value)
            df.loc[mask, column] = new_value
            df[column] = df[column].astype('category')
        else:
            mask = df[column].astype(str) == str(old_value)
            df.loc[mask, column] = new_value

        if all(str(val).replace('.', '', 1).replace('-', '', 1).isdigit()
               for val in df[column].dropna().unique()):
            try:
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

    df = df.copy()

    try:
        if pd.api.types.is_categorical_dtype(df[column]):
            df[column] = df[column].astype(str)
            df[column] = new_value
        else:
            df[column] = new_value

        if all(str(val).replace('.', '', 1).replace('-', '', 1).isdigit()
               for val in df[column].dropna().unique()):
            try:
                df[column] = pd.to_numeric(df[column], errors='ignore')
            except:
                pass

        return df
    except Exception as e:
        print(f"Error in replace_all_values: {e}")
        import traceback
        traceback.print_exc()
        sg.popup(f"Error replacing values: {e}")
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

    dummies = pd.get_dummies(df[column], prefix=column, drop_first=True)
    result_df = pd.concat([df.drop(columns=[column]), dummies], axis=1)
    return result_df


def binary_encoding(df, column):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in the dataframe")

    df = df.copy()
    df[column] = df[column].astype('category')
    df[f"{column}_encoded"] = df[column].cat.codes
    return df


def target_encoding(df, column, target):
    if column not in df.columns or target not in df.columns:
        raise ValueError(f"Column '{column}' or target '{target}' not found in the dataframe")

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

    df_copy = df.copy()

    for col in chosen_cols:
        df_copy[col] = df_copy[col].fillna(df_copy[col].mean())

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


# Tab 0: Logo & Creators Info
tab0_layout = [
    [sg.Text("", size=(1, 1))],
    [sg.Column([
        [sg.Text("Subject: Data Warehousing", font=("Helvetica", 26), justification='center')],
        [sg.Image(filename="assets/logo.png", key="-LOGO-", size=(250, 250))],
        [sg.Text("Created by:", font=("Helvetica", 26), justification='center')],
        [sg.Text("Dawid Olko", font=("Helvetica", 24), justification='center')],
        [sg.Text("Piotr Smoła", font=("Helvetica", 24), justification='center')],
        [sg.Button("Go to data", key="-ENTER-", font=("Helvetica", 24), size=(35, 1), pad=((30, 30), (30, 30)))]
    ], justification='center', element_justification='center')],
    [sg.Text("", size=(1, 1))],
]

# Tab 1: Data & Statistics
tab1_layout = [
    [sg.Column([
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
        ], size=(1120, 120))],

        [sg.Multiline(size=(157, 6), key="-DATA_INFO-", disabled=True)],

        [sg.Frame("Data Statistics", [
            [sg.Button("Compute Stats", key="-STATS-"), sg.Button("Correlation", key="-CORR-")],

            [sg.Text("Numeric Columns Statistics:")],
            [sg.Table(values=[],
                      headings=["Column", "Min", "Max", "Mean", "Median", "Std", "Mode", "Variance", "Skewness",
                                "Kurtosis"],
                      key="-NUMERIC_STATS-",
                      auto_size_columns=False,
                      col_widths=[15, 10, 10, 10, 10, 10, 10, 10, 10, 10],
                      justification='center',
                      expand_x=True,
                      expand_y=True,
                      size=(1100, 10))],

            [sg.Text("Categorical Columns Statistics:")],
            [sg.Table(values=[],
                      headings=["Column", "Value Counts", "Mode"],
                      key="-CATEGORICAL_STATS-",
                      auto_size_columns=False,
                      col_widths=[15, 50, 15],
                      justification='center',
                      expand_x=True,
                      expand_y=True,
                      size=(1100, 10))]
        ], expand_x=True)],
    ], scrollable=True, vertical_scroll_only=False, expand_x=True, expand_y=True, size=(1180, 750))]
]

# Tab 2: Extract Subtable
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

        [sg.Text("Selected Columns:"), sg.Input("", key="-SELECTED_COLS-", size=(50, 1), readonly=True)],

        [sg.Button("Extract Subtable", key="-EXTRACT_BTN-"),
         sg.Button("Save Subtable", key="-SAVE_SUBTABLE-")],

    ], size=(1120, 200))],

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
    ], size=(1120, 245))]
]

# Tab 3: Scaling & Visualization
tab3_layout = [
    [sg.Frame("Data Scaling", [
        [sg.Text("Columns to Scale:"),
         sg.Combo([], key="-SCALE_COL_SELECT-", size=(30, 1)),
         sg.Button("Add", key="-ADD_SCALE_COL-"),
         sg.Button("Clear", key="-CLEAR_SCALE_COLS-")],

        [sg.Text("Selected Columns:"), sg.Input("", key="-SELECTED_SCALE_COLS-", size=(50, 1), readonly=True)],

        [sg.Radio("StandardScaler", "SCALER", default=True, key="-STD_SCALER-"),
         sg.Radio("MinMaxScaler", "SCALER", key="-MINMAX_SCALER-")],

        [sg.Button("Apply Scaling", key="-APPLY_SCALING-"),
         sg.Button("Save Scaled Data", key="-SAVE_SCALED-"),
         sg.Button("Restore Original", key="-RESTORE_SCALED-")],
    ], size=(1200, 150))],

    [sg.Column([
        [sg.Frame("Data Visualization", [
            [sg.Text("Select Column to Plot:"),
             sg.Combo([], key="-PLOT_SELECT-", size=(30, 1))],

            [sg.Text("Select Chart Type:")],
            [sg.Radio("Histogram", "CHART", default=True, key="-CHART_HIST-"),
             sg.Radio("Boxplot", "CHART", key="-CHART_BOX-"),
             sg.Radio("Bar Chart", "CHART", key="-CHART_BAR-"),
             sg.Radio("Line Plot", "CHART", key="-CHART_LINE-"),
             sg.Radio("Pie Chart", "CHART", key="-CHART_PIE-")],

            [sg.Text("Chart Options:")],
            [sg.Checkbox("Show Value Labels", default=True, key="-SHOW_VALUES-"),
             sg.Checkbox("Show Grid Lines", default=True, key="-SHOW_GRID-")],

            [sg.Text("Label Size:"),
             sg.Slider(range=(6, 16), default_value=9, orientation='h', size=(15, 15), key="-LABEL_SIZE-"),
             sg.Text("Chart Title:"),
             sg.Input(key="-CHART_TITLE-", size=(25, 1))],

            [sg.Text("Color Theme:"),
             sg.Combo(values=["Default", "Pastel", "Dark", "Colorblind", "Grayscale"],
                      default_value="Default", key="-COLOR_THEME-", size=(15, 1))],

            [sg.Button("Generate Plot", key="-PLOT_BTN-"),
             sg.Button("Save Plot as Image", key="-SAVE_PLOT-")],

            [sg.Column([
                [sg.Canvas(key="-CANVAS-", size=(1100, 20))]
            ], vertical_scroll_only=True, size=(1200, 20))]
        ])]
    ])]
]

# Tab 4: Data Cleaning & Transformation
tab4_layout = [
    [sg.Frame("Handling Missing Values", [
        [sg.Combo(["remove", "fill_mean", "fill_median", "fill_mode"], default_value="remove",
                  key="-MISSING_STRATEGY-"),
         sg.Button("Apply Missing Values Handling", key="-APPLY_MISSING-"),
         sg.Button("Save Cleaned Data", key="-SAVE_CLEANED-")]
    ], size=(1200, 50))],

    [sg.Frame("Duplicates", [
        [sg.Button("Remove Duplicates", key="-REMOVE_DUPLICATES-"),
         sg.Button("Save After Duplicate Removal", key="-SAVE_DEDUP-")]
    ], size=(1200, 50))],

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
    ], size=(1200, 150))],
]

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
            size=(1200, 800))]
    ], scrollable=False, expand_x=True, expand_y=True)]
]

window = sg.Window("DataFusion - Project",
                   layout,
                   size=(WINDOW_WIDTH, WINDOW_HEIGHT),
                   resizable=False,
                   finalize=True,
                   element_justification='center')

set_window_icon(window, "assets/logo.png")

try:
    root = window.TKroot

    root.resizable(False, False)

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - WINDOW_WIDTH) // 2
    y = (screen_height - WINDOW_HEIGHT) // 2
    root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{x}+{y}")

    if os.name == 'nt':
        import ctypes
        from ctypes import windll

        hwnd = windll.user32.GetParent(root.winfo_id())

        style = windll.user32.GetWindowLongW(hwnd, -16)
        style = style & ~0x10000
        windll.user32.SetWindowLongW(hwnd, -16, style)
except Exception as e:
    print(f"Błąd podczas konfiguracji okna: {e}")

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED:
        break

    if event == "-ENTER-":
        window['-TABGROUP-'].Widget.select(1)

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
                file_path = values.get("-FILE_PATH-", "")

                if not file_path:
                    file_path = sg.popup_get_file('Choose CSV file',
                                                  file_types=(("CSV Files", "*.csv"),))

                    if file_path:
                        window["-FILE_PATH-"].update(file_path)
                        window["-FILE_PATH-"].update(visible=True)
                    else:
                        continue

                df = load_dataset(file_path, is_predefined=False)

            window["-DATA_INFO-"].update(f"Dataset loaded with {len(df)} rows and {len(df.columns)} columns:\n" +
                                         ", ".join(df.columns))
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
        window_title = f"Save Current Data_{hash(str(df))}"
        save_window = create_save_options_window(window_title)

        while True:
            save_event, save_values = save_window.read()
            if save_event in (sg.WIN_CLOSED, "-CANCEL_SAVE-"):
                break

            elif save_event == "-CONFIRM_SAVE-":
                unique_suffix = f"_{id(window_title)}_{window_title}"
                include_index = save_values[f"-SAVE_INCLUDE_INDEX{unique_suffix}"]
                include_header = save_values[f"-SAVE_INCLUDE_HEADER{unique_suffix}"]
                save_dataframe_to_csv(df, include_index, include_header)
                break
        save_window.close()
        del save_window

    elif event == "-STATS-":
        if df is None:
            sg.popup("Please load the dataset first.")
        else:
            try:
                dataset_type = values["-SELECT_FILE-"] if values["-PREDEFINED_DATASET-"] else "Custom"
                stats_data = compute_statistics(df, dataset_type)

                numeric_stats = []
                categorical_stats = []
                for stat in stats_data:
                    if len(stat) > 2 and isinstance(stat[1], (int, float)) and isinstance(stat[2], (int, float)):
                        numeric_stats.append(stat)
                    elif len(stat) <= 3:
                        categorical_stats.append(stat)

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
            try:
                pearson_corr, spearman_corr = compute_correlation(df)
                if pearson_corr is None or spearman_corr is None or pearson_corr.empty or spearman_corr.empty:
                    sg.popup("Not enough numeric columns for correlation or error in computation.")
                else:
                    numeric_cols = list(pearson_corr.columns)
                    pearson_data = []
                    for idx, row in pearson_corr.iterrows():
                        row_data = [idx]
                        for col in numeric_cols:
                            row_data.append(round(row[col], 3))
                        pearson_data.append(row_data)
                    spearman_data = []
                    for idx, row in spearman_corr.iterrows():
                        row_data = [idx]
                        for col in numeric_cols:
                            row_data.append(round(row[col], 3))
                        spearman_data.append(row_data)
                    headers = ["Column"] + numeric_cols
                    try:
                        correlation_frame = None
                        for element in window.element_list():
                            if isinstance(element, sg.Frame) and element.Title == "Correlation Results":
                                correlation_frame = element
                                break
                        if correlation_frame:
                            correlation_frame.update(visible=False)
                            new_correlation_layout = [
                                [sg.Text("Pearson Correlation Matrix:")],
                                [sg.Table(values=pearson_data,
                                          headings=headers,
                                          key="-PEARSON_CORR_NEW-",
                                          auto_size_columns=False,
                                          col_widths=[15] + [10] * (len(headers) - 1),
                                          justification='center',
                                          expand_x=True,
                                          size=(800, len(pearson_data)),
                                          display_row_numbers=False)],
                                [sg.Text("Spearman Correlation Matrix:")],
                                [sg.Table(values=spearman_data,
                                          headings=headers,
                                          key="-SPEARMAN_CORR_NEW-",
                                          auto_size_columns=True,
                                          justification='center',
                                          expand_x=True,
                                          size=(800, len(spearman_data)),
                                          display_row_numbers=False)]
                            ]
                            new_correlation_frame = sg.Frame("Correlation Results", new_correlation_layout,
                                                             expand_x=True)
                            corr_window = sg.Window("Correlation Results",
                                                    [[new_correlation_frame]],
                                                    modal=True,
                                                    finalize=True,
                                                    resizable=True)
                            while True:
                                corr_event, corr_values = corr_window.read()
                                if corr_event == sg.WIN_CLOSED:
                                    break
                            corr_window.close()
                            correlation_frame.update(visible=True)
                        else:
                            corr_window = sg.Window("Correlation Results",
                                                    [[sg.Frame("Correlation Results", [
                                                        [sg.Text("Pearson Correlation Matrix:")],
                                                        [sg.Column([
                                                            [sg.Table(values=pearson_data,
                                                                      headings=headers,
                                                                      key="-PEARSON_CORR_NEW-",
                                                                      auto_size_columns=False,
                                                                      col_widths=[15] + [10] * (len(headers) - 1),
                                                                      justification='center',
                                                                      expand_x=True,
                                                                      size=(800, len(pearson_data)),
                                                                      display_row_numbers=False)]
                                                        ], scrollable=True, vertical_scroll_only=False, expand_x=True,
                                                            size=(820, min(400, len(pearson_data) * 25 + 40)))],
                                                        [sg.Text("Spearman Correlation Matrix:")],
                                                        [sg.Column([
                                                            [sg.Table(values=spearman_data,
                                                                      headings=headers,
                                                                      key="-SPEARMAN_CORR_NEW-",
                                                                      auto_size_columns=False,
                                                                      col_widths=[15] + [10] * (len(headers) - 1),
                                                                      justification='center',
                                                                      expand_x=True,
                                                                      size=(800, len(spearman_data)),
                                                                      display_row_numbers=False)]
                                                        ], scrollable=True, vertical_scroll_only=False, expand_x=True,
                                                            size=(820, min(400, len(spearman_data) * 25 + 40)))]
                                                    ])]],
                                                    modal=True,
                                                    finalize=True,
                                                    resizable=True)
                            while True:
                                corr_event, corr_values = corr_window.read()
                                if corr_event == sg.WIN_CLOSED:
                                    break
                            corr_window.close()
                    except Exception as e:
                        print(f"Error updating correlation frame: {e}")
                        import traceback

                        traceback.print_exc()
                        corr_window = sg.Window("Correlation Results",
                                                [[sg.Frame("Correlation Results", [
                                                    [sg.Text("Pearson Correlation Matrix:")],
                                                    [sg.Table(values=pearson_data,
                                                              headings=headers,
                                                              key="-PEARSON_CORR_NEW-",
                                                              auto_size_columns=True,
                                                              justification='center',
                                                              expand_x=True,
                                                              size=(800, len(pearson_data)),
                                                              display_row_numbers=False)],
                                                    [sg.Text("Spearman Correlation Matrix:")],
                                                    [sg.Table(values=spearman_data,
                                                              headings=headers,
                                                              key="-SPEARMAN_CORR_NEW-",
                                                              auto_size_columns=True,
                                                              justification='center',
                                                              expand_x=True,
                                                              size=(800, len(spearman_data)),
                                                              display_row_numbers=False)]
                                                ])]],
                                                modal=True,
                                                finalize=True,
                                                resizable=True)
                        while True:
                            corr_event, corr_values = corr_window.read()
                            if corr_event == sg.WIN_CLOSED:
                                break
                        corr_window.close()
            except Exception as e:
                print(f"Error in correlation calculation: {e}")
                import traceback

                traceback.print_exc()
                sg.popup(f"Error calculating correlation: {str(e)}")

    elif event == "-ADD_COL-":
        if df is None:
            sg.popup("Please load the dataset first.")
            continue

        selected_col = values.get("-COL_SELECT-", "")
        if not selected_col:
            continue

        current_cols = values.get("-SELECTED_COLS-", "").strip()
        if current_cols:
            updated_cols = current_cols + ", " + selected_col
        else:
            updated_cols = selected_col

        if "-SELECTED_COLS-" in window.key_dict:
            window["-SELECTED_COLS-"].update(updated_cols)
        else:
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
                row_input = values["-ROW_INPUT-"].strip()
                col_input = values["-SELECTED_COLS-"].strip()
                row_indices = None

                if row_input:
                    row_indices = [part.strip() for part in row_input.split(",") if part.strip()]
                    row_indices = [int(i) if i.isdigit() else i for i in row_indices]
                    print(f"Parsed row indices: {row_indices}")
                col_indices = None

                if col_input:
                    col_indices = [part.strip() for part in col_input.split(",") if part.strip()]
                    col_indices = [int(i) if i.isdigit() else i for i in col_indices]
                    print(f"Parsed column indices: {col_indices}")
                keep = values["-KEEP_EXTRACT-"]
                sub_df = extract_subtable(df, row_indices, col_indices, keep)

                if sub_df is None or sub_df.empty:
                    window["-EXTRACT_OUT-"].update(values=[[]])
                    sg.popup("Invalid range or empty subtable.")
                else:
                    display_df = sub_df.head(100)

                    table_data = []
                    for i, row in display_df.iterrows():
                        row_data = [str(i)]
                        for val in row:
                            if isinstance(val, (float, int)):
                                row_data.append(str(round(val, 4)) if val % 1 != 0 else str(int(val)))
                            else:
                                row_data.append(str(val))
                        table_data.append(row_data)

                    headers = ["Index"] + list(display_df.columns)

                    subtable_layout = [
                        [sg.Text(f"Subtable Results (showing {len(table_data)} of {len(sub_df)} rows)")],
                        [sg.Table(
                            values=table_data,
                            headings=headers,
                            key="-SUBTABLE_RESULT-",
                            auto_size_columns=True,
                            justification='center',
                            expand_x=True,
                            num_rows=min(20, len(table_data)),
                            display_row_numbers=False
                        )],
                        [sg.Button("Close", key="-CLOSE_SUBTABLE-")]
                    ]

                    subtable_window = sg.Window(
                        "Subtable Results",
                        [[sg.Frame("Extracted Subtable", subtable_layout)]],
                        modal=True,
                        finalize=True,
                        resizable=True,
                        size=(800, 600)
                    )

                    while True:
                        subtable_event, subtable_values = subtable_window.read()
                        if subtable_event in (sg.WIN_CLOSED, "-CLOSE_SUBTABLE-"):
                            break

                    subtable_window.close()

                    df = sub_df

            except Exception as e:
                print(f"Error in subtable extraction: {e}")
                import traceback

                traceback.print_exc()
                sg.popup(f"Error extracting subtable: {e}")

    elif event == "-SAVE_SUBTABLE-":
        window_title = f"Save Subtable_{hash(str(df))}"
        save_window = create_save_options_window(window_title)
        while True:
            save_event, save_values = save_window.read()
            if save_event in (sg.WIN_CLOSED, "-CANCEL_SAVE-"):
                break
            elif save_event == "-CONFIRM_SAVE-":
                unique_suffix = f"_{id(window_title)}_{window_title}"
                include_index = save_values[f"-SAVE_INCLUDE_INDEX{unique_suffix}"]
                include_header = save_values[f"-SAVE_INCLUDE_HEADER{unique_suffix}"]
                save_dataframe_to_csv(df, include_index, include_header)
                break

        save_window.close()
        del save_window
    elif event == "-GET_VALUES-":
        if df is None:
            sg.popup("Please load the dataset first.")
            continue
        selected_col = values["-REPLACE_COL-"]
        if not selected_col or selected_col not in df.columns:
            sg.popup("Please select a valid column.")
            continue
        unique_values = df[selected_col].dropna().unique().tolist()

        if pd.api.types.is_numeric_dtype(df[selected_col]):
            unique_values.sort()

        else:
            try:
                unique_values.sort()

            except TypeError:
                unique_values = [str(val) for val in unique_values]
                unique_values.sort()

        unique_values = [str(val) for val in unique_values]
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
                    temp_df = df.copy()
                    df = replace_values(df, col_to_replace, old_value, new_value)
                    changes_made = not df[col_to_replace].equals(temp_df[col_to_replace])

                    if changes_made:
                        display_df = df.head(100)

                        table_data = []
                        for i, row in display_df.iterrows():
                            row_data = [str(i)]
                            for val in row:
                                if isinstance(val, (float, int)):
                                    row_data.append(str(round(val, 4)) if val % 1 != 0 else str(int(val)))
                                else:
                                    row_data.append(str(val))
                            table_data.append(row_data)

                        headers = ["Index"] + list(display_df.columns)

                        replace_layout = [
                            [sg.Text(
                                f"Replacement Results: '{old_value}' -> '{new_value}' in column '{col_to_replace}'")],
                            [sg.Table(
                                values=table_data,
                                headings=headers,
                                key="-REPLACE_RESULT-",
                                auto_size_columns=True,
                                justification='center',
                                expand_x=True,
                                num_rows=min(20, len(table_data)),
                                display_row_numbers=False
                            )],
                            [sg.Button("Close", key="-CLOSE_REPLACE-")]
                        ]

                        replace_window = sg.Window(
                            "Replace Results",
                            [[sg.Frame("Data After Replacement", replace_layout)]],
                            modal=True,
                            finalize=True,
                            resizable=True,
                            size=(800, 600)
                        )

                        while True:
                            replace_event, replace_vals = replace_window.read()
                            if replace_event in (sg.WIN_CLOSED, "-CLOSE_REPLACE-"):
                                break

                        replace_window.close()

                        sg.popup(f"Replaced '{old_value}' with '{new_value}' in column '{col_to_replace}'")
                        unique_values = df[col_to_replace].dropna().unique().tolist()
                        unique_values = [str(val) for val in unique_values]
                        window["-OLD_VAL-"].update(values=unique_values)
                    else:
                        sg.popup(f"No values '{old_value}' found in column '{col_to_replace}' to replace.")
                except Exception as e:
                    print(f"Error replacing values: {e}")
                    import traceback

                    traceback.print_exc()
                    sg.popup(f"Error replacing values: {e}")

    elif event == "-SAVE_REPLACED-":
        window_title = f"Save After Replacement_{hash(str(df))}"
        save_window = create_save_options_window(window_title)
        while True:
            save_event, save_values = save_window.read()
            if save_event in (sg.WIN_CLOSED, "-CANCEL_SAVE-"):
                break
            elif save_event == "-CONFIRM_SAVE-":
                unique_suffix = f"_{id(window_title)}_{window_title}"
                include_index = save_values[f"-SAVE_INCLUDE_INDEX{unique_suffix}"]
                include_header = save_values[f"-SAVE_INCLUDE_HEADER{unique_suffix}"]
                save_dataframe_to_csv(df, include_index, include_header)
                break
        save_window.close()
        del save_window

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
                    if col_to_replace_all in df.columns:
                        display_df = df.head(100)

                        table_data = []
                        for i, row in display_df.iterrows():
                            row_data = [str(i)]
                            for val in row:
                                if isinstance(val, (float, int)):
                                    row_data.append(str(round(val, 4)) if val % 1 != 0 else str(int(val)))
                                else:
                                    row_data.append(str(val))
                            table_data.append(row_data)

                        headers = ["Index"] + list(display_df.columns)

                        replace_all_layout = [
                            [sg.Text(
                                f"Replacement Results: All values in '{col_to_replace_all}' -> '{new_value_all}'")],
                            [sg.Table(
                                values=table_data,
                                headings=headers,
                                key="-REPLACE_ALL_RESULT-",
                                auto_size_columns=True,
                                justification='center',
                                expand_x=True,
                                num_rows=min(20, len(table_data)),
                                display_row_numbers=False
                            )],
                            [sg.Button("Close", key="-CLOSE_REPLACE_ALL-")]
                        ]

                        replace_all_window = sg.Window(
                            "Replace All Results",
                            [[sg.Frame("Data After Replacement", replace_all_layout)]],
                            modal=True,
                            finalize=True,
                            resizable=True,
                            size=(800, 600)
                        )

                        while True:
                            replace_all_event, replace_all_vals = replace_all_window.read()
                            if replace_all_event in (sg.WIN_CLOSED, "-CLOSE_REPLACE_ALL-"):
                                break

                        replace_all_window.close()

                        sg.popup(f"Replaced all values in column '{col_to_replace_all}' with '{new_value_all}'")
                    else:
                        sg.popup(f"Column '{col_to_replace_all}' does not exist in the dataframe.")
                except Exception as e:
                    print(f"Error replacing all values: {e}")
                    import traceback

                    traceback.print_exc()
                    sg.popup(f"Error replacing all values: {e}")

    elif event == "-ADD_SCALE_COL-":
        if df is None:
            sg.popup("Please load the dataset first.")
            continue

        selected_col = values.get("-SCALE_COL_SELECT-", "")
        if not selected_col:
            continue

        current_cols = values.get("-SELECTED_SCALE_COLS-", "").strip()
        if current_cols:
            updated_cols = current_cols + ", " + selected_col
        else:
            updated_cols = selected_col

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
            cols_list = [c.strip() for c in cols_str.split(",") if c.strip() != ""]

            if not cols_list:
                sg.popup("Please enter column names to scale, separated by commas.")
                continue

            method = "standard" if values["-STD_SCALER-"] else "minmax"

            try:
                scaled_df = scale_columns(df, cols_list, method=method)

                for col in cols_list:
                    if col in scaled_df.columns and col in df.columns:
                        df[col] = scaled_df[col]

                preview_df = df[cols_list].head(100)

                table_data = []
                for i, row in preview_df.iterrows():
                    row_data = [str(i)]
                    for val in row:
                        if isinstance(val, (float, int)):
                            row_data.append(str(round(val, 4)) if val % 1 != 0 else str(int(val)))
                        else:
                            row_data.append(str(val))
                    table_data.append(row_data)

                headers = ["Index"] + list(preview_df.columns)

                scaling_layout = [
                    [sg.Text(f"Scaling Results using {method.capitalize()} Scaler")],
                    [sg.Table(
                        values=table_data,
                        headings=headers,
                        key="-SCALING_RESULT-",
                        auto_size_columns=True,
                        justification='center',
                        expand_x=True,
                        num_rows=min(20, len(table_data)),
                        display_row_numbers=False
                    )],
                    [sg.Button("Close", key="-CLOSE_SCALING-")]
                ]

                scaling_window = sg.Window(
                    "Scaling Results",
                    [[sg.Frame(f"{method.capitalize()} Scaling Results", scaling_layout)]],
                    modal=True,
                    finalize=True,
                    resizable=True,
                    size=(800, 500)
                )

                while True:
                    scaling_event, scaling_values = scaling_window.read()
                    if scaling_event in (sg.WIN_CLOSED, "-CLOSE_SCALING-"):
                        break

                scaling_window.close()

                sg.popup(f"Applied {method} scaling to columns: {', '.join(cols_list)}")

            except Exception as e:
                print(f"Error scaling columns: {e}")
                import traceback

                traceback.print_exc()
                sg.popup(f"Error scaling columns: {str(e)}")

    elif event == "-SAVE_SCALED-":
        window_title = f"Save Scaled Data_{hash(str(df))}"
        save_window = create_save_options_window(window_title)
        while True:
            save_event, save_values = save_window.read()
            if save_event in (sg.WIN_CLOSED, "-CANCEL_SAVE-"):
                break
            elif save_event == "-CONFIRM_SAVE-":
                unique_suffix = f"_{id(window_title)}_{window_title}"
                include_index = save_values[f"-SAVE_INCLUDE_INDEX{unique_suffix}"]
                include_header = save_values[f"-SAVE_INCLUDE_HEADER{unique_suffix}"]
                save_dataframe_to_csv(df, include_index, include_header)
                break
        save_window.close()
        del save_window
    elif event == "-RESTORE_SCALED-":
        if original_df is not None:
            df = restore_original_data()

            if "SELECTED_SCALE_COLS" in values and values["-SELECTED_SCALE_COLS-"]:
                cols_to_show = [c.strip() for c in values["-SELECTED_SCALE_COLS-"].split(",") if c.strip() != ""]
            else:
                cols_to_show = list(df.columns[:min(5, len(df.columns))])
            preview_df = df[cols_to_show].head(100)
            data_rows = [preview_df.index.astype(str).tolist()] + preview_df.values.tolist()
            table_data = [[str(data_rows[0][i])] + [str(round(val, 4)) if isinstance(val, (int, float)) else str(val)
                                                    for val in row]
                          for i, row in enumerate(data_rows[1:])]
            window["-SCALED_DATA-"].update(values=table_data)

            try:
                window["-SCALED_DATA-"].ColumnHeadings = ["Index"] + list(preview_df.columns)
            except:
                pass
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

            plot_options = {
                "show_values": values["-SHOW_VALUES-"],
                "show_grid": values["-SHOW_GRID-"],
                "label_size": int(values["-LABEL_SIZE-"]),
                "chart_title": values["-CHART_TITLE-"] or f"{chart_type} of {col_to_plot}",
                "color_theme": values["-COLOR_THEME-"]
            }

            try:
                fig = generate_plot(df, col_to_plot, chart_type, plot_options)
                display_plot_in_new_window(fig, title=f"{chart_type} of {col_to_plot}")
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
                if not file_path.lower().endswith('.png'):
                    file_path += '.png'

                plt.savefig(file_path, dpi=300, bbox_inches='tight')
                sg.popup(f"Plot saved to {file_path}")
            except Exception as e:
                sg.popup(f"Error saving plot: {e}")

    elif event == "-APPLY_MISSING-":
        if df is None:
            sg.popup("Please load the dataset first.")
        else:
            strategy = values["-MISSING_STRATEGY-"]
            try:
                df = handle_missing_values(df, strategy)

                preview_df = df.head(100)

                table_data = []
                for i, row in preview_df.iterrows():
                    row_data = [str(i)]
                    for val in row:
                        if isinstance(val, (float, int)):
                            row_data.append(str(round(val, 4)) if val % 1 != 0 else str(int(val)))
                        else:
                            row_data.append(str(val))
                    table_data.append(row_data)

                headers = ["Index"] + list(preview_df.columns)

                cleaning_layout = [
                    [sg.Text(f"Missing Values Handling Results using '{strategy}' strategy")],
                    [sg.Column([
                        [sg.Table(values=table_data,
                                  headings=headers,
                                  key="-MISSING_RESULT-",
                                  auto_size_columns=False,
                                  col_widths=[10] + [15] * (len(headers) - 1),
                                  justification='center',
                                  expand_x=True,
                                  num_rows=min(20, len(table_data)),
                                  display_row_numbers=False)]
                    ], scrollable=True, vertical_scroll_only=False, expand_x=True,
                        size=(880, min(500, len(table_data) * 25 + 40)))],
                    [sg.Button("Close", key="-CLOSE_MISSING-")]
                ]

                cleaning_window = sg.Window(
                    "Missing Values Handling",
                    [[sg.Frame(f"Data After {strategy.replace('_', ' ').title()}", cleaning_layout)]],
                    modal=True,
                    finalize=True,
                    resizable=True,
                    size=(900, 500)
                )

                while True:
                    cleaning_event, cleaning_values = cleaning_window.read()
                    if cleaning_event in (sg.WIN_CLOSED, "-CLOSE_MISSING-"):
                        break

                cleaning_window.close()

                sg.popup(f"Missing values handled with strategy: {strategy}")

            except Exception as e:
                print(f"Error handling missing values: {e}")
                import traceback

                traceback.print_exc()
                sg.popup(f"Error handling missing values: {e}")

    elif event == "-SAVE_CLEANED-":
        window_title = f"Save Cleaned Data_{hash(str(df))}"
        save_window = create_save_options_window(window_title)
        while True:
            save_event, save_values = save_window.read()
            if save_event in (sg.WIN_CLOSED, "-CANCEL_SAVE-"):
                break

            elif save_event == "-CONFIRM_SAVE-":
                unique_suffix = f"_{id(window_title)}_{window_title}"
                include_index = save_values[f"-SAVE_INCLUDE_INDEX{unique_suffix}"]
                include_header = save_values[f"-SAVE_INCLUDE_HEADER{unique_suffix}"]
                save_dataframe_to_csv(df, include_index, include_header)
                break
        save_window.close()
        del save_window

    elif event == "-REMOVE_DUPLICATES-":
        if df is None:
            sg.popup("Please load the dataset first.")
        else:
            try:
                original_len = len(df)

                df = remove_duplicates(df)

                removed_count = original_len - len(df)

                preview_df = df.head(100)

                table_data = []
                for i, row in preview_df.iterrows():
                    row_data = [str(i)]
                    for val in row:
                        if isinstance(val, (float, int)):
                            row_data.append(str(round(val, 4)) if val % 1 != 0 else str(int(val)))
                        else:
                            row_data.append(str(val))
                    table_data.append(row_data)

                headers = ["Index"] + list(preview_df.columns)

                dedup_layout = [
                    [sg.Text(f"Duplicate Removal Results - Removed {removed_count} rows")],
                    [sg.Column([
                        [sg.Table(values=table_data,
                                  headings=headers,
                                  key="-DEDUP_RESULT-",
                                  auto_size_columns=False,
                                  col_widths=[10] + [15] * (len(headers) - 1),
                                  justification='center',
                                  expand_x=True,
                                  num_rows=min(20, len(table_data)),
                                  display_row_numbers=False)]
                    ], scrollable=True, vertical_scroll_only=False, expand_x=True, size=(880, min(500, len(table_data) * 25 + 40)))],
                    [sg.Button("Close", key="-CLOSE_DEDUP-")]
                ]

                dedup_window = sg.Window(
                    "Duplicate Removal",
                    [[sg.Frame("Data After Duplicate Removal", dedup_layout)]],
                    modal=True,
                    finalize=True,
                    resizable=True,
                    size=(900, 500)
                )

                while True:
                    dedup_event, dedup_values = dedup_window.read()
                    if dedup_event in (sg.WIN_CLOSED, "-CLOSE_DEDUP-"):
                        break

                dedup_window.close()

                sg.popup(f"Duplicate rows removed successfully. Removed {removed_count} rows.")

            except Exception as e:
                print(f"Error removing duplicates: {e}")
                import traceback

                traceback.print_exc()
                sg.popup(f"Error removing duplicates: {e}")

    elif event == "-SAVE_DEDUP-":
        window_title = f"Save After Duplicate Removal_{hash(str(df))}"
        save_window = create_save_options_window(window_title)
        while True:
            save_event, save_values = save_window.read()
            if save_event in (sg.WIN_CLOSED, "-CANCEL_SAVE-"):
                break
            elif save_event == "-CONFIRM_SAVE-":
                unique_suffix = f"_{id(window_title)}_{window_title}"
                include_index = save_values[f"-SAVE_INCLUDE_INDEX{unique_suffix}"]
                include_header = save_values[f"-SAVE_INCLUDE_HEADER{unique_suffix}"]
                save_dataframe_to_csv(df, include_index, include_header)
                break
        save_window.close()

        del save_window

    elif event == "-APPLY_ENCODING-":
        if df is None:
            sg.popup("Please load the dataset first.")
        else:
            column = values["-ENCODE_COL-"]
            if not column:
                sg.popup("Please select a column to encode")
                continue

            try:
                encoding_method = ""
                if values["-ONE_HOT-"]:
                    df = one_hot_encoding(df, column)
                    encoding_method = "One-Hot Encoding"
                elif values["-BINARY_ENCODE-"]:
                    df = binary_encoding(df, column)
                    encoding_method = "Binary Encoding"
                elif values["-TARGET_ENCODE-"]:
                    target_column = values["-TARGET_COL-"]
                    if not target_column:
                        sg.popup("Please select a target column for Target Encoding.")
                        continue
                    df = target_encoding(df, column, target_column)
                    encoding_method = f"Target Encoding (target: {target_column})"

                preview_df = df.head(100)

                table_data = []
                for i, row in preview_df.iterrows():
                    row_data = [str(i)]
                    for val in row:
                        if isinstance(val, (float, int)):
                            row_data.append(str(round(val, 4)) if val % 1 != 0 else str(int(val)))
                        else:
                            row_data.append(str(val))
                    table_data.append(row_data)

                headers = ["Index"] + list(preview_df.columns)

                encoding_layout = [
                    [sg.Text(f"Encoding Results: {encoding_method} applied to column '{column}'")],
                    [sg.Column([
                        [sg.Table(values=table_data,
                                  headings=headers,
                                  key="-ENCODING_RESULT-",
                                  auto_size_columns=False,
                                  col_widths=[10] + [15] * (len(headers) - 1),
                                  justification='center',
                                  expand_x=True,
                                  num_rows=min(20, len(table_data)),
                                  display_row_numbers=False)]
                    ], scrollable=True, vertical_scroll_only=False, expand_x=True, size=(980, min(500, len(table_data) * 25 + 40)))],
                    [sg.Button("Close", key="-CLOSE_ENCODING-")]
                ]

                encoding_window = sg.Window(
                    "Encoding Results",
                    [[sg.Frame(f"Data After {encoding_method}", encoding_layout)]],
                    modal=True,
                    finalize=True,
                    resizable=True,
                    size=(1000, 500)
                )

                while True:
                    encoding_event, encoding_values = encoding_window.read()
                    if encoding_event in (sg.WIN_CLOSED, "-CLOSE_ENCODING-"):
                        break

                encoding_window.close()

                all_columns = list(df.columns)
                window["-PLOT_SELECT-"].update(values=all_columns)
                window["-ENCODE_COL-"].update(values=all_columns)
                window["-TARGET_COL-"].update(values=all_columns)
                window["-REPLACE_COL-"].update(values=all_columns)
                window["-ALL_REPLACE_COL-"].update(values=all_columns)
                window["-COL_SELECT-"].update(values=all_columns)
                window["-SCALE_COL_SELECT-"].update(values=all_columns)

                sg.popup(f"{encoding_method} applied to column: {column}")

            except Exception as e:
                print(f"Error applying encoding: {e}")
                import traceback

                traceback.print_exc()
                sg.popup(f"Error applying encoding: {str(e)}")

    elif event == "-SAVE_ENCODED-":
        window_title = f"Save Encoded Data_{hash(str(df))}"
        save_window = create_save_options_window(window_title)

        while True:
            save_event, save_values = save_window.read()
            if save_event in (sg.WIN_CLOSED, "-CANCEL_SAVE-"):
                break
            elif save_event == "-CONFIRM_SAVE-":
                unique_suffix = f"_{id(window_title)}_{window_title}"
                include_index = save_values[f"-SAVE_INCLUDE_INDEX{unique_suffix}"]
                include_header = save_values[f"-SAVE_INCLUDE_HEADER{unique_suffix}"]
                save_dataframe_to_csv(df, include_index, include_header)
                break
        save_window.close()
        del save_window