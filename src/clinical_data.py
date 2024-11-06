import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

def filter_patient_info(clinical_data_df, patient_id):
    # Load the data from the Excel file
    if isinstance(clinical_data_df, str):
        data = pd.read_excel(clinical_data_df, 'dataset_info')
    else:
        data = clinical_data_df
    
    # Filter the data for the specified patient_id
    patient_data = data[data['patient_id'] == patient_id]
    
    # Display patient information
    if not patient_data.empty:
        return patient_data
    else:
        return None

def filter_clinical_data(clinical_data_df, **conditions):
    # Load the data from the Excel file
    if isinstance(clinical_data_df, str):
        data = pd.read_excel(clinical_data_df, 'dataset_info')
    else:
        data = clinical_data_df
    filtered_data = clinical_data_df
    for column, value in conditions.items():
        filtered_data = filtered_data[filtered_data[column] == value]
    return filtered_data

def unique_values_in_columns(clinical_data_df):
    # Load the data from the Excel file
    if isinstance(clinical_data_df, str):
        data = pd.read_excel(clinical_data_df, 'dataset_info')
    else:
        data = clinical_data_df
    for column in clinical_data_df.columns:
        unique_vals = clinical_data_df[column].unique()
        print(f"{column}: {len(unique_vals)} unique values, sample: {unique_vals[:5]}")

def group_and_aggregate(clinical_data_df, group_by_column, agg_column, agg_func='mean'):
    # Load the data from the Excel file
    if isinstance(clinical_data_df, str):
        data = pd.read_excel(clinical_data_df, 'dataset_info')
    else:
        data = clinical_data_df
    grouped_data = data.groupby(group_by_column)[agg_column].agg(agg_func)
    display(grouped_data)

def plot_categorical_counts(clinical_data_df, column):
    if isinstance(clinical_data_df, str):
        data = pd.read_excel(clinical_data_df, 'dataset_info')
    else:
        data = clinical_data_df
    plt.figure(figsize=(10, 6))
    data[column].value_counts().plot(kind='bar', color='salmon', edgecolor='black')
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.title(f"Counts of {column}")
    plt.show()

def plot_histogram(clinical_data_df, column, bins=20):
    # Load the data from the Excel file
    if isinstance(clinical_data_df, str):
        data = pd.read_excel(clinical_data_df, 'dataset_info')
    else:
        data = clinical_data_df
    plt.figure(figsize=(10, 6))
    plt.hist(data[column].dropna(), bins=bins, color='skyblue', edgecolor='black')
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {column}")
    plt.show()

def plot_comparison(data, numeric_column, category_column):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=category_column, y=numeric_column, data=data, palette="Set2")
    plt.title(f"{numeric_column} by {category_column}")
    plt.show()
