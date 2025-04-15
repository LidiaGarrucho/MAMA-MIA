"""
scoring_prediction.py

This script evaluates classification predictions (Task 2) for the MAMA-MIA Challenge.
It computes performance and fairness metrics, generates subgroup-wise confusion matrices,
and produces a radar plot to visualize fairness disparities.

Main functionalities:
- Compute Balanced Accuracy as the main performance metric.
- Evaluate fairness using Equalized Odds (True Positive Rate and False Positive Rate disparities)
  across multiple subgroups (e.g., age, menopausal status, breast density).
- Plot and save confusion matrices for each subgroup.
- Generate a radar chart summarizing fairness disparities across selected variables.
- Print subgroup-specific performance metrics including Balanced Accuracy and AUC.

Expected Inputs:
- Clinical and imaging metadata (Excel file with 'dataset_info' sheet)
- Ground truth and predicted PCR labels (simulated with random predictions in this template)
Note:
Replace the placeholder prediction generation with your model’s predicted probabilities and binary labels.

Expected Outputs:
- CSV file with subgroup predictions
- Confusion matrix plots for each fairness variable
- Radar plot showing fairness disparities
- Printed metrics for overall and subgroup-level performance

Author: Lidia Garrucho, Universitat de Barcelona
Date: April 2025
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

# ----------------------------
# Utility Functions
# ----------------------------

def plot_confusion_matrices(df, group_var, y_true_col='pcr', y_pred_col='pcr_pred', save_path=None):
    """Plot confusion matrices for each group in the fairness variable."""
    groups = df[group_var].dropna().unique()
    num_groups = len(groups)
    cols = num_groups
    rows = 1
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten()
    # The order of the age groups should be from '0-40' to '71+'
    if group_var == 'age':
        age_order = ['0-40', '41-50', '51-60', '61-70', '71+']
        groups = sorted(groups, key=lambda x: age_order.index(x) if x in age_order else len(age_order))
    elif group_var == 'breast_density':
        density_order = ['a', 'b', 'c', 'd']
        groups = sorted(groups, key=lambda x: density_order.index(x) if x in density_order else len(density_order))
    elif group_var == 'menopausal_status':
        menopause_order = ['premenopause', 'postmenopause']
        groups = sorted(groups, key=lambda x: menopause_order.index(x) if x in menopause_order else len(menopause_order))
    for i, group in enumerate(groups):
        subset = df[df[group_var] == group]
        y_true = subset[y_true_col].astype(int)
        y_pred = subset[y_pred_col].astype(int)
        if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
            axes[i].set_visible(False)
            continue
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], annot_kws={"size": 14})
        axes[i].set_title(f'{group_var} = {group}', fontsize=16)
        axes[i].set_xlabel('Predicted', fontsize=14)
        axes[i].set_ylabel('True', fontsize=14)
        axes[i].tick_params(axis='both', labelsize=14)
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_fairness_radar(equalized_odds_disparities, output_path):
    """Plot a radar chart of Equalized Odds disparities across fairness variables."""
    variables = list(equalized_odds_disparities.keys())
    disparities = list(equalized_odds_disparities.values())
    
    # Radar plot setup
    angles = np.linspace(0, 2 * np.pi, len(variables), endpoint=False).tolist()
    disparities += disparities[:1]  # Cierre del radar
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, disparities, 'o-', linewidth=2, label='Equalized Odds Disparity')
    ax.fill(angles, disparities, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), variables)
    ax.set_title('Fairness Disparity (TPR+FPR)')
    ax.set_ylim(0, max(disparities) + 0.1)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def print_subgroup_metrics(df, fairness_variables):
    """Print balanced accuracy and AUC for each subgroup."""
    for var in fairness_variables:
        print(f'\nSubgroup metrics by {var}')
        for group_name, group in df.groupby(var):
            y_true = group['pcr'].astype(int)
            y_pred = group['pcr_pred'].round().astype(int)
            y_prob = group['pcr_pred']
            try:
                bacc = balanced_accuracy_score(y_true, y_pred)
                auc = roc_auc_score(y_true, y_prob)
                print(f'  {group_name}: Balanced Acc = {bacc:.3f}, AUC = {auc:.3f} (Samples: {len(group)})')
            except ValueError:
                print(f'  {group_name}: Not enough samples for AUC')

# ----------------------------
# Main Execution
# ----------------------------

if __name__ == '__main__':

    # Settings
    alpha = 0.5  # Weight for balancing performance and fairness
    selected_fairness_variables = ['age', 'menopausal_status', 'breast_density']
    # The challenge will also evaluate the breast density variable, but it is not included in all the training data

    # Define paths (modify as needed)
    data_dir = '/path/to/dataset/root/directory'  # Path to the data directory
    clinical_data_xlsx = '/path/to/clinical_and_imaging_info.xlsx' # Path to the clinical data
    output_csv = f'{data_dir}/results_task2.csv'
    output_plots_dir = f'{data_dir}/plots'

    # Read clinical data and get the fairness groups
    clinical_df = pd.read_excel(clinical_data_xlsx, sheet_name='dataset_info')
    # For fairness_varibles_df, we will drop all the clinical_df columns except the selected_fairness_variables and patient_id
    fairness_varibles_df = clinical_df[['patient_id', 'pcr'] + selected_fairness_variables]
    # Modify age column values mapping them by age groups
    fairness_varibles_df['age'] = pd.cut(fairness_varibles_df['age'], bins=[0, 40, 50, 60, 70, 100], labels=['0-40', '41-50', '51-60', '61-70', '71+'])
    # Clean menopausal status
    fairness_varibles_df['menopausal_status'] = (
        fairness_varibles_df['menopausal_status']
        .fillna('unknown')
        .str.lower()
        .apply(lambda x: 'pre' if 'peri' in x or 'pre' in x else ('post' if 'post' in x else x))
    )

    # Create output directories if they do not exist
    os.makedirs(output_plots_dir, exist_ok=True)

    # Generate random predictions (to be replaced with model output)
    np.random.seed(42)
    pred_scores = np.random.rand(len(fairness_varibles_df))
    fairness_varibles_df['pcr_pred'] = pred_scores > 0.5
    fairness_varibles_df['pcr_pred'] = fairness_varibles_df['pcr_pred'].astype(int)
    fairness_varibles_df.to_csv(output_csv, index=False)

    # Score performance
    y_true = fairness_varibles_df['pcr'].fillna(0).astype(int)
    y_pred = fairness_varibles_df['pcr_pred'].fillna(0).astype(int)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    print(f'Average Balanced Accuracy: {balanced_accuracy:.4f}')
    performance_score = balanced_accuracy

    # Fairness analysis
    equalized_odds_disparities = {}

    for var in selected_fairness_variables:
        groups = fairness_varibles_df.groupby(var)
        tpr, fpr = {}, {}

        for i, (_, group) in enumerate(groups):
            yt = group['pcr'].astype(int)
            yp = group['pcr_pred'].astype(int)
            try:
                tn, fp_, fn, tp = confusion_matrix(yt, yp).ravel()
                tpr[i] = tp / (tp + fn) if (tp + fn) else 0
                fpr[i] = fp_ / (fp_ + tn) if (fp_ + tn) else 0
            except ValueError:
                tpr[i], fpr[i] = 0, 0

        disparity = max(tpr.values()) - min(tpr.values()) + max(fpr.values()) - min(fpr.values())
        equalized_odds_disparities[var] = disparity

    fairness_score = np.mean(list(equalized_odds_disparities.values()))
    fairness_score = np.clip(fairness_score, 0, 1)
    ranking_score = (1 - alpha) * performance_score + alpha * (1 - fairness_score)

    print(f'Fairness Score: {1 - fairness_score:.4f}')
    print(f'Ranking Score: {ranking_score:.4f}')

    # Print subgroup metrics
    print_subgroup_metrics(fairness_varibles_df, selected_fairness_variables)

    # Generate plots
    radar_plot_path = os.path.join(output_plots_dir, 'radar_fairness_disparities.png')
    plot_fairness_radar(equalized_odds_disparities, radar_plot_path)

    for var in selected_fairness_variables:
        plot_confusion_matrices(
            fairness_varibles_df, group_var=var,
            save_path=os.path.join(output_plots_dir, f'cm_by_{var}.png')
        )