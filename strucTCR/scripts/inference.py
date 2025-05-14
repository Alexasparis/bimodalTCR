#!/usr/bin/env python3

# This script is used for inference of a classifier given a csv of TCR-pMHC positive and negative data.
# Example of execution:
# python inference.py -in ../test_set.csv -out predicitons/ -metrics metrics_predictions/ -model ../output_model/model_nomhc_spw_5_lr_0.21_md_4.json

import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc, precision_recall_curve
import seaborn as sns
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

energy_columns = [
    'score_tcr_p1', 'score_tcr_p2', 'score_tcr_p3', 'score_tcr_p4', 'score_tcr_p5', 'score_tcr_p6', 'score_tcr_p7',
    'score_tcr_p8', 'score_tcr_p9', 'score_tcr_p10', 'score_tcr_p11', 'score_tcr_p12', 'score_tcr_p13',
    'contacts_tcr_p1', 'contacts_tcr_p2', 'contacts_tcr_p3', 'contacts_tcr_p4', 'contacts_tcr_p5', 'contacts_tcr_p6',
    'contacts_tcr_p7', 'contacts_tcr_p8', 'contacts_tcr_p9', 'contacts_tcr_p10', 'contacts_tcr_p11', 'contacts_tcr_p12', 'contacts_tcr_p13']

energy_columns_mhc = [
    'score_tcr_mhc', 'contacts_tcr_mhc']

def epitope_to_int(epitope):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    amino_acid_to_int = {aa: idx + 1 for idx, aa in enumerate(amino_acids)}
    return [amino_acid_to_int[aa] for aa in epitope if aa in amino_acid_to_int]

def prepare_data(df):
    X_energy = df[energy_columns].values
    y = df['Label'].values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    epitope_int_sequences = [epitope_to_int(epitope) for epitope in df['Epitope']]
    maxlen = 13
    X_epitope_padded = pad_sequences(epitope_int_sequences, maxlen=maxlen, padding='post', truncating='post')
    X_combined = np.hstack((X_energy, X_epitope_padded)) 
    return X_combined, y_encoded

def prepare_data_mhc(df):
    energy_columns_2 = energy_columns + energy_columns_mhc
    X_energy = df[energy_columns_2].values
    y = df['Label'].values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    epitope_int_sequences = [epitope_to_int(epitope) for epitope in df['Epitope']]
    maxlen = 13
    X_epitope_padded = pad_sequences(epitope_int_sequences, maxlen=maxlen, padding='post', truncating='post')
    X_combined = np.hstack((X_energy, X_epitope_padded)) 
    return X_combined, y_encoded

def binary_roc_auc_score(y_true, y_score, sample_weight=None, max_fpr=None):

    if len(np.unique(y_true)) != 2:
        raise ValueError("Only one class present in y_true. ROC AUC score is not defined in that case.")

    fpr, tpr, _ = roc_curve(y_true, y_score, sample_weight=sample_weight)
    if max_fpr is None or max_fpr == 1:
        return auc(fpr, tpr)
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError("Expected max_fpr in range (0, 1], got: %r" % max_fpr)

    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    partial_auc = auc(fpr, tpr)

    min_area = 0.5 * max_fpr**2
    max_area = max_fpr
    return 0.5 * (1 + (partial_auc - min_area) / (max_area - min_area))

def save_metrics_and_plots(test_own, y_test, y_prob, metric_folder, model_name):
    os.makedirs(metric_folder, exist_ok=True)

    # ROC y PR curves
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    avg_precision = average_precision_score(y_test, y_prob)
    auc_01 = binary_roc_auc_score(y_test, y_prob, max_fpr=0.1)

    # Save main curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}', color='blue')
    axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray')
    axes[0].set_title('ROC Curve'); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(recall, precision, label=f'PR AUC = {pr_auc:.2f}', color='green')
    axes[1].set_title('Precision-Recall Curve'); axes[1].legend(); axes[1].grid(True)

    # Partial ROC
    fpr_partial_mask = fpr <= 0.12
    fpr_01 = fpr[fpr_partial_mask]
    tpr_01 = tpr[fpr_partial_mask]
    if len(fpr_01) == 0 or fpr_01[-1] < 0.1:
        fpr_01 = np.append(fpr_01, 0.1)
        tpr_01 = np.append(tpr_01, np.interp(0.1, fpr, tpr))

    axes[2].plot(fpr_01, tpr_01, label=f'Partial AUC@0.1 = {auc_01:.3f}', color='purple')
    axes[2].plot([0, 0.1], [0, 0.1], 'k--', label='Random')
    axes[2].set_xlim(0, 0.1); axes[2].set_ylim(0, 1)
    axes[2].set_title('Partial ROC Curve (FPR ≤ 0.1)')
    axes[2].legend(); axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(metric_folder, f'{model_name}_curves.jpg'))
    plt.close()

    # Success at N
    model_data = pd.DataFrame({'True Values': test_own['Label'], 'Predicted Probabilities': y_prob})
    model_data_sorted = model_data.sort_values(by='Predicted Probabilities', ascending=False)
    thresholds = [10, 100, 1000]
    success_rates = [(model_data_sorted.head(n)['True Values'] == 1).mean() * 100 for n in thresholds]

    plt.figure(figsize=(5, 5))
    plt.bar([str(t) for t in thresholds], success_rates, color=['midnightblue', 'cornflowerblue', 'lightsteelblue', 'lightgray'])
    for i, v in enumerate(success_rates):
        plt.text(i, v + 1, f"{v:.2f}%", ha='center')
    plt.title('Success Rate at Top N Predictions'); plt.xlabel('Top N'); plt.ylabel('Success Rate (%)')
    plt.ylim(0, 100); plt.tight_layout()
    plt.savefig(os.path.join(metric_folder, f'{model_name}_success_rate.jpg'))
    plt.close()

    # Save summary metrics to CSV
    metrics_summary = {
        'Accuracy': accuracy_score(y_test, y_prob >= 0.5),
        'Precision': precision_score(y_test, y_prob >= 0.5),
        'Recall': recall_score(y_test, y_prob >= 0.5),
        'F1': f1_score(y_test, y_prob >= 0.5),
        'ROC AUC': roc_auc,
        'PR AUC': pr_auc,
        'AUC 0.1': auc_01,
        'Avg Precision': avg_precision,
        'Success@10': success_rates[0],
        'Success@100': success_rates[1],
        'Success@1000': success_rates[2]}
    
    pd.DataFrame([metrics_summary]).to_csv(os.path.join(metric_folder, f'{model_name}_metrics.csv'), index=False)

    # Métricas por epítopo
    results_per_epitope = {}

    for epitope in np.unique(test_own['Epitope']):
        epitope_length = len(epitope)
        epitope_counts = len(test_own[test_own['Epitope'] == epitope])
    
        subset = test_own[test_own['Epitope'] == epitope]
        y_true = subset['Label']
        y_probs = subset['probs']
        if len(np.unique(y_true)) < 2:
            continue
        try:
            auc01 = binary_roc_auc_score(y_true, y_probs, max_fpr=0.1)
        except ValueError:
            auc01 = np.nan
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        results_per_epitope[epitope] = {
            'ROC AUC': roc_auc_score(y_true, y_probs),
            'PR AUC': auc(recall, precision),
            'Avg Precision': average_precision_score(y_true, y_probs),
            'AUC 0.1': auc01,
            'Epitope Length': epitope_length,
            'Epitope Counts': epitope_counts}

    results_df = pd.DataFrame.from_dict(results_per_epitope, orient='index')
    results_df.to_csv(os.path.join(metric_folder, f'{model_name}_metrics_per_epitope.csv'))

    # Boxplot
    metrics = ['ROC AUC', 'AUC 0.1', 'PR AUC', "Avg Precision" ]
    plt.figure(figsize=(10, 10))
    sns.boxplot(data=results_df[metrics], palette="Set2", showfliers=False)
    sns.stripplot(data=results_df[metrics], color='black', size=3, jitter=True)
    for i, m in enumerate(results_df[metrics].mean()):
        plt.text(i, m + 0.05, f"{m:.2f}", ha='center', fontsize=14, fontweight='bold')
    plt.xticks(fontsize=16); plt.yticks(fontsize=14)
    plt.grid(axis='y')
    plt.savefig(os.path.join(metric_folder, f'{model_name}_epitope_boxplot.jpg'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate predictions for TCR-pMHC data.")
    parser.add_argument('-in', '--input', type=str, required=True, help="Input CSV file with TCR-pMHC data")
    parser.add_argument('-model','--model_path', type=str, required=True, help="Path to the model file")
    parser.add_argument('-out', '--output', type=str, required=True, help="Output folder for preditions")
    parser.add_argument('-metrics', '--metric_folder', type=str, required=False, help="Path to the folder with metrics")
    args = parser.parse_args()
    # Create folders
    os.makedirs(args.output, exist_ok=True)
    if args.metric_folder:
        os.makedirs(args.metric_folder, exist_ok=True)

    model_name = args.model_path.split('/')[-1].split('.')[0]
    test_own = pd.read_csv(args.input)
    final_model = xgb.Booster()
    final_model.load_model(args.model_path)

    if "nomhc" in model_name:
        print("Using model without MHC")
        X_test, y_test = prepare_data(test_own)
    else:
        print("Using model with MHC")
        X_test, y_test = prepare_data_mhc(test_own)

    print("Shape X_test:", X_test.shape)

    # Prediction
    y_prob = final_model.predict(xgb.DMatrix(X_test))
    test_own['probs'] = y_prob
    test_own.to_csv(os.path.join(args.output, f'{model_name}_predictions.csv'), index=False)
    print("Predictions saved to:", os.path.join(args.output, f'{model_name}_predictions.csv'))

    if args.metric_folder:
        save_metrics_and_plots(test_own, y_test, y_prob, args.metric_folder, model_name)
        print("Metrics and plots saved to:", args.metric_folder)

if __name__ == '__main__':
    main()
