#!/usr/bin/env python3

# This script is used to train a classifier given a csv of TCR-pMHC positive and negative data.
# Example of execution:
# python train_xgb.py -in ../training_test.csv -out output_model --mhc

import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import sys
import os
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

def train_fold(df, fold, params, final=False, mhc=False):
    if final:
        train_df = df[df['fold'] != fold]
        val_df = df[df['fold'] == fold]
        print(f"Training on all folds except {fold}...")
    else:
        train_df = df[(df['fold'] != fold) & (df['fold'] != 6)]
        val_df = df[df['fold'] == fold]
        print(f"Training on folds except {fold} and validation...")

    print(f"Training on {len(train_df)} samples, validation on {len(val_df)} samples...")
    if mhc:
        X_train, y_train = prepare_data_mhc(train_df)
        X_val, y_val = prepare_data_mhc(val_df)
    else:
        X_train, y_train = prepare_data(train_df)
        X_val, y_val = prepare_data(val_df)

    train_dmatrix = xgb.DMatrix(X_train, label=y_train)
    val_dmatrix = xgb.DMatrix(X_val, label=y_val)

    evals = [(train_dmatrix, 'train'), (val_dmatrix, 'eval')]
    evals_result = {}

    model = xgb.train(params=params, 
                      dtrain=train_dmatrix,
                      num_boost_round=1000,               
                      evals=evals,
                      early_stopping_rounds=10,         
                      verbose_eval=False,
                      evals_result=evals_result)

    try:
        train_aucpr = evals_result['train']['aucpr'][model.best_iteration]
        val_aucpr = evals_result['eval']['aucpr'][model.best_iteration]
        print("Train AUCPR:", train_aucpr)
        print("Val AUCPR:", val_aucpr)
    except KeyError:
        print("AUCPR not found in evals_result. Check if 'aucpr' is included in 'eval_metric' parameter.")
        train_aucpr = None
        val_aucpr = None

    return model, train_aucpr, val_aucpr

def grid_search(df, params, params_dict, mhc=False):
    params.update(params_dict)
    auc_pr_list_val = []
    auc_pr_list_train = []

    for fold in range(1, 6):
        print(f"Training fold {fold}...")
        _, train_aucpr, val_aucpr = train_fold(df, fold, params, final=False, mhc=mhc)
        auc_pr_list_val.append(val_aucpr)
        auc_pr_list_train.append(train_aucpr)

    return np.mean(auc_pr_list_val), np.mean(auc_pr_list_train), params_dict

def main():
    parser = argparse.ArgumentParser(description="Train a classifier on TCR-pMHC data")
    parser.add_argument('-in', '--input', type=str, required=True, help="Input CSV file with TCR-pMHC data")
    parser.add_argument('-out','--output', type=str, required=True, help="Output directory for the trained model.")
    parser.add_argument('--mhc', action='store_true', help="Use MHC data for training.")
    args = parser.parse_args()

    training_own = pd.read_csv(args.input)
    mhc= args.mhc
    print("Parameter MHC:", mhc)

    params = {
                "objective": "binary:logistic",
                "eval_metric": "aucpr",
                "n_jobs": -1,}

    if mhc:
        X_combined, y = prepare_data_mhc(training_own)
    else:
        X_combined, y = prepare_data(training_own)

    print("Training data shape:", X_combined.shape)

    param_grid = { 
            'learning_rate': np.arange(0.01, 1.05, 0.05).tolist(),
            'max_depth': list(range(1, 7)),
            'scale_pos_weight': [5],}

    param_combinations = []
    for learning_rate in param_grid['learning_rate']:
        for max_depth in param_grid['max_depth']:
            for scale_pos_weight in param_grid['scale_pos_weight']:
                param_combinations.append({'learning_rate': learning_rate, 'max_depth': max_depth, 'scale_pos_weight': scale_pos_weight})

    print("Total combinations:", len(param_combinations))

    best_auc_pr_val = 0
    best_params = None
    best_params_train = None

    for params_dict in param_combinations:
        print()
        print("Testing parameters:", params_dict)
        auc_pr_val, auc_pr_train, params_dict = grid_search(training_own, params, params_dict, mhc=mhc)
        print("Validation AUCPR:", auc_pr_val)
        print("Train AUCPR:", auc_pr_train)

        if auc_pr_val > best_auc_pr_val:
            best_auc_pr_val = auc_pr_val
            best_params = params_dict
            best_params_train = auc_pr_train
    print()
    print("Best parameters:", best_params)
    print("Best validation AUCPR:", best_auc_pr_val)
    print("Best train AUCPR:", best_params_train)

    # Train the final model with the best parameters
    params.update(best_params)
    final_model, final_auc_pr_train, final_auc_pr_val = train_fold(training_own, 6, params, final=True, mhc=mhc)
    print()
    print("Final model training complete.")
    print("Final validation AUCPR:", final_auc_pr_val)
    print("Final train AUCPR:", final_auc_pr_train)

    # Create dir to save the model
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Save the final model
    # Round the parameters to 2 decimal places
    best_params['learning_rate'] = round(best_params['learning_rate'], 2)
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['scale_pos_weight'] = int(best_params['scale_pos_weight'])
    
    if mhc: 
        model_name = f"model_mhc_spw_{best_params['scale_pos_weight']}_lr_{best_params['learning_rate']}_md_{best_params['max_depth']}.json"
    else:
        model_name = f"model_nomhc_spw_{best_params['scale_pos_weight']}_lr_{best_params['learning_rate']}_md_{best_params['max_depth']}.json"

    final_model.save_model(os.path.join(args.output, model_name))
    print(f"Model saved to {os.path.join(args.output, model_name)}")

if __name__ == '__main__':
    main()
