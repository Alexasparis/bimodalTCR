#!/usr/bin/env python3

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, roc_curve
import numpy as np
import pickle
import pandas as pd
import xgboost as xgb
import pickle
import argparse
import os
import csv
import h5py
from itertools import product
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def global_average_pooling(embedding, tcr_id=None):
    if embedding.size > 0:
        return np.mean(embedding, axis=1)
    else:
        # Print a warning with the tcr_id if provided
        if tcr_id is not None:
            print(f"Warning: Empty array passed for TCR ID {tcr_id}")
        else:
            print("Warning: Empty array passed to global_average_pooling")
        return None

def collect_sequences_and_convert_to_arrays(all_last_hidden_states, all_cdrs):
    collected_data = {}
    for key, tensor in all_last_hidden_states.items():
        tcr_id = "_".join(key.split('_')[:2])
        if tcr_id not in collected_data:
            collected_data[tcr_id] = {}
        sequence_type = "_".join(key.split('_')[2:])
        collected_data[tcr_id][sequence_type] = tensor
    for key, tensor in all_cdrs.items():
        tcr_id = "_".join(key.split('_')[:2])
        if tcr_id not in collected_data:
            collected_data[tcr_id] = {}
        sequence_type = "_".join(key.split('_')[2:])
        collected_data[tcr_id][sequence_type] = tensor
    return collected_data

def load_embeddings(file_path):
    embeddings = {}
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            embeddings[key] = f[key][:]
    return embeddings

def process_embeddings(data, tcr_id, keys, pooling_function):
    """
    Process the embeddings for a given TCR ID and data, applying the pooling function to the specified keys.
    Returns a dictionary where the keys are the original keys and the values are the processed embeddings,
    or None if any embedding is invalid.
    """
    embeddings_dict = {}
    for key in keys:
        emb = pooling_function(np.array(data[key]), tcr_id)
        if emb is None or emb.size == 0:
            return None  
        embeddings_dict[key] = emb 
    
    return embeddings_dict

def process_data_for_type(data_dict, data_type, pooling_function, train=True):
    """
    Process the data for a given type (var, cdrs, etc.) and return the embeddings and labels.
    """
    embeddings = {'tcra': [], 'tcrb': [], 'epitope': [], 'mhc': []}
    labels = []
    structural=[]
    tcr_ids = []

    # Iterate over each TCR and process its data
    for tcr_id, data in data_dict.items():
        if "MHC_seq" in data and "Epitope" in data:
            if data_type == "var":
                keys = ["vara", "varb", "Epitope", "MHC_seq"]
            elif data_type == "cdrs":
                keys = ["cdr1a", "cdr1b", "cdr2a", "cdr2b", "cdr3a", "cdr3b", "Epitope", "MHC_seq"]
            elif data_type == "cdr3s":
                keys = ["cdr3a", "cdr3b", "Epitope", "MHC_seq"]
            elif data_type == "cdr3b":
                keys = ["cdr3b", "Epitope", "MHC_seq"]
            elif data_type == "cdr3a":
                keys = ["cdr3a", "Epitope", "MHC_seq"]
            else:
                print(f"Invalid data type: {data_type}")
                continue 

            # Process embeddings using the general pooling function
            if all(key in data for key in keys):
                processed_data = process_embeddings(data, tcr_id, keys, pooling_function)
                if processed_data is None:
                    continue  # Skip if any embedding is invalid
            
            if "structural" not in data:
                continue

            # Depending on the data type, assign the embeddings to the correct TCR chain (tcra/tcrb)
            if data_type == "var":
                embeddings['tcra'].append(processed_data['vara'])
                embeddings['tcrb'].append(processed_data['varb'])
            elif data_type == "cdrs":
                embeddings['tcra'].append(np.concatenate([processed_data['cdr1a'], processed_data['cdr2a'], processed_data['cdr3a']], axis=1))
                embeddings['tcrb'].append(np.concatenate([processed_data['cdr1b'], processed_data['cdr2b'], processed_data['cdr3b']], axis=1))
            elif data_type == "cdr3s":
                embeddings['tcra'].append(processed_data['cdr3a'])
                embeddings['tcrb'].append(processed_data['cdr3b'])
            elif data_type == "cdr3b":
                embeddings['tcrb'].append(processed_data['cdr3b'])
            elif data_type == "cdr3a":
                embeddings['tcra'].append(processed_data['cdr3a'])
            
            embeddings['epitope'].append(processed_data['Epitope'])  
            embeddings['mhc'].append(processed_data['MHC_seq']) 
            structural.append (data['structural'])
            labels.append(data["Label"])
            tcr_ids.append(tcr_id)

    return embeddings, labels, structural, tcr_ids

def process_all_data(all_data, type_seq, fold, pooling_function):
    tcra_embeddings_train, tcrb_embeddings_train, epitope_embeddings_train, mhc_embeddings_train, labels_train, structural_train, tcr_ids_train = [], [], [], [], [], [], []
    tcra_embeddings_val, tcrb_embeddings_val, epitope_embeddings_val, mhc_embeddings_val, labels_val, structural_val, tcr_ids_val = [], [], [], [], [], [], []

    collected_data_val = all_data[fold]
    data_train = {k: v for k, v in all_data.items() if k != fold and k != 6 and k != 7}

    # Collect training data
    collected_data_train = {}
    for key, value in data_train.items():
        for tcr, embedding_dict in value.items():
            collected_data_train[tcr] = embedding_dict

    print(f"Number of TCRs in collected_data_train: {len(collected_data_train)}")
    print(f"Number of TCRs in collected_data_val: {len(collected_data_val)}")

    # Process the training data
    embeddings_train, labels_train, structural_train, tcr_ids_train = process_data_for_type(collected_data_train, type_seq, pooling_function)
    tcra_embeddings_train.extend(embeddings_train['tcra'])
    tcrb_embeddings_train.extend(embeddings_train['tcrb'])
    epitope_embeddings_train.extend(embeddings_train['epitope'])
    mhc_embeddings_train.extend(embeddings_train['mhc'])

    # Process the validation data
    embeddings_val, labels_val, structural_val, tcr_ids_val = process_data_for_type(collected_data_val, type_seq, pooling_function)
    tcra_embeddings_val.extend(embeddings_val['tcra'])
    tcrb_embeddings_val.extend(embeddings_val['tcrb'])
    epitope_embeddings_val.extend(embeddings_val['epitope'])
    mhc_embeddings_val.extend(embeddings_val['mhc'])

    print(len(labels_train), len(labels_val), len(structural_train), len(structural_val))
    return tcra_embeddings_train, tcrb_embeddings_train, epitope_embeddings_train, mhc_embeddings_train, labels_train, structural_train, tcr_ids_train, \
           tcra_embeddings_val, tcrb_embeddings_val, epitope_embeddings_val, mhc_embeddings_val, labels_val, structural_val, tcr_ids_val

def create_embedding_matrix(tcra_embeddings, tcrb_embeddings, epitope_embeddings, mhc_embeddings, structural_emb):
    structural_emb = np.array(structural_emb).squeeze()

    if len(tcra_embeddings) == 0:
        return np.concatenate([
            np.array(tcrb_embeddings).squeeze(),
            np.array(epitope_embeddings).squeeze(),
            np.array(mhc_embeddings).squeeze(), structural_emb],
            axis=1)
    elif len(tcrb_embeddings) == 0:
        return np.concatenate([
            np.array(tcra_embeddings).squeeze(),
            np.array(epitope_embeddings).squeeze(),
            np.array(mhc_embeddings).squeeze(),structural_emb],
            axis=1)
    else:
        return np.concatenate([
            np.array(tcra_embeddings).squeeze(),
            np.array(tcrb_embeddings).squeeze(),
            np.array(epitope_embeddings).squeeze(),
            np.array(mhc_embeddings).squeeze(),structural_emb],
            axis=1)

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
    
def main():
    parser = argparse.ArgumentParser(description='Train XGBoost model using TCR and MHC data')
    parser.add_argument("-df", "--dataframe", type=str, help='Path to the folder with dataframes with the TCRs and labels')
    parser.add_argument("-emb", "--embeddings", type=str, help='Path to the embeddings folder')
    parser.add_argument("-t", "--type", type=str, help='Options: "var", "cdrs", cdr3s" or "cdr3b" ')
    args = parser.parse_args()

    if not os.path.exists(args.dataframe):
        raise FileNotFoundError(f"The data file at {args.dataframe} does not exist.")

    if not os.path.exists(args.embeddings):
        raise FileNotFoundError(f"The embeddings file at {args.embeddings} does not exist.")
    
    dataframe_folder = os.path.basename(os.path.normpath(args.dataframe))

    # Load data
    print("Loading data...")
    all_data = {}
    for fold in range(1, 8):
        df_file = os.path.join(args.dataframe, f"training_fold_{fold}.csv")
        general_embeddings_file = os.path.join(args.embeddings, f"training_fold_{fold}/last_hidden_state_esmc_600m.h5py")
        cdr_embeddings_file = os.path.join(args.embeddings, f"training_fold_{fold}/cdrs.h5py")
        if not os.path.exists(df_file):
            raise FileNotFoundError(f"Data file not found: {df_file}")
        if not os.path.exists(general_embeddings_file):
            raise FileNotFoundError(f"General embeddings file not found: {general_embeddings_file}")
        if not os.path.exists(cdr_embeddings_file):
            raise FileNotFoundError(f"CDR embeddings file not found: {cdr_embeddings_file}")
        
        print(f"Loading fold {fold} data...")
        df = pd.read_csv(df_file)
        #df['TCR_name'] = df['TCR_name'].astype(str)
        df['TCR_name'] = df['TCR_name'].astype(str).str.strip()
        if "TCR_name" not in df.columns or "Label" not in df.columns:
            raise KeyError(f"Missing required columns in {df_file}")

        general_embeddings = load_embeddings(general_embeddings_file)
        cdr_embeddings = load_embeddings(cdr_embeddings_file)
        collected_data = collect_sequences_and_convert_to_arrays(general_embeddings, cdr_embeddings)
        all_data[fold] = collected_data
        #for index, tcr_id in enumerate(collected_data.keys()):
        #    tcr_row = df[df["TCR_name"] == str(tcr_id)]
        #    if not tcr_row.empty:
        #        label = tcr_row['Label'].iloc[0] 
        #        structural_features, _ = prepare_data_mhc(tcr_row)
        #        structural_features = structural_features[0]
        #        collected_data[tcr_id]["Label"] = label
        #        collected_data[tcr_id]["structural"] = structural_features
        #    else:
        #        print(f"No matching TCR_name found for {tcr_id}")
        for index, tcr_id in enumerate(collected_data.keys()):
            tcr_id = str(tcr_id).strip()  # Asegúrate de que tcr_id no tenga espacios extras
            if tcr_id in df["TCR_name"].values:  # Verifica si tcr_id está presente en TCR_name
                print(tcr_id) 
                
                tcr_row = df[df["TCR_name"] == tcr_id]
                print(tcr_row)
                label = tcr_row['Label'].iloc[0]
                structural_features, _ = prepare_data_mhc(tcr_row)
                structural_features = structural_features[0]
        
                collected_data[tcr_id]["Label"] = label
                collected_data[tcr_id]["structural"] = structural_features
            else:
                print(f"No matching TCR_name found for {tcr_id}")
        
        print(f"Number of TCRs in collected_data: {len(collected_data)} for fold {fold}")

    # Define param grid
    #param_grid = {
    #    'learning_rate': np.arange(0.01, 1.05, 0.05).tolist(),
    #    'max_depth': list(range(1, 7)),
    #    'alpha': [0],
    #    'lambda': [1]}
    param_grid = {
        'learning_rate': [0.21],
        'max_depth': [4],
        'alpha': [0],
        'lambda': [1]}

    # Create directory for best models if it doesn't exist
    if not os.path.exists('best_models_xgb'):
        os.makedirs('best_models_xgb')
    
    # CSV file for storing results
    csv_file = f'./best_models_xgb/{dataframe_folder}_grid_search_results_xgboost_{args.type}.csv'
    csv_columns = [
        'learning_rate',
        'max_depth',
        'alpha',
        'lambda',
        'avg_val_auc_pr',
        'avg_train_auc_pr']

    # Write headers to CSV file if it doesn't exist
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()

    # Grid search:
    best_auc = 0
    best_params = None
    print("Starting grid search...")
    
    for learning_rate, max_depth, alpha, reg_lambda in product(
        param_grid['learning_rate'],
        param_grid['max_depth'],
        param_grid['alpha'],
        param_grid['lambda']):

        print(f"\n---------- Training with LR={learning_rate}, Max Depth={max_depth}, Alpha={alpha}, Lambda={reg_lambda} ----------")

        params = {
        "objective": "binary:logistic",
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "eval_metric": "aucpr",
        "n_jobs": -1,
        "scale_pos_weight": 5,
        "alpha": alpha,
        "lambda": reg_lambda}
        
        aucs_training, aucs_validation = [], []
        
        # Cross validation
        for fold in range(1, 6):
            print(f"\n----------Training fold {fold}----------")
            
            tcra_embeddings_train, tcrb_embeddings_train, epitope_embeddings_train, mhc_embeddings_train, labels_train,structural_train, tcr_ids_train, \
            tcra_embeddings_val, tcrb_embeddings_val, epitope_embeddings_val, mhc_embeddings_val, labels_val, structural_val, tcr_ids_val = process_all_data(all_data, args.type, fold, global_average_pooling)
            
            print(f"Train tcra_embeddings[0].shape: {tcra_embeddings_train[0].shape if len(tcra_embeddings_train) > 0 else 'empty'}, "
                f"tcrb_embeddings[0].shape: {tcrb_embeddings_train[0].shape if len(tcrb_embeddings_train) > 0 else 'empty'}, "
                f"mhc_embeddings[0].shape: {mhc_embeddings_train[0].shape if len(mhc_embeddings_train) > 0 else 'empty'}, "
                f"epitope_embeddings[0].shape: {epitope_embeddings_train[0].shape if len(epitope_embeddings_train) > 0 else 'empty'}, "
                f"structural_embeddings[0].shape: {structural_train[0].shape if len(structural_train) > 0 else 'empty'}")

            print(f"Validation tcra_embeddings[0].shape: {tcra_embeddings_val[0].shape if len(tcra_embeddings_val) > 0 else 'empty'}, "
                f"tcrb_embeddings[0].shape: {tcrb_embeddings_val[0].shape if len(tcrb_embeddings_val) > 0 else 'empty'}, "
                f"mhc_embeddings[0].shape: {mhc_embeddings_val[0].shape if len(mhc_embeddings_val) > 0 else 'empty'}, "
                f"epitope_embeddings[0].shape: {epitope_embeddings_val[0].shape if len(epitope_embeddings_val) > 0 else 'empty'}, "
                f"structural_embeddings[0].shape: {structural_val[0].shape if len(structural_val) > 0 else 'empty'}")

            print(f"Length of training data: TCRA {len(tcra_embeddings_train)}, TCRB {len(tcrb_embeddings_train)}, Epitope {len(epitope_embeddings_train)}, "
                f"MHC {len(mhc_embeddings_train)}, Structural {len(structural_train)}, tcrids {len(tcr_ids_train)}, labels {len(labels_train)}")

            print(f"Length of validation data: TCRA {len(tcra_embeddings_val)}, TCRB {len(tcrb_embeddings_val)}, Epitope {len(epitope_embeddings_val)}, "
                f"MHC {len(mhc_embeddings_val)}, Structural {len(structural_val)}, tcrids {len(tcr_ids_val)}, labels {len(labels_val)}")
            
            X_train = create_embedding_matrix(tcra_embeddings_train, tcrb_embeddings_train, epitope_embeddings_train, mhc_embeddings_train,structural_train)
            X_val = create_embedding_matrix(tcra_embeddings_val, tcrb_embeddings_val, epitope_embeddings_val, mhc_embeddings_val, structural_val)
                
            tcr_ids_train = np.array(tcr_ids_train)
            tcr_ids_val = np.array(tcr_ids_val)

            y_train = np.array(labels_train)
            y_val = np.array(labels_val)

            # Debug
            print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")
            print(f"X_val.shape: {X_val.shape}, y_val.shape: {y_val.shape}")
            print(f"tcr_ids_train.shape: {tcr_ids_train.shape}, tcr_ids_val.shape: {tcr_ids_val.shape}")
            print(f"y_train: {y_train.shape}, y_val: {y_val.shape}")
        
            train_dmatrix = xgb.DMatrix(X_train, label=y_train)
            val_dmatrix = xgb.DMatrix(X_val, label=y_val)

            evals = [(train_dmatrix, 'train'), (val_dmatrix, 'eval')]
            model = xgb.train(params=params, 
                              dtrain=train_dmatrix,
                              maximize=True,
                              num_boost_round=1000,               
                              evals=evals,
                              early_stopping_rounds=10,         
                              verbose_eval=True)
            
            evals_result = model.eval_set(evals)
            print("Evals_result", evals_result)
            evals_result_list = evals_result.split('\t') 
            print("Evals_result List:", evals_result_list)

            if len(evals_result_list) > 1:
                train_aucpr = float(evals_result_list[1].split(":")[1])
                val_aucpr = float(evals_result_list[2].split(":")[1])
            print("Train AUCPR:", train_aucpr)
            print("Val AUCPR:", val_aucpr)
            aucs_training.append(train_aucpr)
            aucs_validation.append(val_aucpr)
        
        print(aucs_training)
        print(aucs_validation)

        mean_training_auc = np.mean(aucs_training)
        mean_validation_auc = np.mean(aucs_validation) 

        print(f"Mean training AUC-PR: {mean_training_auc}")
        print(f"Mean validation AUC-PR: {mean_validation_auc}")

        if mean_validation_auc > best_auc:
            best_auc = mean_validation_auc
            best_params = {
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "alpha": alpha,
                "lambda": reg_lambda}
        
        results = {
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'alpha': alpha,
            'lambda': reg_lambda,
            'avg_val_auc_pr': mean_validation_auc,
            'avg_train_auc_pr': mean_training_auc}
       
        # Results to csv
        with open(csv_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writerow(results)
    
    print(f"Best AUC-PR: {best_auc} with params: {best_params}")

    # Train the best model
    final_params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "learning_rate": best_params['learning_rate'],
        "max_depth": best_params['max_depth'],
        "alpha":best_params['alpha'],
        "lambda": best_params['lambda'],
        "scale_pos_weight": 5,
        "n_jobs": -1}
    
    # Split the data into training and validation sets
    collected_data_val = all_data[6]
    data_train = {k: v for k, v in all_data.items() if k != 6 and k != 7}
    collected_data_train = {}
    for key, value in data_train.items():
        for tcr, embedding_dict in value.items():
            collected_data_train[tcr] = embedding_dict

    print(f"Number of TCRs in final collected_data_train: {len(collected_data_train)}")
    print(f"Number of TCRs in final collected_data_val: {len(collected_data_val)}")
    
    # Process training and validation data for each type
    tcra_embeddings_train, tcrb_embeddings_train, epitope_embeddings_train, mhc_embeddings_train, labels_train, structural_train, tcr_ids_train, \
    tcra_embeddings_val, tcrb_embeddings_val, epitope_embeddings_val, mhc_embeddings_val, labels_val, structural_val, tcr_ids_val = process_all_data(all_data, args.type, 6, global_average_pooling)

    X_train = create_embedding_matrix(tcra_embeddings_train, tcrb_embeddings_train, epitope_embeddings_train, mhc_embeddings_train, structural_train)
    X_val = create_embedding_matrix(tcra_embeddings_val, tcrb_embeddings_val, epitope_embeddings_val, mhc_embeddings_val, structural_val)

    tcr_ids_train = np.array(tcr_ids_train)
    tcr_ids_val = np.array(tcr_ids_val)

    y_train = np.array(labels_train)
    y_val = np.array(labels_val)

    train_dmatrix_final = xgb.DMatrix(X_train, label=y_train)
    val_dmatrix_final = xgb.DMatrix(X_val, label=y_val)
    evals=[(train_dmatrix_final, "train"), (val_dmatrix_final, "val")] 
    best_model = xgb.train(params=final_params,
                           dtrain=train_dmatrix_final,
                           maximize=True,
                           num_boost_round=1000, 
                           evals=evals,
                           early_stopping_rounds=10,
                           verbose_eval=True)

    evals_result = best_model.eval_set(evals)
    print("Evals_result", evals_result)
    evals_result_list = evals_result.split('\t') 
    print("Evals_result List:", evals_result_list)

    if len(evals_result_list) > 1:
        train_aucpr = float(evals_result_list[1].split(":")[1])
        val_aucpr = float(evals_result_list[2].split(":")[1])
    print("Final Train AUCPR:", train_aucpr)
    print("Final Val AUCPR:", val_aucpr)
    best_params['learning_rate'] = round(best_params['learning_rate'], 2)

    model_name = (
        f"{dataframe_folder}_model_lr_{best_params['learning_rate']}_"
        f"max_depth_{best_params['max_depth']}_"
        f"alpha_{best_params['alpha']}_"
        f"lambda_{best_params['lambda']}_"
        f"{args.type}")
    model_path = f"best_models_xgb/{model_name}.json"
    
    val_probs = best_model.predict(val_dmatrix_final)
    y_val_true = val_dmatrix_final.get_label()
    
    # Save this predictions and tcrids
    val_predictions = {"probs": val_probs, "true": y_val_true}
    with open(f"./best_models_xgb/val_predictions_{model_name}.pkl", "wb") as f:
        pickle.dump(val_predictions, f)

    # ROC curve
    fpr, tpr, thresholds_roc = roc_curve(y_val_true, val_probs)
    roc_auc = auc(fpr, tpr)
    roc_data = {"fpr": fpr, "tpr": tpr, "roc_auc": roc_auc, "thresholds_roc": thresholds_roc}

    # PR-curve
    precision, recall, thresholds_pr = precision_recall_curve(y_val_true, val_probs)
    auc_pr = auc(recall, precision)
    pr_data = {"precision": precision,"recall": recall,"auc_pr": auc_pr,"thresholds_pr": thresholds_pr}

    # Print AUCs
    print(f"AUC-ROC (Receiver Operating Characteristic): {roc_auc:.4f}")
    print(f"AUC-PR (Precision-Recall Curve): {auc_pr:.4f}")

    # Save data
    with open(f"./best_models_xgb/roc_data_{model_name}.pkl", "wb") as f:
        pickle.dump(roc_data, f)

    with open(f"./best_models_xgb/pr_data_{model_name}.pkl", "wb") as f:
        pickle.dump(pr_data, f)

    # Save the best model
    best_model.save_model(model_path)
    print(f"Model saved as {model_path}")

    # Save train and validation tcr_ids in a dict
    tcr_ids_dict = {"train": tcr_ids_train, "val": tcr_ids_val}
    with open(f"./best_models_xgb/tcr_ids_{model_name}.pkl", "wb") as f:
        pickle.dump(tcr_ids_dict, f)
    
    # Test set
    collected_data_val = all_data[7]
    tcra_embeddings_train, tcrb_embeddings_train, epitope_embeddings_train, mhc_embeddings_train, labels_train, structural_train,tcr_ids_train, \
    tcra_embeddings_test, tcrb_embeddings_test, epitope_embeddings_test, mhc_embeddings_test, labels_test, structural_test, tcr_ids_test = process_all_data(all_data, args.type, 7, global_average_pooling)
    X_test = create_embedding_matrix(tcra_embeddings_test, tcrb_embeddings_test, epitope_embeddings_test, mhc_embeddings_test, structural_test)
    y_test = np.array(labels_test)
    test_dmatrix_final = xgb.DMatrix(X_test, label=y_test)
    test_probs = best_model.predict(test_dmatrix_final)
    y_test_true = test_dmatrix_final.get_label()
    test_df=pd.read_csv(os.path.join(args.dataframe, f"training_fold_7.csv"))
    assert len(test_probs) == len(test_df), "Mismatch in number of predictions and test DataFrame rows"
    test_df["probs"] = test_probs
    output_path = os.path.join("./best_models_xgb", f"test_preds_{model_name}.csv")
    test_df.to_csv(output_path, index=False)
    print("Saved test dataframe to", output_path)

if __name__ == "__main__":
    main()


