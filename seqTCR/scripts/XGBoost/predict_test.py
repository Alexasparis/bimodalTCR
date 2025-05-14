#!/usr/bin/env python3

# This script is used to predict the binding affinity of TCRs to MHC-peptide complexes using a pre-trained XGBoost model.

# The script requires the following arguments:

# -emb: Path to the embeddings folder. The folder should contain the embeddings of the TCRs and MHC-peptide complexes in the format of h5py files named last_hidden_state_esmc_600m.h5py and cdrs.h5py.
# -model: Path to the pre-trained XGBoost model.
# -type: Type of sequences to use for prediction (var, cdrs, cdr3s, cdr3a, cdr3b).
# -out: Path to save the predictions in a CSV file.

# The script will output a CSV file containing the predictions of the binding affinity of TCRs to MHC-peptide complexes in the format of TCR_name,prediction.

# Load libraries
import h5py
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import Booster
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def global_average_pooling(embedding, tcr_id=None):
    if embedding.size > 0:
        return np.mean(embedding, axis=1)
    else:
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
    try:
        with h5py.File(file_path, 'r') as f:
            for key in f.keys():
                embeddings[key] = f[key][:]
        return embeddings
    except Exception as e:
        print(f"Error reading embeddings file {embeddings}: {e}")
    return embeddings

def get_embedding(data, keys):
    return [global_average_pooling(np.array(data[k])) for k in keys]

def main():
    parser = argparse.ArgumentParser(description='Predict the binding affinity of TCRs to MHC-peptide complexes using a pre-trained XGBoost model.')
    parser.add_argument('-emb', type=str, help='Path to the embeddings folder.')
    parser.add_argument('-df', type=str, help='Path to the dataframe containing the TCRs and MHC-peptide complexes.')
    parser.add_argument('-model', type=str, help='Path to the pre-trained XGBoost model.')
    parser.add_argument('-type', type=str, help='Type of sequences to use for prediction (var, cdrs, cdr3s, cdr3a, cdr3b).')
    parser.add_argument('-out', type=str, help='Path to save the predictions.')
    args = parser.parse_args()

    embeddings_folder = args.emb
    embeddings_path = f"{embeddings_folder}/last_hidden_state_esmc_600m.h5py"
    cdrs_path = f"{embeddings_folder}/cdrs.h5py"
    print("Loading embeddings from", embeddings_path)
    print("Loading cdrs from", cdrs_path)
    type_seq= args.type

    # Load embeddings of sequences
    embeddings = load_embeddings(embeddings_path)
    cdrs = load_embeddings(cdrs_path)
    collected_data = collect_sequences_and_convert_to_arrays(embeddings, cdrs)

    print("Loaded TCRs:", len(collected_data))

    tcra_embeddings, tcrb_embeddings, mhc_embeddings, epitope_embeddings, tcr_ids = [], [], [], [], []

    if args.type == "var":
        # Process the training data
        for tcr_id, data in collected_data.items():
            if all(key in data for key in ["MHC_seq", "Epitope", "vara", "varb"]):
                tcra_emb = global_average_pooling(np.array(data["vara"]), tcr_id)
                tcrb_emb = global_average_pooling(np.array(data["varb"]), tcr_id)
                epitope_emb = global_average_pooling(np.array(data["Epitope"]), tcr_id)
                mhc_emb = global_average_pooling(np.array(data["MHC_seq"]), tcr_id)
                if any(x is None or x.size == 0 for x in [tcra_emb, tcrb_emb, epitope_emb, mhc_emb]):
                    continue
                tcra_embeddings.append(tcra_emb)
                tcrb_embeddings.append(tcrb_emb)
                epitope_embeddings.append(epitope_emb)
                mhc_embeddings.append(mhc_emb)
                tcr_ids.append(tcr_id)

    elif args.type == "cdrs":
        for tcr_id, data in collected_data.items():
            if all(key in data for key in ["MHC_seq", "Epitope", "cdr1a", "cdr1b", "cdr2a", "cdr2b", "cdr3a", "cdr3b"]):
                emb_cdr1a = global_average_pooling(np.array(data["cdr1a"]))
                emb_cdr1b = global_average_pooling(np.array(data["cdr1b"]))
                emb_cdr2a = global_average_pooling(np.array(data["cdr2a"]))
                emb_cdr2b = global_average_pooling(np.array(data["cdr2b"]))
                emb_cdr3a = global_average_pooling(np.array(data["cdr3a"]))
                emb_cdr3b = global_average_pooling(np.array(data["cdr3b"]))
                emb_epitope = global_average_pooling(np.array(data["Epitope"]), tcr_id)
                emb_mhc = global_average_pooling(np.array(data["MHC_seq"]), tcr_id)
                if any(x is None or x.size == 0 for x in [emb_cdr1a, emb_cdr1b, emb_cdr2a, emb_cdr2b, emb_cdr3a, emb_cdr3b, emb_epitope, emb_mhc]):
                    continue
                concat_tra = np.concatenate([emb_cdr1a, emb_cdr2a, emb_cdr3a], axis=1)
                concat_trb = np.concatenate([emb_cdr1b, emb_cdr2b, emb_cdr3b], axis=1)
                tcra_embeddings.append(concat_tra)
                tcrb_embeddings.append(concat_trb)
                epitope_embeddings.append(emb_epitope)  # Epitope first
                mhc_embeddings.append(emb_mhc)  # MHC second
                tcr_ids.append(tcr_id)


    elif args.type == "cdr3s":
        for tcr_id, data in collected_data.items():
            if all(key in data for key in ["MHC_seq", "Epitope", "cdr3a", "cdr3b"]):
                emb_cdr3a = global_average_pooling(np.array(data["cdr3a"]), tcr_id)
                emb_cdr3b = global_average_pooling(np.array(data["cdr3b"]), tcr_id)
                emb_epitope = global_average_pooling(np.array(data["Epitope"]), tcr_id)
                emb_mhc = global_average_pooling(np.array(data["MHC_seq"]), tcr_id)
                if any(x is None or x.size == 0 for x in [emb_cdr3a, emb_cdr3b, emb_epitope, emb_mhc]):
                    continue
                tcra_embeddings.append(emb_cdr3a)
                tcrb_embeddings.append(emb_cdr3b)
                epitope_embeddings.append(emb_epitope)
                mhc_embeddings.append(emb_mhc)
                tcr_ids.append(tcr_id)
        
    elif args.type == "cdr3b":
        for tcr_id, data in collected_data.items():
            if all(key in data for key in ["MHC_seq", "Epitope", "cdr3b"]):
                emb_cdr3b = global_average_pooling(np.array(data["cdr3b"]), tcr_id)
                emb_epitope = global_average_pooling(np.array(data["Epitope"]), tcr_id)
                emb_mhc = global_average_pooling(np.array(data["MHC_seq"]), tcr_id)
                if any(x is None or x.size == 0 for x in [emb_cdr3b, emb_epitope, emb_mhc]):
                    continue
                tcrb_embeddings.append(emb_cdr3b)
                epitope_embeddings.append(emb_epitope)
                mhc_embeddings.append(emb_mhc)
                tcr_ids.append(tcr_id)

    elif args.type == "cdr3a":
        # Process the training data for "cdr3a"
        for tcr_id, data in collected_data.items():
            if all(key in data for key in ["MHC_seq", "Epitope", "cdr3a"]):
                emb_cdr3a = global_average_pooling(np.array(data["cdr3a"]), tcr_id)
                emb_epitope = global_average_pooling(np.array(data["Epitope"]), tcr_id)
                emb_mhc = global_average_pooling(np.array(data["MHC_seq"]), tcr_id)

                if any(x is None or x.size == 0 for x in [emb_cdr3a, emb_epitope, emb_mhc]):
                    continue

                tcra_embeddings.append(emb_cdr3a)
                epitope_embeddings.append(emb_epitope)
                mhc_embeddings.append(emb_mhc)
                tcr_ids.append(tcr_id)

    tcr_ids = np.array(tcr_ids)
    print("Shapes:")
    print("TCRA:", np.array(tcra_embeddings).shape)
    print("TCRB:", np.array(tcrb_embeddings).shape)
    print("Epitope:", np.array(epitope_embeddings).shape)
    print("MHC:", np.array(mhc_embeddings).shape)

    if len(tcra_embeddings) == 0:
        X = np.concatenate([
                    np.array(tcrb_embeddings).squeeze(),
                    np.array(epitope_embeddings).squeeze(),
                    np.array(mhc_embeddings).squeeze()],
                    axis=1)
    elif len(tcrb_embeddings) == 0:
        X = np.concatenate([
                    np.array(tcra_embeddings).squeeze(),
                    np.array(epitope_embeddings).squeeze(),
                    np.array(mhc_embeddings).squeeze()],
                    axis=1)
    else:
        X = np.concatenate([
                    np.array(tcra_embeddings).squeeze(),
                    np.array(tcrb_embeddings).squeeze(),
                    np.array(epitope_embeddings).squeeze(),
                    np.array(mhc_embeddings).squeeze()],
                    axis=1)

    data_matrix = xgb.DMatrix(X)
    tcr_ids = np.array(tcr_ids)

    # Load the pre-trained model
    print("Loading model...")
    model_path = args.model
    model = Booster()
    model.load_model(model_path)

    # Predict the binding affinity of TCRs to MHC-peptide complexes
    print("Predicting...")
    predictions_proba = model.predict(data_matrix)
    df_predictions = pd.DataFrame({'TCR_name': tcr_ids, 'prediction': predictions_proba})
    
    # Merge with df 
    df = pd.read_csv(args.df)
    df = df.merge(df_predictions, on="TCR_name", how="left")
    df.to_csv(args.out, index=False)
    print("Predictions saved to", args.out)

if __name__ == "__main__":
    main()
