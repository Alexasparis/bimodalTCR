import os
import numpy as np
import pickle
import torch
import pandas as pd
import argparse
import esm
from esm.sdk.api import ESMProtein, LogitsConfig  # Asegúrate de importar desde el módulo correcto
import h5py

def print_v(text, verbose):
    """Print verbose messages"""
    if verbose:
        print(text)

def load_esm3_model(model_path, device = "cpu"):
    """Load ESM-3 model"""
    try:
        # Load the model from the specified path
        model = torch.load(model_path)
        model.to(device)  # Ensure the model is moved to the "cpu" or "cuda"
        return model
    except Exception as e:
        print(f"Error loading model '{model_path}': {e}")
        raise

def read_csv(input_path):
    """Read a CSV file and extract TCR IDs and CDR sequences."""
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(input_path)
        chain_columns = ['TRA_aa', 'TRB_aa','MHC_seq', 'Epitope']
        sequences = []
        
        # Extract sequences and their associated TCR IDs
        for index, row in df.iterrows():
            tcr_id = row['TCR_name']
            for chain_col in chain_columns:
                sequences.append((f"{tcr_id}_{chain_col}", row[chain_col]))
        return sequences
    except Exception as e:
        print(f"Error reading CSV file '{input_path}': {e}")
        raise

def extract_features(model, data, emb_normalized, model_path="", output_path=".", verbose=False):
    """Extract embeddings using ESM-3"""
    all_last_hidden_states = {}

    for id, sequence in data:
        try:
            print_v(f"Processing sequence ID: {id}", verbose)

            # Validate that the sequence is not empty
            if not sequence or not isinstance(sequence, str):
                raise ValueError(f"Invalid sequence for ID {id}: {sequence}")

            # Create an ESMProtein object for the sequence
            protein = ESMProtein(sequence=sequence)

            # Encode the protein to get embeddings
            with torch.no_grad():
                protein_tensor = model.encode(protein)
                logits_output = model.logits(protein_tensor, LogitsConfig(sequence=True, return_embeddings=True, return_hidden_states=True, ith_hidden_layer=6))
                embeddings_tensor = logits_output.hidden_states.squeeze(1)

            # Normalize the embeddings if requested
            if emb_normalized:
                embeddings_tensor = torch.nn.functional.normalize(embeddings_tensor, dim=-1)

            # Store the embeddings in the dictionary
            all_last_hidden_states[str(id)] = embeddings_tensor

        except Exception as e:
            print(f"Error processing sequence ID {id}: {e}")
            continue  # Skip this sequence and continue with the next one

    # Save the embeddings as a pickle file
    try:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        model_name = model_path.split("/")[-1].split(".")[0]
        output_file_path = os.path.join(output_path, f"last_hidden_state_{model_name}.h5py")

        #with open(output_file_path, "wb") as f:
        #    pickle.dump(all_last_hidden_states, f)

        with h5py.File(output_file_path, 'w') as f:
            # Check if data is a dictionary and save accordingly
            if isinstance(all_last_hidden_states, dict):
                for key, value in all_last_hidden_states.items():
                    # Convert tensor to numpy array of dtype float32 before saving
                    f.create_dataset(key, data=np.array(value.cpu(), dtype=np.float32), compression="gzip")
            else:
                print("Revise this")
        print_v(f"Embeddings successfully saved to {output_file_path}", verbose)
    except Exception as e:
        print(f"Error saving embeddings to '{output_path}': {e}")
        raise

def main():
    # Define input and output arguments
    parser = argparse.ArgumentParser(description="This script extracts the embeddings, attention weights, and other results generated "
                                                 "by using an ESM-3 model and a given list of protein sequences from a CSV file. "
                                                 "This information is stored as a tensor object.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("-i", "--input_path", type=str, help="Path to the input CSV file containing TCR IDs and CDR sequences.", required=True)
    parser.add_argument("-m", "--model_path", type=str, help="ESM-3 model path (e.g., ./esm3-medium-2024-03.pth", required=True)
    parser.add_argument("-o", "--output_path", type=str, help="Path to save the output embeddings.", default="./")
    parser.add_argument("-norm", "--normalized", action="store_true", help="Normalize the embeddings.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("-d", "--device", choices=["cpu", "cuda"], default="cpu",  help="Choose the device for computation: 'cpu' or 'cuda'. Default is 'cpu'.")

    args = parser.parse_args()

    try:
        # Load the ESM-3 model
        model = load_esm3_model(args.model_path, args.device)

        # Read the input CSV file
        data = read_csv(args.input_path)

        # Extract embeddings and save results
        extract_features(model, data, args.normalized, args.model_path, args.output_path, args.verbose)

    except Exception as e:
        print(f"An error occurred in the main pipeline: {e}")

if __name__ == "__main__":
    main()
    print("\nWork completed!\n")

