
# This script extracts the embeddings of the Var and CDRs from the full TCR chain embeddings.
# It requires the ANARCI tool to be installed and available in the PATH.
# The input is a .csv file containing the TCR chain sequences and a .pkl file containing the embeddings of the TCR chain sequences.
# The output is a .pkl file containing the embeddings of the CDRs.

import h5py
import numpy as np
import subprocess
import argparse
import pandas as pd
import pickle
import argparse

def run_anarci(sequence):
    try:
        if '*' in sequence:
            print("Warning: Stop codon (*) found in the sequence. Removing it...")
            sequence = sequence.replace('*', '')
            
        command = f"ANARCI -i {sequence} --scheme imgt"
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr}")
        return e.stdout

def parse_anarci_output(anarci_output):
    """
    Parse the output of ANARCI to extract IMGT numbering and residues, ensuring uniqueness of IMGT numbers.
    
    Args:
        anarci_output (str): Output from ANARCI as a string.
    
    Returns:
        list of tuples: A list where each tuple contains (IMGT_number, residue).
    """
    lines = anarci_output.split('\n')
    imgt_numbered_seq = []
    
    for line in lines:
        if line.startswith('#') or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 3:
            chain_letter = parts[0]
            imgt_num = int(parts[1])  # IMGT number
            residue = parts[-1]        # Residue
            imgt_numbered_seq.append((imgt_num, residue))
    return imgt_numbered_seq

def search_variable_region(protein_sequence, parsed_anarci_output):
    """
    Search the full variable region in the protein sequence and return the start and end positions.

    Args:
        protein_sequence (str): The full protein sequence.
        parsed_anarci_output (list of tuples): The parsed ANARCI output as a list of (IMGT_number, residue).

    Returns:
        tuple: (start_index, end_index) representing the full variable region in 0-indexed format.
    """
    # Extract the full variable region sequence from ANARCI output
    var_sequence = ''.join(residue for _, residue in parsed_anarci_output if residue != "-")

    # Find the full variable region within the protein sequence
    var_start = protein_sequence.find(var_sequence)
    
    if var_start == -1:
        raise ValueError("Full variable region not found in the protein sequence.")

    var_end = var_start + len(var_sequence) - 1
    
    return var_sequence, var_start, var_end

def extract_cdrs(protein_sequence, parsed_anarci_output):
    """
    Extrae la secuencia del CDR3 y sus posiciones en la secuencia completa.

    Args:
        protein_sequence (str): La secuencia completa de la proteína.
        parsed_anarci_output (list of tuples): Lista de tuplas (IMGT_number, residuo) de la salida de ANARCI.

    Returns:
        tuple: (cdr3_sequence, start_index, end_index), donde start_index y end_index están en formato 0-indexed.
    """
    cdrs= {}
    # Extraer los residuos de CDR3 según la numeración IMGT
    cdr1_residues = ''.join(residue for imgt_num, residue in parsed_anarci_output if 27 <= imgt_num <= 38 and residue != "-")
    cdr2_residues = ''.join(residue for imgt_num, residue in parsed_anarci_output if 56 <= imgt_num <= 65 and residue != "-")
    cdr3_residues = ''.join(residue for imgt_num, residue in parsed_anarci_output if 104 <= imgt_num <= 118 and residue != "-")

    # Buscar la secuencia de CDR3 en la secuencia completa
    cdr1_start = protein_sequence.find(cdr1_residues)
    cdr2_start = protein_sequence.find(cdr2_residues)
    cdr3_start = protein_sequence.find(cdr3_residues)
    if cdr1_start == -1:
        raise ValueError("CDR1 no encontrado en la secuencia de la proteína.")
    if cdr2_start == -1:
        raise ValueError("CDR2 no encontrado en la secuencia de la proteína.")
    if cdr3_start == -1:
        raise ValueError("CDR3 no encontrado en la secuencia de la proteína.")
    cdr1_end = cdr1_start + len(cdr1_residues) - 1
    cdr2_end = cdr2_start + len(cdr2_residues) - 1
    cdr3_end = cdr3_start + len(cdr3_residues) - 1
    cdrs["cdr1"] = {"sequence": cdr1_residues, "start": cdr1_start, "end": cdr1_end}
    cdrs["cdr2"] = {"sequence": cdr2_residues, "start": cdr2_start, "end": cdr2_end}
    cdrs["cdr3"] = {"sequence": cdr3_residues, "start": cdr3_start, "end": cdr3_end}

    return cdrs

def select_embeddings (tensor, start_index, end_index):
    """
    Select the embeddings corresponding to the CDRs.

    Args:
        tensor (torch.Tensor): The tensor containing the embeddings.
        start_index (int): The start index of the CDR.
        end_index (int): The end index of the CDR.
    """
    return tensor[:, start_index:end_index+1, :]


def main():
    argparser = argparse.ArgumentParser(description="Extract CDRs embeddings from full TCR chain embeddings.")
    argparser.add_argument("-df", "--dataframe", required=True, help="The .csv file containing the TCR chain sequences.")
    argparser.add_argument("-emb", "--embeddings", required=True, help="The .pkl file containing the embeddings of the TCR chain sequences.")
    argparser.add_argument("-o", "--output", required=True, help="The output file to save the embeddings of the CDRs.")
    args = argparser.parse_args()

    df = pd.read_csv(args.dataframe)
    
    embeddings = {}
    with h5py.File(args.embeddings, 'r') as f:
        for key in f.keys():
            embeddings[key] = f[key][:]
            
    print("Length of embeddings keys", len(embeddings))
    
    extracted_embeddings = {}

    for index, row in df.iterrows():
        try:
            tcr_id = row['TCR_name']
            print(f"Processing TCR {tcr_id}...")
            tra_seq = row['TRA_aa']
            trb_seq = row['TRB_aa']

            key_a = f"{tcr_id}_TRA_aa"
            key_b = f"{tcr_id}_TRB_aa"
            
            if key_a not in embeddings or key_b not in embeddings:
                print(f"Warning: Embeddings for {tcr_id} not found. Skipping...")
                continue

            emb_a = embeddings[key_a]
            emb_b = embeddings[key_b]

            print(f"Extracting CDRs embeddings for alpha chain...")
            anarci_a = run_anarci(tra_seq)
            if not anarci_a: 
                print(f"Warning: ANARCI failed for {tcr_id} TRA sequence. Skipping...")
                continue
            parsed_a = parse_anarci_output(anarci_a)
            var_seq_a, start_var_a, end_var_a = search_variable_region(tra_seq, parsed_a)
            
            cdrs_a = extract_cdrs(tra_seq, parsed_a)
            start_1a, end_1a = cdrs_a["cdr1"]["start"], cdrs_a["cdr1"]["end"]
            start_2a, end_2a = cdrs_a["cdr2"]["start"], cdrs_a["cdr2"]["end"]
            start_3a, end_3a = cdrs_a["cdr3"]["start"], cdrs_a["cdr3"]["end"]

            
            emb_var_a = select_embeddings(emb_a, start_var_a+1, end_var_a+1)
            emb_cdr1_a = select_embeddings(emb_a, start_1a+1, end_1a+1)
            emb_cdr2_a = select_embeddings(emb_a, start_2a+1, end_2a+1)
            emb_cdr3_a = select_embeddings(emb_a, start_3a+1, end_3a+1)
            print(f"CDRs embeddings extracted for {tcr_id} chain alpha.")

            print(f"Extracting CDRs embeddings for beta chain...")
            anarci_b = run_anarci(trb_seq)
            if not anarci_b:  
                print(f"Warning: ANARCI failed for {tcr_id} TRB sequence. Skipping...")
                continue
            parsed_b = parse_anarci_output(anarci_b)
            var_seq_b, start_var_b, end_var_b = search_variable_region(trb_seq, parsed_b)
            
            cdrs_b = extract_cdrs(trb_seq, parsed_b)
            start_1b, end_1b = cdrs_b["cdr1"]["start"], cdrs_b["cdr1"]["end"]
            start_2b, end_2b = cdrs_b["cdr2"]["start"], cdrs_b["cdr2"]["end"]
            start_3b, end_3b = cdrs_b["cdr3"]["start"], cdrs_b["cdr3"]["end"]

            emb_var_b = select_embeddings(emb_b, start_var_b+1, end_var_b +1)
            emb_cdr1_b = select_embeddings(emb_b, start_1b+1, end_1b+1)
            emb_cdr2_b = select_embeddings(emb_b, start_2b+1, end_2b+1)
            emb_cdr3_b = select_embeddings(emb_b, start_3b+1, end_3b+1)
            print(f"CDRs embeddings extracted for {tcr_id} chain beta.")

            extracted_embeddings[f"{tcr_id}_vara"] = emb_var_a
            extracted_embeddings[f"{tcr_id}_cdr1a"] = emb_cdr1_a
            extracted_embeddings[f"{tcr_id}_cdr2a"] = emb_cdr2_a
            extracted_embeddings[f"{tcr_id}_cdr3a"] = emb_cdr3_a

            extracted_embeddings[f"{tcr_id}_varb"] = emb_var_b
            extracted_embeddings[f"{tcr_id}_cdr1b"] = emb_cdr1_b
            extracted_embeddings[f"{tcr_id}_cdr2b"] = emb_cdr2_b
            extracted_embeddings[f"{tcr_id}_cdr3b"] = emb_cdr3_b

        except Exception as e:
            print(f"Error processing TCR {tcr_id}: {e}. Skipping...")
            continue
    
    import traceback

    with h5py.File(args.output, 'w') as f:
        for key, value in extracted_embeddings.items():
            try:
                f.create_dataset(key, data=np.array(value, dtype=np.float32), compression="gzip")
            except Exception as e:
                print(f"[ERROR] Could not create dataset for key: {key}")
                print(f"Value type: {type(value)}, Value length: {len(value) if hasattr(value, '__len__') else 'N/A'}")
                print("Traceback:")
                traceback.print_exc()

if __name__ == "__main__":
    main()
