# Find contact map:

# This file contains functions to find the most similar TCR given an input TCR using tcrdist3. 

# 1) get_germlines(seq:str)
# 2) find_closest_tcr(df, alpha_seq, beta_seq, epitope_seq, tcr_name, mhc_allele, mhc_seq = None, structure= True, one_value=True)


#Import libraries
from anarci import anarci
import pandas as pd
from tcrdist.repertoire import TCRrep
import os
import multiprocessing
from functools import partial
import numpy as np
import os

from utils import extract_specific_sequences, extract_sequences, parse_general_file, calculate_sequence_distance
from mapping import run_anarci, parse_CDR3, global_alignment, parse_anarci_output
from config import STRUCTURES_ANNOTATION_DIR, DATA_DIR, PDB_DIR

general_path = os.path.join(STRUCTURES_ANNOTATION_DIR, "general.txt")

### TCRdist 

def compute_alignment(pdb_file, pdb_id, PDB_DIR, seq_dict, mhc_seq_query):
    if pdb_file.endswith('.pdb') and pdb_file != f"{pdb_id}.pdb":
        pdb_idmhc = pdb_file.split('.')[0]
        mhc_seqid = seq_dict.get(pdb_idmhc, {}).get('mhc_chain', 'A')
        pdb_path = os.path.join(PDB_DIR, pdb_file)
        seqs = extract_sequences(pdb_path)

        if mhc_seqid in seqs:
            mhc_seq = seqs[mhc_seqid]
            _, _, score = global_alignment(mhc_seq_query, mhc_seq)
            return (pdb_idmhc, score)
    return None

def parallel_global_alignment(PDB_DIR, pdb_id, seq_dict, mhc_seq_query):
    """Ejecuta el alineamiento global en paralelo."""
    pdb_files = os.listdir(PDB_DIR)
    
    with multiprocessing.Pool(processes=os.cpu_count()) as pool:
        func = partial(compute_alignment, pdb_id=pdb_id, PDB_DIR=PDB_DIR, seq_dict=seq_dict, mhc_seq_query=mhc_seq_query)
        results = pool.map(func, pdb_files)

    # Filtrar los valores None y convertir en diccionario
    scores = {pdb_idmhc: score for pdb_idmhc, score in results if pdb_idmhc is not None}
    
    return scores

def get_germlines(seq:str):
    '''
    Get the VJ germlines from TCRa or TCRb sequences
    '''
    input_seq = [('seq', seq)]
    try:
        results = anarci(input_seq, scheme="imgt", output=False, assign_germline=True)
        v_gene = results[1][0][0]['germlines']['v_gene'][0][1]
        j_gene = results[1][0][0]['germlines']['j_gene'][0][1]
    except:
        v_gene = 'NA'
        j_gene = 'NA'
    return v_gene, j_gene

def find_closest_tcr(df, alpha_seq, beta_seq, epitope_seq, tcr_name, mhc_allele, seq_dict, mhc_seq=None, structure=True, one_value=True, verbose=False):
    result_string = ""  # Initialize an empty string to accumulate messages
    pdb_id = tcr_name
    df = df.copy()
    df['tcr_id'] = df['tcr_id'].astype(str)
    df['count'] = 1

    result_string += "Processing the alpha chain...\n"
    # Process alpha chain
    anarci_output_alpha = run_anarci(alpha_seq)
    parsed_anarci_output_a = parse_anarci_output(anarci_output_alpha)
    cdr3_alpha, _ = parse_CDR3(parsed_anarci_output_a)
    v_gene_alpha, j_gene_alpha = get_germlines(alpha_seq)

    result_string += "Processing the beta chain...\n"
    # Process beta chain
    anarci_output_beta = run_anarci(beta_seq)
    parsed_anarci_output_b = parse_anarci_output(anarci_output_beta)
    cdr3_beta, _ = parse_CDR3(parsed_anarci_output_b)
    v_gene_beta, j_gene_beta = get_germlines(beta_seq)

    # Create a new TCR entry
    new_row = pd.DataFrame({
        'tcr_id': [pdb_id],
        'cdr3_a_aa': [cdr3_alpha],
        'v_a_gene': [v_gene_alpha],
        'j_a_gene': [j_gene_alpha],
        'cdr3_b_aa': [cdr3_beta],
        'v_b_gene': [v_gene_beta],
        'j_b_gene': [j_gene_beta],
        'count': [1]})

    result_string += "Computing TCRrep for the current TCR...\n"
    # Compute TCRrep for the new TCR
    tr_current = TCRrep(cell_df=new_row, organism='human', chains=['alpha', 'beta'], 
                        compute_distances=True, db_file='alphabeta_gammadelta_db.tsv')
    
    info_path = os.path.join(STRUCTURES_ANNOTATION_DIR, "crystals_info.csv")
    info_seq = os.path.join(STRUCTURES_ANNOTATION_DIR, "vdjdb_filtered_34533.csv")
    
    df_info_seq = pd.read_csv(info_seq)
    df_info = pd.read_csv(info_path)

    # TCR_name and pdb_id as str
    df_info_seq['TCR_name'] = df_info_seq['TCR_name'].astype(str)
    df_info['pdb_id'] = df_info['pdb_id'].astype(str)
    
    epitope_length = len(epitope_seq)

    # Filter by MHC allele
    result_string += f"Searching for TCRs with the same MHC allele: {mhc_allele}\n"
    pdb_ids_allele_struc = df_info.loc[df_info['MHC_allele'] == mhc_allele, 'pdb_id'].values
    pdb_ids_allele_seq = df_info_seq.loc[df_info_seq['MHC_allele'] == mhc_allele, 'TCR_name'].values
    pdb_ids_allele = np.concatenate((pdb_ids_allele_struc, pdb_ids_allele_seq))
    df_filtered = df[df['tcr_id'].isin(pdb_ids_allele) & (df['tcr_id'] != pdb_id)]

    if df_filtered.empty:
        result_string += "No TCRs with the same MHC allele found. Searching in the entire dataset...\n"
        print("No TCrs with the same MHC allele found. Searching in the entire dataset...")
        if structure:  # Extract MHC sequence from the structure
            result_string += "Extracting MHC sequence from the structure...\n"
            path_pdb_file = os.path.join(PDB_DIR, f"{pdb_id}.pdb")
            seqs_query = extract_sequences(path_pdb_file)
            mhc_seq_query_id = seq_dict.get(pdb_id, {}).get('mhc_chain', 'A')
            mhc_seq_query = seqs_query[mhc_seq_query_id]
        elif mhc_seq:  # Use the provided MHC sequence
            result_string += "Using provided MHC sequence...\n"
            mhc_seq_query = mhc_seq
        else:  # Use MHC sequence from the allele
            result_string += "Using MHC sequence from the allele...\n"
            try:
                mhc_path = os.path.join(DATA_DIR, "all_mhc_seqs.csv")
                mhc_seqs_df = pd.read_csv(mhc_path)
                mhc_allele_search = mhc_allele + ":01:01"
                mhc_seq_query = mhc_seqs_df[mhc_seqs_df['mhc_allele'] == mhc_allele_search]['mhc_seq'].values[0]
            except Exception as e:
                result_string += f"Error: No sequence found in database for the provided allele. {e}\n"
        
        # Perform global alignment with all MHC sequences of the crystals
        result_string += "Performing global alignment with MHC sequences of the crystals using multiprocessing...\n"
        scores = parallel_global_alignment(PDB_DIR, pdb_id, seq_dict, mhc_seq_query)

        # Select PDBs with the highest score
        max_score = max(scores.values())
        max_pdb_ids = [pdb_id for pdb_id, score in scores.items() if score == max_score]

        # Collect MHC alleles for the selected PDBs
        mhc_alleles_max = []
        for pdb_id in max_pdb_ids:
            mhc_allele_struc = df_info[df_info['pdb_id'] == pdb_id]['MHC_allele'].values[0]
            mhc_allele_seq = df_info_seq[df_info_seq['TCR_name'] == pdb_id]['MHC_allele'].values[0]
            mhc_allele = mhc_allele_struc + mhc_allele_seq
            mhc_alleles_max.append(mhc_allele)
            
        # Unique MHC alleles
        mhc_alleles_max = list(set(mhc_alleles_max))

        # Filter the DataFrame for the selected MHC alleles
        result_string += "Filtering the dataset for selected MHC alleles...\n"
        pdb_ids = []
        for mhc_allele in mhc_alleles_max:
            pdb_ids_allele_struc = df_info[df_info['MHC_allele'] == mhc_allele]['pdb_id'].tolist()
            pdb_ids_allele_seq = df_info_seq[df_info_seq['MHC_allele'] == mhc_allele]['TCR_name'].tolist()
            pdb_ids_allele = pdb_ids_allele_struc + pdb_ids_allele_seq
            pdb_ids.extend(pdb_ids_allele)
        
        # Delete current TCR_id from pdb_ids
        pdb_ids = [pdb_id for pdb_id in pdb_ids if pdb_id != tcr_name]

        # Select TCRs with the same epitope length
        result_string += "Selecting TCRs with the same epitope length...\n"
        samelength_struc = [pdb_id for pdb_id in pdb_ids if df_info[df_info['pdb_id'] == pdb_id]['epitope_length'].values[0] == epitope_length]
        samelength_seq = [pdb_id for pdb_id in pdb_ids if len(df_info_seq[df_info_seq['TCR_name'] == pdb_id]['Epitope'].values[0]) == epitope_length]
        samelength = samelength_struc + samelength_seq

        # Filter the DataFrame by length
        df_filtered = df[df['tcr_id'].isin(samelength)]
    
    if df_filtered.empty:
        result_string += "No PDB files with similar MHC alleles and the same epitope length. Filtering by length only...\n"
        pdb_ids_samelength_struc = df_info[df_info['epitope_length'] == epitope_length]['pdb_id'].tolist()
        pdb_ids_samelength_seq = df_info_seq[df_info_seq['Epitope'].apply(lambda x: len(x)) == epitope_length]['TCR_name'].tolist()
        pdb_ids_samelength = pdb_ids_samelength_struc + pdb_ids_samelength_seq

        # Exclude current TCR_id from pdb_ids_samelength
        pdb_ids_samelength = [pdb_id for pdb_id in pdb_ids_samelength if pdb_id != tcr_name]

        # Filter scores dict to keep only those in pdb_ids_samelength
        scores_len = {pdb_id: score for pdb_id, score in scores.items() if pdb_id in pdb_ids_samelength}
        
        # Find the max score
        max_score_len = max(scores_len.values())
        max_pdb_ids_len = [len_pdb_id for len_pdb_id, score_len in scores_len.items() if score_len == max_score_len]

        mhc_alleles_max_len = []
        for len_pdb_id in max_pdb_ids_len:
            mhc_allele_len_struc = df_info[df_info['pdb_id'] == len_pdb_id]['MHC_allele'].tolist()
            mhc_allele_len_seq = df_info_seq[df_info_seq['TCR_name'] == len_pdb_id]['MHC_allele'].tolist()
            mhc_allele_len = mhc_allele_len_struc + mhc_allele_len_seq
            mhc_alleles_max_len.append(mhc_allele_len)
        
        # Unique MHC alleles
        mhc_alleles_max_len = list(set(mhc_alleles_max_len))

        # Filter the DataFrame for the selected MHC alleles
        result_string += "Filtering the dataset by selected MHC alleles based on epitope length...\n"
        pdb_ids_len = []
        for mhc_allele_len in mhc_alleles_max_len:
            pdb_ids_allele_len_struc = df_info[df_info['MHC_allele'] == mhc_allele_len]['pdb_id'].tolist()
            pdb_ids_allele_len_seq = df_info_seq[df_info_seq['MHC_allele'] == mhc_allele_len]['TCR_name'].tolist()
            pdb_ids_allele_len = pdb_ids_allele_len_struc + pdb_ids_allele_len_seq
            pdb_ids_len.extend(pdb_ids_allele_len)
            
        df_filtered = df[df['tcr_id'].isin(pdb_ids_len)]

    # Compute TCRrep distance for the filtered PDBs
    result_string += "Computing TCRrep distance for filtered PDBs...\n"
    tr = TCRrep(cell_df=df_filtered, organism='human', chains=['alpha', 'beta'], 
                compute_distances=True, db_file='alphabeta_gammadelta_db.tsv')

    tr.compute_rect_distances(df=tr_current.clone_df, df2=tr.clone_df)
    tr.alpha_beta = tr.rw_alpha + tr.rw_beta

    min_value = np.min(tr.alpha_beta)
    min_indices = np.where(tr.alpha_beta == min_value)[1]
    min_pdb_ids = tr.clone_df['tcr_id'].values[min_indices].tolist()

    distances = {
        pdb_id: calculate_sequence_distance(epitope_seq, extract_specific_sequences(os.path.join(PDB_DIR, f"{pdb_id}.pdb"), seq_dict)[2])for pdb_id in min_pdb_ids}

    min_pdb_ids = [k for k, v in distances.items() if v == min(distances.values())]
    
    result_string += f"Returning PDB with the closest TCR: {min_pdb_ids[0]}\n"
    if verbose == True:
        print(result_string)
    return min_pdb_ids[0] if one_value else min_pdb_ids

    
