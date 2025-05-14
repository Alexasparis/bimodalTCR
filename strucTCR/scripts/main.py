#!/usr/bin/env python3

# This script is used to give an score to the binding of a query TCR to a query pMHC complex.


# Example of execution: python main.py -i ../input/training_test.csv -tcrp ../models/Model-training -tcrm ../models/TCR_MHC.csv -o ../output/ -w 1 -v
import argparse
import pandas as pd
import os  
from concurrent.futures import ProcessPoolExecutor
import sys
import warnings
warnings.simplefilter("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils import parse_general_file, extract_sequences, extract_specific_sequences
from potential_calc import get_potential
from extract_contacts import filter_contacts
from mapping import run_anarci, add_imgt_mappings, map_epitope_residue, parse_anarci_output, extract_residues_and_resids, map_imgt_to_original, global_alignment, map_alignment_to_residues
from find_contact_map import find_closest_tcr
from config import STRUCTURES_ANNOTATION_DIR, DATA_DIR, CONTACT_MAPS_DIR, PDB_DIR, MODELS_DIR

def process_tcr(contact_map_dir, pdb_dir, tcr_id, alpha_seq, beta_seq, epitope_seq, tcr_p_potential, tcr_mhc_potential, mhc_df, chain_dict, mhc_allele, verbose):
    """Process a TCR and compute scores."""
    result_string = "" # Verbose
    result_string += f"\n{'-'*40}\n------ Processed TCR: {tcr_id} ------\n{'-'*40}\n"
    peptide_length = len(epitope_seq)

    results = {
        "TCR_name": None,
        **{f"score_tcr_p{i+1}": 0 for i in range(13)},  # P1 a P13 con valor 0 por defecto
        **{f"contacts_tcr_p{i+1}": 0 for i in range(13)},  # P1 a P13 con valor 0 por defecto
        "score_tcr_all": 0,
        "contacts_tcr_p_all": 0,
        "score_tcr_mhc": 0,
        "contacts_tcr_mhc": 0}
    
    # For similar contact map
    try:
        #if mhc_allele.endswith(":01:01"):
        #    allele_find = mhc_allele[:-6]
        #similar_tcr = find_closest_tcr(tcrdist_df, str(alpha_seq), str(beta_seq), str(epitope_seq), str(tcr_id), str(allele_find), chain_dict, mhc_seq=None, structure=False, one_value=True, verbose = False)
        #result_string += f"\nThe most similar TCR to {tcr_id} is {similar_tcr}."
        similar_tcr = tcr_id # To use own contact map
    except Exception as e:
        result_string += f"\nNo similar TCRs found for {tcr_id}. {e}\n"

    ##### PROCESING TCR #####
    chains = {} 

    if similar_tcr:
        chains = chain_dict.get(similar_tcr, {'tcra_chain': 'D',
                                                  'tcrb_chain': 'E',
                                                  'peptide_chain': 'C',
                                                  'b2m_chain': 'B',
                                                  'mhc_chain': 'A'})
        
        pdb_cm_path =  os.path.join(contact_map_dir, f"{similar_tcr}_contacts.csv")
        pdb_file_path = os.path.join(pdb_dir, f"{similar_tcr}.pdb")
        # See if paths exist
        if not os.path.exists(pdb_cm_path):
            print(f"Contact map not found: {pdb_cm_path}")
            return None
        if not os.path.exists(pdb_file_path):
            print(f"PDB not found: {pdb_file_path}")
            return None
        if os.path.isfile(pdb_cm_path):
            contacts_df = pd.read_csv(pdb_cm_path)
            if all(chains.values()):  
                contacts_TCR_p, contacts_TCR_MHC = filter_contacts(
                    contacts_df,
                    chains['tcra_chain'],
                    chains['tcrb_chain'],
                    chains['peptide_chain'],
                    chains['mhc_chain'],
                    remove_X=True)
        
        # Mapping with IMGT convention the similar TCR
        result_string += "\nSequences reenumbered with IMGT convention\n"
        
        alpha_pdb, beta_pdb, epitope_pdb = extract_specific_sequences(pdb_file_path, chain_dict)

        anarci_a = run_anarci (alpha_pdb)
        anarci_b = run_anarci (beta_pdb)
        
        parsed_anarci_a = parse_anarci_output(anarci_a)
        parsed_anarci_b = parse_anarci_output(anarci_b)
        
        residues_a = extract_residues_and_resids(pdb_file_path, chains['tcra_chain'])
        residues_b = extract_residues_and_resids(pdb_file_path, chains['tcrb_chain'])
        
        mapping_a = map_imgt_to_original(parsed_anarci_a, residues_a)
        mapping_b = map_imgt_to_original(parsed_anarci_b, residues_b)

        imgt_mappings = {similar_tcr: {chains['tcra_chain']: mapping_a, chains['tcrb_chain']: mapping_b}}
        contacts_TCR_p = add_imgt_mappings(contacts_TCR_p, imgt_mappings)
        contacts_TCR_MHC = add_imgt_mappings(contacts_TCR_MHC, imgt_mappings)
        
        # Process input TCR
        result_string += f"\nMapping TCR {tcr_id} into {similar_tcr}\n"
        anarci_input_a = run_anarci (alpha_seq)
        anarci_input_b = run_anarci (beta_seq)
        
        parsed_input_a = parse_anarci_output(anarci_input_a)
        parsed_input_b = parse_anarci_output(anarci_input_b)
        
        imgt_dict_a = dict(parsed_input_a)
        imgt_dict_b = dict(parsed_input_b)

        contacts_TCR_p[tcr_id] = contacts_TCR_p.apply(lambda row: imgt_dict_a.get(row['imgt_from'], None) if row['chain_from'] == chains['tcra_chain']
                            else imgt_dict_b.get(row['imgt_from'], None) if row['chain_from'] == chains['tcrb_chain']
                            else None, axis=1)
        
        contacts_TCR_MHC[tcr_id] = contacts_TCR_MHC.apply(lambda row: imgt_dict_a.get(row['imgt_from'], None) if row['chain_from'] == chains['tcra_chain']
                            else imgt_dict_b.get(row['imgt_from'], None) if row['chain_from'] == chains['tcrb_chain']
                            else None, axis=1)

    ##### PROCESSING EPITOPE #####
        result_string += f"\n-> Processed Epitope: {epitope_seq}\n"
        contacts_TCR_p['epitope'] = contacts_TCR_p.apply(lambda row: map_epitope_residue(row, epitope_seq), axis=1)

        # Add TCR-Potential (looping for P1-P9)
        result_string += "\nCalculated TCR-peptide potential\n"
        total_scores = {}
        # Calculate potential scores and totals for each position
        try:
            sum_total_potential = 0
            if peptide_length > 13 or peptide_length < 8:
                raise ValueError(f"Unsupported peptide length: {peptide_length}. Max supported length is 13.")

            elif peptide_length <= 10:
                for i in range(1, peptide_length+1):
                    potential_column = f'potential_P{i}'
                    potential_column_contacts = f'contacts_P{i}'

                    # Apply the potential function based on the residue assignment
                    contacts_TCR_p[potential_column] = contacts_TCR_p.apply(
                        lambda row: get_potential(row, tcr_p_potential[f'tcr_p_potential_P{i}'], tcr_id, "epitope")
                        if row['resid_to'] == i else 0, axis=1)
                    
                    # Sum the potential scores for each position
                    total_scores[potential_column] = contacts_TCR_p[potential_column].sum(skipna=True)
                    sum_total_potential += total_scores[potential_column]
                    total_contacts_position = (contacts_TCR_p['resid_to'] == i).sum()
                    total_scores[potential_column_contacts] = total_contacts_position

                    # Add the total score to the result string
                    result_string += f"Total score for P{i} with contacts ({total_contacts_position}): {total_scores[potential_column]}\n"
            else:
                assignments_dict = {11: {1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 5, 8: 8, 9: 5, 10: 9, 11: 10},
                                12: {1: 1, 2: 2, 3: 3, 4: 2, 5: 5, 6: 7, 7: 7, 8: 6, 9: 9, 10: 9, 11: 9, 12: 10},
                                13: {1: 1, 2: 2, 3: 3, 4: 6, 5: 3, 6: 4, 7: 8, 8: 6, 9: 8, 10: 9, 11: 9, 12: 9, 13: 10}}
                
                assignments = assignments_dict.get(int(peptide_length))
                for i in range(1, peptide_length+1):
                    potential_column = f'potential_P{i}'
                    potential_column_contacts = f'contacts_P{i}'

                    real_position = assignments.get(i)
                    contacts_TCR_p[potential_column] = contacts_TCR_p.apply(
                        lambda row: get_potential(row, tcr_p_potential[f'tcr_p_potential_P{real_position}'], tcr_id, "epitope")
                        if row['resid_to'] == i else 0, axis=1)
                    
                    # Sum the potential scores for each position
                    total_scores[potential_column] = contacts_TCR_p[potential_column].sum(skipna=True)
                    sum_total_potential += total_scores[potential_column]
                    total_contacts_position = (contacts_TCR_p['resid_to'] == i).sum()
                    total_scores[potential_column_contacts] = total_contacts_position

                    # Add the total score to the result string
                    result_string += f"Total score for P{i} with contacts ({total_contacts_position}): {total_scores[potential_column]}\n"

        except Exception as e:
            print("Error calculating TCR-p potential")

        # Calculate total score for all positions
        total_contacts_tcr = len(contacts_TCR_p)
        total_score_all = sum_total_potential/total_contacts_tcr
        result_string += f"Total score for all positions normalized by the number of total contacts ({total_contacts_tcr}): {total_score_all}\n"

    #### PROCESSING MHC ####
        if mhc_allele:
            try:
                mhc_seq_match = mhc_df[mhc_df['mhc_allele'] == mhc_allele]
                if not mhc_seq_match.empty:
                    mhc_seq = mhc_seq_match['mhc_seq'].values[0]
                else:
                    print(f"Warning: No MHC sequence found for allele {mhc_allele}")
                    mhc_seq = None
                    return None
                
                result_string += f"\n-> Processing MHC-I allele: {mhc_allele}\n"
                seq_pdb = extract_sequences(pdb_file_path)
                aligned_seq_pdb, aligned_seq_query, score = global_alignment(seq_pdb[chains['mhc_chain']], mhc_seq)
                residues_M = extract_residues_and_resids(pdb_file_path, chains['mhc_chain']) 
                mapped_residues = map_alignment_to_residues(aligned_seq_pdb, aligned_seq_query, residues_M)
                    
                df_tuples = pd.DataFrame(mapped_residues, columns=['resid', 'mhc_pdb', mhc_allele])
                contacts_TCR_MHC = pd.merge(
                        contacts_TCR_MHC, 
                        df_tuples[['resid', mhc_allele]], 
                        left_on='resid_to', 
                        right_on='resid', 
                        how='left')
                    
                contacts_TCR_MHC = contacts_TCR_MHC.drop(columns=['resid'])

                # Add TCR-MHC potential
                result_string += "\nCalculating TCR-MHC-I potential\n"
                contacts_TCR_MHC['potential'] = contacts_TCR_MHC.apply(
                        lambda row: get_potential(row, tcr_mhc_potential, tcr_id, mhc_allele), axis=1)
                    
                contacts_TCR_MHC['potential'] = pd.to_numeric(contacts_TCR_MHC['potential'], errors='coerce')
                total_contacts_mhc=len(contacts_TCR_MHC)
                sum_total_potential = contacts_TCR_MHC['potential'].sum(skipna=True)
                total_score_mhc = sum_total_potential/total_contacts_mhc
                result_string +=f"Total score for TCR-MHC {mhc_allele} normalized by the total number of contacts ({total_contacts_mhc}): {total_score_mhc}"

            except Exception as e:
                print(f"Error processing MHC sequence for allele {mhc_allele}: {e}")
                mhc_seq = None

        else:
            result_string += "\nNo MHC allele provided. Skipping MHC processing.\n"
            total_score_mhc = None

        # Append results to the results container
        results["TCR_name"] = tcr_id

        # Assign scores and contact counts for positions P1 to P13
        for i in range(1, 14):
            results[f"score_tcr_p{i}"] = round(total_scores.get(f"potential_P{i}", 0), 4)  # Format to 4 decimal places
            results[f"contacts_tcr_p{i}"] = total_scores.get(f"contacts_P{i}", 0)  # Default to 0 if not found

        # Assign total scores and contact counts
        results["score_tcr_all"] = round(total_score_all if total_score_all else 0, 4)
        results["contacts_tcr_p_all"] = total_contacts_tcr if total_contacts_tcr else 0

        # Assign MHC-related scores and contact counts
        results["score_tcr_mhc"] = round(sum_total_potential if sum_total_potential else 0, 4)
        results["contacts_tcr_mhc"] = total_contacts_mhc if total_contacts_mhc else 0
    else:
        result_string += f"\nNo similar TCR found for input TCR. Skipping {tcr_id}.\n"
    
    if verbose:
        print(result_string)

    return results

def main():
    parser = argparse.ArgumentParser(description='Process input TCRs and rank them based on statistic potential.')
    parser.add_argument("-i", "--input_file", type=str, required=True, help='Input CSV file path with TCRs to process.')
    parser.add_argument("-cmd", "--contact_maps_folder", type=str, required=True, default= CONTACT_MAPS_DIR ,help='Folder with contact maps.')
    parser.add_argument("-pdb", "--pdb_folder", type=str, required=True, default=PDB_DIR,help='Folder with PDB files.')
    parser.add_argument("-tcrp", "--tcr_potential_folder", type=str, required=True, help="TCR-peptide potential.")
    parser.add_argument("-tcrm", "--mhc_potential", type=str, required=True, default=os.path.join(MODELS_DIR, "TCR_MHC.csv"), help='TCR-MHC potential.')
    parser.add_argument("-o", "--output_file", type=str, required=False, default="output.csv", help='Output path for CSV file containing scores')
    #parser.add_argument("-tcrdist", "--tcrdist_df", type=str, required=False, default=None, help='TCRdist dataframe path.')
    parser.add_argument("-w", "--max_workers", type=int, required=False, default=os.cpu_count(), help='Num of workers.')
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help='Verbose mode')
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    tcr_df = pd.read_csv(args.input_file)    
    chain_dict = parse_general_file(os.path.join(STRUCTURES_ANNOTATION_DIR, "chain_info.txt"))
    mhc_df = pd.read_csv(os.path.join(DATA_DIR, "all_mhc_seqs.csv"))
    tcr_df['TCR_name'] = tcr_df['TCR_name'].astype(str)
    #tcrdist_df = pd.read_csv(args.tcrdist_df)

    # Load models acording to epitope_length
    print("Loading TCR-MHC model...")
    tcr_mhc_potential = pd.read_csv(args.mhc_potential)

    print("Loading TCR-p models")
    # Get the CSV files from the appropriate folder
    models_dict={}
    for folder in os.listdir(args.tcr_potential_folder):
        folder_path = os.path.join(args.tcr_potential_folder, folder)
        if not os.path.isdir(folder_path):
            continue
        target_length = folder.split('-')[-1].replace('L', '')
        if target_length not in models_dict:
            models_dict[target_length] = {}
        csv_files = [os.path.join(args.tcr_potential_folder, folder, f) for f in os.listdir(os.path.join(args.tcr_potential_folder, folder)) if f.endswith('.csv')]
        if not csv_files:
            print(f"No CSV files found in the directory: {args.tcr_potential_folder}/{folder}")
        else:
            for file in csv_files:
                try:
                    file_name = os.path.basename(file)
                    potential_number = file_name.split('_P')[-1].split('.csv')[0]
                    potential_key = f'tcr_p_potential_P{potential_number}'
                    models_dict[target_length][potential_key] = pd.read_csv(file)
                except Exception as e:
                    print(f"Failed to load {file}: {e}")

    # Process each TCR in parallel
    results = []
    with ProcessPoolExecutor(max_workers=int(args.max_workers)) as executor:
        for index, row in tcr_df.iterrows():
            tcr_id = row['TCR_name']
            alpha_seq = row['TCRA']
            beta_seq = row['TCRB']
            epitope_seq = row['Epitope']
            mhc_allele = row['MHC_allele']
            if mhc_allele:
                mhc_allele = mhc_allele + ":01:01"
            epitope_length = len(epitope_seq)
            len_search = str(epitope_length) if epitope_length < 11 else 11
            tcr_p_potential = models_dict.get(str(len_search), None)
            try:
                result = executor.submit(process_tcr, args.contact_maps_folder, args.pdb_folder, tcr_id, alpha_seq, beta_seq, epitope_seq, tcr_p_potential, tcr_mhc_potential, mhc_df, chain_dict, mhc_allele, args.verbose)
                results.append(result)

            except Exception as e:
                print(f"Error processing TCR {tcr_id}: {e}")
                continue

    # Collect and save results
    results_list = []
    for r in results:
        try:
            res = r.result()
            if res is not None:
                results_list.append(res)
            else:
                print("[Warning] A TCR processing result was None.")
        except Exception as e:
            print(f"[Error] Exception while collecting result: {e}")
    
    # results_list = [r.result() for r in results]
    results_df = pd.DataFrame(results_list)

    # Merge results with input data on TCR_name
    results_df['TCR_name'] = results_df['TCR_name'].astype(str)
    merged = pd.merge(tcr_df, results_df, on='TCR_name', how='left')

    # Save the results as a CSV in the specified file path
    merged.to_csv(args.output_file, index=False)
    print(f"\n\nResults saved to {args.output_file}")

if __name__ == "__main__":
    main()

