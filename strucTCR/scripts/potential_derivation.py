#!/usr/bin/env python3

# This script is used to train the model given a database of contact_maps.

# Example of execution: python potential_derivation.py -cm ../data/contact_maps -pdb ../data/all_human_aq -out ../models
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


import pandas as pd
import copy
from potential_calc import calculate_potential
from utils import parse_general_file, extract_specific_sequences
from extract_contacts import filter_contacts
from config import STRUCTURES_ANNOTATION_DIR, MODELS_DIR, PDB_DIR, CONTACT_MAPS_DIR
import argparse

def create_empty_contact_model(length):
    """Creates an empty contact model for a given peptide length."""
    return {f'P{i+1}': pd.DataFrame() for i in range(length)}

def main():
    parser = argparse.ArgumentParser(description="Trains model given a database of contact maps")
    parser.add_argument("-cm", "--contact_maps", required=True, default=CONTACT_MAPS_DIR, help="Path to the folder containing contact_maps for training.")
    parser.add_argument("-pdb", "--pdb_dir", required=False, default= PDB_DIR , help="Path to the folder containing PDB files.")
    parser.add_argument("-out", "--output_folder", required=False, default= MODELS_DIR , help="Path to save the models folder.")
    parser.add_argument("-t", "--type", required=False, default = "pure",  help="pure, 9+all")
    args = parser.parse_args()

    #### TRAIN TCR-p POTENTIAL ######

    # Dictionary to store models by peptide length
    contact_models = {}

    # Parse chain information from the general.txt file
    general_path = os.path.join(STRUCTURES_ANNOTATION_DIR, "chain_info.txt")
    chain_dict = parse_general_file(general_path)

    # Path to the folder containing contact maps
    folder_path = args.contact_maps
    contact_files = [
        f for f in os.listdir(folder_path)
        if f.endswith('_contacts.csv')]
    
    print(len(contact_files), "contact files found in the folder.")

    # Process each contact file
    if args.type == "pure":
        contact_maps_11 = []
        for contact_file in contact_files:
            pdb_id = contact_file.split('_')[0]
            pdb_path = os.path.join(args.pdb_dir, f"{pdb_id}.pdb")
            _, _, peptide_sequence = extract_specific_sequences(pdb_path, chain_dict)
            peptide_length = len(peptide_sequence)
            if peptide_length <= 10 and peptide_length >= 8:
                print(f"Processing contacts for contact map: {pdb_id}")
                
                # Read the contacts for the current PDB ID
                contacts = pd.read_csv(os.path.join(folder_path, contact_file))

                # Check if the pdb_id exists in chain_dict, if not, use the default settings
                chains = chain_dict.get(pdb_id, {
                    'tcra_chain': 'D',
                    'tcrb_chain': 'E',
                    'peptide_chain': 'C',
                    'b2m_chain': 'B',
                    'mhc_chain': 'A'})
                
                chain_dict[pdb_id] = chains

                if chains and all(chains.values()):
                    try:
                        # Filter contacts for TCR-peptide and TCR-MHC
                        contacts_TCR_p, contacts_TCR_MHC = filter_contacts(
                            contacts,
                            chains['tcra_chain'],
                            chains['tcrb_chain'],
                            chains['peptide_chain'],
                            chains['mhc_chain'])

                        if not contacts_TCR_p.empty:
                            if peptide_length not in contact_models:
                                contact_models[peptide_length] = create_empty_contact_model(peptide_length)

                            # Distribute contacts into appropriate positions (P1, P2, ..., Pn)
                            for _, contact in contacts_TCR_p.iterrows():
                                resid_to = contact['resid_to']

                                # Ensure resid_to falls within the peptide length
                                if 1 <= resid_to <= peptide_length:
                                    position = f'P{resid_to}'
                                    contact_models[peptide_length][position] = pd.concat([contact_models[peptide_length][position], contact.to_frame().T])
                    except Exception as e:
                        print(f"Error processing contact map {pdb_id}: {e}")
            elif peptide_length >= 11:
                contact_maps_11.append(contact_file)
            else:
                print(f"Peptide length {peptide_length} for {pdb_id} is not valid. Skipping.")
                continue
        
        for contact_file in contact_maps_11:
            assignments_dict = {11: {1: 1, 2: 2, 3: 3, 4: 4, 5: 4, 6: 7, 7: 8, 8: 8, 9: 7, 10: 9, 11: 10}, 
                                12: {1: 1, 2: 2, 3: 3, 4: 6, 5: 7, 6: 5, 7: 9, 8: 9, 9: 9, 10: 10, 11: 10, 12: 10}, 
                                13: {1: 1, 2: 2, 3: 3, 4: 4, 5: 7, 6: 4, 7: 7, 8: 6, 9: 9, 10: 9, 11: 10, 12: 10}}
            
            pdb_id = contact_file.split('_')[0]
            pdb_path = os.path.join(args.pdb_dir, f"{pdb_id}.pdb")
            _, _, peptide_sequence = extract_specific_sequences(pdb_path, chain_dict)
            peptide_length = len(peptide_sequence)

            print(f"Processing contacts for tcr with > 10 epitope length: {pdb_id}")
            contacts = pd.read_csv(os.path.join(folder_path, contact_file))
            chains = chain_dict.get(pdb_id, {
                'tcra_chain': 'D',
                'tcrb_chain': 'E',
                'peptide_chain': 'C',
                'b2m_chain': 'B',
                'mhc_chain': 'A'})
            chain_dict[pdb_id] = chains

            if chains and all(chains.values()):
                try:
                    contacts_TCR_p, contacts_TCR_MHC = filter_contacts(
                        contacts,
                        chains['tcra_chain'],
                        chains['tcrb_chain'],
                        chains['peptide_chain'],
                        chains['mhc_chain'])
                    if not contacts_TCR_p.empty:
                        if 11 not in contact_models:
                            contact_models[11] = copy.deepcopy(contact_models.get(10))
                        for _, contact in contacts_TCR_p.iterrows():
                            original_position = contact['resid_to']
                            resid_to = assignments_dict[peptide_length][original_position]
                            position = f'P{resid_to}'
                            contact_models[11][position] = pd.concat([contact_models[11][position], contact.to_frame().T])
                            
                except Exception as e:
                    print(f"Error processing contact map {pdb_id}: {e}")

    if args.type == "9+all":
        for contact_file in contact_files:
            pdb_id = contact_file.split('_')[0]
            pdb_path = os.path.join(args.pdb_dir, f"{pdb_id}.pdb")
            _, _, peptide_sequence = extract_specific_sequences(pdb_path, chain_dict)
            peptide_length = len(peptide_sequence)
            if peptide_length >= 13 or peptide_length <= 8:
                print(f"Unsupported epitope length for {pdb_id} with length {peptide_length}. Skipping.")
                continue

            assignments_dict = {8: {1: 1, 2: 2, 3: 3, 4: 5, 5: 5, 6: 7, 7: 6, 8: 9}, 
                10: {1: 1, 2: 2, 3: 3, 4: 4, 5: 6, 6: 4, 7: 6, 8: 6, 9: 8, 10: 9}, 
                11: {1: 1, 2: 2, 3: 3, 4: 3, 5: 7, 6: 8, 7: 7, 8: 9, 9: 8, 10: 8, 11: 9}, 
                12: {1: 1, 2: 2, 3: 3, 4: 5, 5: 7, 6: 5, 7: 8, 8: 6, 9: 8, 10: 8, 11: 9, 12: 9}, 
                13: {1: 1, 2: 2, 3: 3, 4: 6, 5: 6, 6: 4, 7: 7, 8: 8, 9: 7, 10: 8, 11: 9, 12: 9}}
            
            print(f"Processing contacts for contact map: {pdb_id}")    

            # Read the contacts for the current PDB ID
            contacts = pd.read_csv(os.path.join(folder_path, contact_file))

            # Check if the pdb_id exists in chain_dict, if not, use the default settings
            chains = chain_dict.get(pdb_id, {
                    'tcra_chain': 'D',
                    'tcrb_chain': 'E',
                    'peptide_chain': 'C',
                    'b2m_chain': 'B',
                    'mhc_chain': 'A'})
                
            chain_dict[pdb_id] = chains

            if chains and all(chains.values()):
                try:
                    # Filter contacts for TCR-peptide and TCR-MHC
                    contacts_TCR_p, contacts_TCR_MHC = filter_contacts(
                            contacts,
                            chains['tcra_chain'],
                            chains['tcrb_chain'],
                            chains['peptide_chain'],
                            chains['mhc_chain'])

                    if not contacts_TCR_p.empty:
                        if 9 not in contact_models:
                            contact_models[9] = create_empty_contact_model(9)

                        # Distribute contacts into appropriate positions (P1, P2, ..., Pn)
                        for _, contact in contacts_TCR_p.iterrows():
                            original_position = contact['resid_to']
                            resid_to = assignments_dict[peptide_length][original_position]

                            # Ensure resid_to falls within the peptide length
                            if 1 <= resid_to <= 9:
                                position = f'P{resid_to}'
                                contact_models[9][position] = pd.concat([contact_models[9][position], contact.to_frame().T])

                except Exception as e:
                    print(f"Error processing contact map {pdb_id}: {e}")

    # Create directories and save potentials for each peptide length
    for length, model in contact_models.items():
        try:
            model_directory = os.path.join(args.output_folder, f"Model")
            os.makedirs(model_directory, exist_ok=True)
            length_directory = os.path.join(model_directory, f"TCR-p-L{length}")
            os.makedirs(length_directory, exist_ok=True)

            for position, contacts in model.items():
                if not contacts.empty:
                    print(f"Calculating TCR-peptide potential for length {length}, position {position}")

                    data_TCR_p = calculate_potential(contacts, peptide=True)
                    
                    # Generate the output file name based on the residue position
                    output_p = os.path.join(length_directory, f"tcr_p_potential_{position}.csv")
                    
                    # Save the result to a CSV file in the corresponding folder
                    data_TCR_p.to_csv(output_p, index=False)
                    print(f"TCR-peptide potential for length {length}, position {position} saved to {output_p}")
                else:
                    print(f"No valid TCR-peptide contacts for length {length}, position {position}.")
        except Exception as e:
            print(f"Error processing length {length}: {e}")
            
    print("Training complete for all contact maps.")
    
    ###### TRAIN TCR-MHC POTENTIAL ######
    
    print("Training TCR-MHC potential...")

    # Initialize DataFrame to store all contacts
    all_contacts_TCR_MHC = pd.DataFrame()

    # Iterate over all contact files
    for contact_file in contact_files:
        pdb_id = contact_file.split('_')[0] 

        print(f"Processing contacts for PDB ID: {pdb_id}")
        
        # Read the contacts for the current PDB ID
        contacts = pd.read_csv(os.path.join(folder_path, contact_file))
        chains = chain_dict.get(pdb_id, {
            'tcra_chain': 'D',
            'tcrb_chain': 'E',
            'peptide_chain': 'C',
            'b2m_chain': 'B',
            'mhc_chain': 'A'
        })
        chain_dict[pdb_id] = chains

        if chains and all(chains.values()):  # Ensure all chains are present
            try:
                # Filter contacts for the current PDB ID
                contacts_TCR_p, contacts_TCR_MHC = filter_contacts(
                    contacts,
                    chains['tcra_chain'],
                    chains['tcrb_chain'],
                    chains['peptide_chain'],
                    chains['mhc_chain'])

                # Accumulate the filtered contacts
                if not contacts_TCR_MHC.empty:
                    all_contacts_TCR_MHC = pd.concat([all_contacts_TCR_MHC, contacts_TCR_MHC], ignore_index=True)

            except Exception as e:
                print(f"Error processing contacts for PDB ID {pdb_id}: {e}")
        else:
            print(f"Missing chains for PDB ID: {pdb_id}. Skipping...")

    # Step 5: Check if we have valid data to calculate potentials
    if not all_contacts_TCR_MHC.empty:
        print("Calculating TCR-MHC potential")
        data_TCR_MHC = calculate_potential(all_contacts_TCR_MHC, peptide=False)
        output_p = os.path.join(args.output_folder, "TCR-MHC_potential.csv")
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_p), exist_ok=True)

        data_TCR_MHC.to_csv(output_p, index=False)
        print(f"TCR-MHC potential saved to {output_p}")
    else:
        print("No valid TCR-MHC contacts for the training set.")

if __name__ == "__main__":
    main()

    