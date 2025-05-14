# SeqTCR

This project aims to predict TCR-peptide binding based on squence data, using ESMC embeddings derived from TCR, epitope and MHC chains.

## Project Overview

The core of the project involves predicting how T-cell receptors (TCRs) pair to peptide-major histocompatibility complex class I (pMHC) complexes.

## Requirements

- Python 3.6 or later.

Required Python packages:

- numpy
- pandas
- xgboost
- scikit-learn
- matplotlib
- h5py
- torch
- esm  *(Facebook's Evolutionary Scale Modeling library)*
- shap
- anarci

You can install the required Python packages with the following command:

```bash
pip install -r requirements.txt
```

## Directory Structure

The project follows the following directory structure:

```bash
strucTCR/
│
├── data/                         # Data files
│
├── potentials/                   # Pre-computed interaction potentials
│
├── scripts/                      # Python scripts for processing
│   ├── embeddings                # Script to extract ESMC embedding and parse the parts important for TCR-pMHC pairing
│       ├── emb_generator.py      # Script to obtain full-sequence embeddings.
│       ├── extract_all_emb.py    # Script to parse certain parts of the embedding
│   ├── XGBoost                   # Scripts to train a classifier, predict pairing propensity and extract shap values.
│       ├── train_xgb.py          # Script to train XGBoost classifier
│       ├── predict_test.py       # Script to make predictions
│       ├── shap_values.py        # Script to extract shap values.
│
│
├── output/                       # Directory for output results
├── input/                        # Directory for input files
├── classifiers/                  # Directory with pre-trained classifiers
├── README.md                     # Project documentation
└── requirements.txt              # Python dependencies
```


## How to Run the Script

1. Clone this repository:
```bash
   git clone https://github.com/Alexasparis/biTCR
   cd biTCR/SeqTCR
```
2. Install the required Python packages:
```bash
    pip install -r requirements.txt
```
3. If you want to train a classifier:
```bash
  python3 emb_generator.py -i "$input_file" -m "$model_file" -o "$output_file" -norm -v -d "cpu/gpu"
  python3 parse_emb.py -df "$input_file" -emb "$embed_file" -o "$output_file"
  python3 train_xgb.py -df "$dataframe" -emb "$embeddings" -t "$type_emb"
```
4. If you want to use a pre-trained potential and classifier:
```bash
  python3 emb_generator.py -i "$input_file" -m "$model_file" -o "$output_file" -norm -v -d "cpu/gpu"
  python3 parse_emb.py -df "$input_file" -emb "$embed_file" -o "$output_file"
  python3 predict_test.py -emb "$embeddings" -df "$dataframe" -model "$model" -type "$type_mod" -out "$out"
```
