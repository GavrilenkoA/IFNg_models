from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse
from pathlib import Path
from utils import load_data, save_model, letter_based_encoding, z_descriptors, filter_peptide_length

# Get the current script's directory
PATH = Path(__file__).parent.parent


def train_model(cv=10, model_params=None):
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptor", required=True)
    args = parser.parse_args()

    if model_params is None:
        model_params = {
            'bootstrap': False,
            'min_samples_split': 5,
            'n_estimators': 1366
        }
    for i in range(cv):

        # Load datasets
        X_train_df = load_data(PATH.joinpath(f'data/train/train_split_{i}.csv'))
        
        # Load datasets and generate descriptors
        if args.descriptor == 'LBE':
            X_train_df = load_data(PATH.joinpath(f'data/train/train_split_{i}.csv')) 
            X_train = letter_based_encoding(X_train_df)


        elif args.descriptor == 'ZS':
            X_train_df = load_data(PATH.joinpath(f'data/train/train_split_{i}.csv'))
            X_train_df = filter_peptide_length(X_train_df, 15)
            X_train = z_descriptors(X_train_df)


        elif args.descriptor == 'EF':
            X_train_df = load_data(PATH.joinpath(f'data/embedding_features/train/train_split_{i}.csv'))
            X_train = X_train_df.iloc[:,:-1]
        
        y_train = X_train_df['labels']

        # Train Random Forest model
        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)

        # Save the model
        save_model(model, PATH.joinpath(f'models/trained_model_{args.descriptor}_{i}.pkl'))

if __name__ == "__main__":
    train_model()