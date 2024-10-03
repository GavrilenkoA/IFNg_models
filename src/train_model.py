from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
from utils import load_data, save_model, letter_based_encoding

# Get the current script's directory
PATH = Path(__file__).parent.parent


def train_model(model_params=None):
    
    if model_params is None:
        model_params = {
            'bootstrap': False,
            'min_samples_split': 5,
            'n_estimators': 1366
        }
    # Load datasets
    X_df = load_data(PATH.joinpath(f'data/ifng_release_dataset.csv'))
        
    X = letter_based_encoding(X_df)

    y = X_df['labels']

    # Train Random Forest model
    model = RandomForestClassifier(**model_params)
    model.fit(X, y)

    # Save the model
    save_model(model, PATH.joinpath(f'models/model.pkl'))

if __name__ == "__main__":
    train_model()
