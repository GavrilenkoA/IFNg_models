from pathlib import Path
import argparse
from sklearn.metrics import classification_report
from utils import load_model, load_data, letter_based_encoding, evaluate_model, save_data

PATH = Path(__file__).parent.parent

def make_predictions():

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    #Load data
    data = load_data(args.input_file)
        
    X = letter_based_encoding(data)

    # Load model
    model = load_model(PATH.joinpath(f'models/model.pkl'))

    # Predict
    y_pred_prob = model.predict_proba(X)
    y_pred = (y_pred_prob[:,1] >= 0.65).astype(int)
    
    # save predictions
    df_result = data.assign(predictions=y_pred)
    save_data(df_result, args.output_file)

if __name__ == "__main__":
    make_predictions()