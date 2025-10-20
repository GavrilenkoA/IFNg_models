from pathlib import Path
import argparse
from utils import load_model, load_data, letter_based_encoding, evaluate_model, get_evaluation, filter_peptide_length, z_descriptors

PATH = Path(__file__).parent.parent


def evaluate_models(cv=5):
    parser = argparse.ArgumentParser()
    parser.add_argument("--descriptor", required=True)
    args = parser.parse_args()

    evaluation_list = []
    for i in range(cv):

        # Load datasets and generate descriptors
        if args.descriptor == 'LBE':
            X_test_df = load_data(PATH.joinpath(f'data/my_test/test_split_{i}.csv'))
            X_test = letter_based_encoding(X_test_df)

        elif args.descriptor == 'ZS':
            X_test_df = load_data(PATH.joinpath(f'data/my_test/test_split_{i}.csv'))
            X_test_df = filter_peptide_length(X_test_df, 15)
            X_test = z_descriptors(X_test_df)

        elif args.descriptor == 'EF':
            X_test_df = load_data(PATH.joinpath(f'data/embedding_features/test/test_split_{i}.csv'))
            X_test = X_test_df.iloc[:,:-1]

        y_test = X_test_df['labels']

        # Load trained model
        model = load_model(PATH.joinpath(f'models/trained_model_{args.descriptor}_{i}.pkl'))

        # Make predictions
        y_pred_prob = model.predict_proba(X_test)
        y_pred = (y_pred_prob[:, 1] >= 0.65).astype(int)

        # Evaluate the model
        evaluation = evaluate_model(y_test, y_pred, y_pred_prob[:, 1], cv=True)
        evaluation_list.append(evaluation)
    get_evaluation(evaluation_list)


if __name__ == "__main__":
    evaluate_models()
