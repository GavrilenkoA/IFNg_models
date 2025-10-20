from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score, matthews_corrcoef, f1_score, roc_auc_score
import pandas as pd
import numpy as np
import joblib
import peptides

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def save_data(data, file_path):
    return data.to_csv(file_path)

def load_model(file_path):
    return joblib.load(file_path)

def save_model(model, file_path):
    joblib.dump(model, file_path)

def seq_encoding(seq):
    mapping = dict(zip('0ACDEFGHIKLMNPQRSTVWYX', range(1,23)))
    mapped_sequence = [mapping[i] for i in seq]
    return mapped_sequence

def padd_peptides(df,max_length):
    df['padd_value'] = df.apply(lambda x: max_length - len(x[0]), axis=1)
    df['padded_epitope'] = df.apply(lambda x: x[0]+'0'*x['padd_value'], axis=1)
    return df

def filter_peptide_length(df, length):
    df['peptide_length'] = df.apply(lambda x: len(x['Epitope - Name']), axis=1)
    filtered_df = df[df['peptide_length'] == length]
    return filtered_df


def letter_based_encoding(df, max_length=None, apply_padding=False):
    if apply_padding and max_length:
        df = padd_peptides(df, max_length)
        epitope_column = df.iloc[:,-1].name
    else:
        epitope_column = df.iloc[:,0].name

    epitope_list, mhc_list = list(), list()

    for epitope, mhc_epitope in zip(df[epitope_column], df.iloc[:,1]):
        epi = seq_encoding(epitope)
        mhc = seq_encoding(mhc_epitope)
        epitope_list.append(epi)
        mhc_list.append(mhc)

    epitope_df = pd.DataFrame(epitope_list).fillna(0)
    mhc_epitope_df = pd.DataFrame(mhc_list).fillna(0)

    merged_df = epitope_df.merge(mhc_epitope_df, left_index=True, right_index=True)
    merged_df.columns = merged_df.columns.astype(str)
    return merged_df


def get_z_descriptors(df, col, max_length):
    pep_desc_df = pd.DataFrame(columns=[f'z{str(x)}_0' for x in range(1,6)])
    for i in range(max_length):
        aminoacid_list = df[col].str[i].to_list()
        aa_desc_df = pd.DataFrame([peptides.Peptide(a).z_scales() for a in aminoacid_list])
        aa_desc_df.columns = [f'z{str(x)}_{i}' for x in range(1,6)]
        if i == 0:
            pep_desc_df = pd.concat([pep_desc_df, aa_desc_df])
        else:
            pep_desc_df = pep_desc_df.merge(aa_desc_df, left_index=True, right_index=True)
    return pep_desc_df


def z_descriptors(df):
    pep_df = get_z_descriptors(df, 'Epitope - Name', 15)
    mhc_df = get_z_descriptors(df, 'MHC_epitope', 34)
    return pep_df.merge(mhc_df, left_index=True, right_index=True)


def evaluate_model(y_test, y_pred, y_score, cv=None):
    ba = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    specificity = recall_score(np.logical_not(y_test) , np.logical_not(y_pred))
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_score)
    if cv is None:
        print('Balanced Accuracy: %.3f' % ba)
        print('Matthew Correlation Coefficient: %.3f' % mcc)
        print('Precision: %.3f' % precision)
        print('Recall: %.3f' % recall)
        print('Specificity: %.3f' % specificity)
        print('AUC Score: %.3f' % auc_score)
    else:
        return [ba, mcc, precision, recall, specificity, f1, auc_score]


def get_evaluation(evaluation_list):
    mean_evaluation = np.mean(evaluation_list, axis=0)
    std_evaluation = np.std(evaluation_list, axis=0)
    print('Balanced Accuracy: %.3f (%.3f)' % (mean_evaluation[0], std_evaluation[0]))
    print('Matthew Correlation Coefficient: %.3f (%.3f)' % (mean_evaluation[1], std_evaluation[1]))
    print('Precision: %.3f (%.3f)' % (mean_evaluation[2], std_evaluation[2]))
    print('Recall: %.3f (%.3f)' % (mean_evaluation[3], std_evaluation[3]))
    print('Specificity: %.3f (%.3f)' % (mean_evaluation[4], std_evaluation[4]))
    print('F1-score: %.3f (%.3f)' % (mean_evaluation[5], std_evaluation[5]))
    print('AUC Score: %.3f (%.3f)' % (mean_evaluation[6], std_evaluation[6]))
