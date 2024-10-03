# IFNg_models
In this work we build a IFNg release classification model using the peptide sequence with the MHC allele pseudo sequence. 

The requiered dependencies can be found in requirements.txt. We used python==3.9.4

```console
pip install -r requirements.txt.
```

For using the model to make predictions, the dataset should contain the peptides in the first column and the MHC allele pseudo sequence in the second column. The output file will contain the input data with a "prediction" column containing the predictions.

```console
python predict.py --input_file --output_file 
```

In the data folder, you will find the IFNg release and T-cell proliferation dataset used in this work. If you wish to replicate this study, you can find the training and test sets in the train and test folder. You can train and evaluate the model by running the train_model_cv.py and evaluate_model_cv.py. In type of descriptors need to be specified. The avergae metric is printed with the standard deviation in parentheses.

Descriptors:

LBE = Letter-based encoding 

ZS = Z-scale descriptors 

EF = Embedding features from ProtBert

```console
python train_model_cv.py --descriptors LBE
python evaluate_model_cv.py --descriptors LBE
```
