# IFNg_models
In this work we build a IFNg release classification model using the peptide sequence with the MHC allele pseudo sequence. 

The requiered dependencies can be found in requirements.txt. We used python==3.9.4

```console
pip install -r requirements.txt.
```

For using the model to make predictions, the dataset should contain the peptides in the first column and the MHC allele pseudo sequence in the second column. The output file will include the original input data along with an additional column for the predictions.

```console
python predict.py --input_file --output_file 
```

In the 'data' folder, you will find the IFNg release and T-cell proliferation datasets used in this study. To replicate the study, the training and test sets are available in the 'train' and 'test' folders. You can train and evaluate the model by running 'train_model_cv.py' and 'evaluate_model_cv.py'. The type of descriptors must be specified. The average metric will be printed, with the standard deviation shown in parentheses.

Descriptors:

LBE = Letter-based encoding 

ZS = Z-scale descriptors 

EF = Embedding features from ProtBert

```console
python train_model_cv.py --descriptors LBE
python evaluate_model_cv.py --descriptors LBE
```
