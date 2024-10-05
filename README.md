### Tabular Classification
- This repository contains deep learning models to predict the row and column from the database given a question
- train.jsonl, val.jsonl and test.jsonl contain the necessary train and val and test file

## Column Predictor
- It is a simple GRU which takes preprocessed train data and outputs the class label trained with cross entropy loss

## Row Predictor
- It works on the approach that most of the train data only has one correct row so we select the row by scoring all the rows interms of match given a query

## How to run 
- run the following command for training the model
```bash
run_model.sh <train_file> <val_file> 
'''

