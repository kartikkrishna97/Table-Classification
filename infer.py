import torch
import numpy as np
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from preprocesstext import *
from dataloader import *
from model import *
from torch.utils.data import  DataLoader
import torch.optim as optim
from tqdm import tqdm
import gensim.downloader as api
import sys

def find_elements(lst, index):
    if index == 0:
        left = []
        right = lst[index + 1:index + 5]
    else:
        left = lst[index - 2:index]
        right = lst[index + 1:index + 3]
    if index == len(lst)-1:
        right = lst[index-4:index]
        left = []
    

    return left+right

def tokenize_row(list):
    return  [word for string in list for word in re.findall(r'\b\w+\b', string)]


def list_interect(list1, list2):
    intersection_percentage = len(set(list1) & set(list2)) / max(len(list1), len(list2)) * 100
    return intersection_percentage
        


data = preprocess(sys.argv[2])


glove = api.load("glove-wiki-gigaword-300")
print('Glove Model downloaded')

Val_Column_Dataset = ValDataset(data,glove)
question, col, label = Val_Column_Dataset[0]

val_dataloader = DataLoader(Val_Column_Dataset, batch_size=1, num_workers=1)

EMBED_SIZE = 300
HIDDEN_SIZE = 512
OUTPUT_SIZE = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder1 = QuestionColumnEncoder(EMBED_SIZE,HIDDEN_SIZE,OUTPUT_SIZE).to(device)
encoder2 = ColumnEncoder(EMBED_SIZE,HIDDEN_SIZE,OUTPUT_SIZE).to(device)
model = ColumnPredictor(encoder1,encoder2,output_size=OUTPUT_SIZE, device=device)
model_path = 'test.pth'





model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()
preds = []
print("STARTING VALIDATION ................")
with torch.no_grad():
    for batch in tqdm(val_dataloader):
        preds_batch = []
        question, cols, col_label = batch
        question = question.to(device)
        for i in range(len(cols)):
            cols[i] = cols[i].to(device)
            logits = model(question,cols[i])
            preds_batch.append(logits)
        
        

        preds.append(preds_batch.index(max(preds_batch)))



count = 0
row_preds = []
for i in range(len(preds)):
    preds_batch = []
    if preds[i]==data[i]['col_label']:
        count+=1

print(f'val column accuracy is: {count/len(preds)}')

row_predicted = []
rows_gold = []

for i in tqdm(range(len(data))):
    question = data[i]['question']
    predicted = []
    for row in data[i]['rows']:
        rows = find_elements(row, preds[i])
        row_tokens = tokenize_row(rows)
        predicted.append(list_interect(question, row_tokens))
    row_predicted.append(predicted.index(max(predicted)))
    rows_gold.append(data[i]['row_labels'])


count = 0
for i in range(len(data)):
    if len(rows_gold[i])>1:
        continue
    elif row_predicted[i]==rows_gold[i][0]:
        count+=1

print(f'val row accuracy is {count/len(data)}')






