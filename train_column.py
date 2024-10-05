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


data = preprocess(sys.argv[1])

data_column = column_data(data)



glove = api.load("glove-wiki-gigaword-300")
print('Glove Model downloaded')
    

train_column_dataset = TrainDataset(data_column, glove)



train_dataloader = DataLoader(train_column_dataset, batch_size=64, num_workers=1, collate_fn=collate_fn)





num_epochs = 55


EMBED_SIZE = 300
HIDDEN_SIZE = 512
OUTPUT_SIZE = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder1 = QuestionColumnEncoder( EMBED_SIZE,HIDDEN_SIZE,OUTPUT_SIZE).to(device)
encoder2 = ColumnEncoder(EMBED_SIZE,HIDDEN_SIZE,OUTPUT_SIZE).to(device)
model = ColumnPredictor(encoder1, encoder2, OUTPUT_SIZE, device=device)
learning_rate = 1e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCELoss()
print(model)
file_path ='test.pth'
# saved_path = 'column_predictor_new_1.pth'
# model.load_state_dict(torch.load(saved_path))
for epoch in tqdm(range(num_epochs)):
    #TODO TRAIN THE BIDIRECTIONAL GRU COLUMN PREDICTOR
    model.train()
    train_loss = []
    tqdm_batch = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
    for batch in tqdm_batch:
        question, column, label = batch
        label = label.float()
        label = label.view(label.shape[0],1)
        question.to(device), column.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(question, column)
        loss = criterion(output, label)  
        loss.backward()
        optimizer.step()
        # print(loss)
        train_loss.append(loss.item())
        tqdm_batch.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), file_path)
    tqdm_batch.close()
    
    print(f'training loss is for {epoch} epoch is {sum(train_loss)/len(train_loss)}')






