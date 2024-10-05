from torch.utils.data import Dataset
from copy import deepcopy
import torch
from copy import deepcopy
from preprocesstext import *
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    # Separate the tensors for question, column, and label
    questions, columns, labels = zip(*batch)
    
    # Pad sequences for questions and columns
    padded_questions = pad_sequence(questions, batch_first=True, padding_value=0)
    padded_columns = pad_sequence(columns, batch_first=True, padding_value=0)
    
    # Return padded questions, columns, and labels
    return padded_questions, padded_columns, torch.stack(labels)

class TrainDataset(Dataset):
    def __init__(self, data, glove):
        self.data = data 
        self.glove = glove

    
    def __len__(self):
        return len(self.data)
    
    
    def get_tokens(self, text):
        self.text = text
        for i in range(len(text)):
            if text[i] in self.glove:
                text[i] = torch.tensor(self.glove[text[i]])
            else:
                text[i] = torch.tensor(self.glove['unk'])

        return text
    
    def __getitem__(self, idx):
        question = self.data[idx]['question']
        column = self.data[idx]['column']
        question_new = deepcopy(question)
        column_new = deepcopy(column)
        label = torch.tensor(self.data[idx]['col_label'])
        question_new = torch.stack(self.get_tokens(question_new))
        column_new_1 = torch.stack(self.get_tokens(column_new))
        

        
        return question_new, column_new_1, label
    

    

class TrainRowDataset(Dataset):
    def __init__(self, data, glove, model):
        self.data = data 
        self.glove = glove
        self.model = model

    
    def __len__(self):
        return len(self.data)
    
    
    def get_tokens(self, text):
        self.text = text
        for i in range(len(text)):
            if text[i] in self.glove:
                
                text[i] = torch.tensor(self.glove[text[i]])
            else:
                text[i] = torch.tensor(self.glove['unk'])

        return text
    
    def get_tokens_new(self, text):
        self.text = text
        for i in range(len(text)):
            if text[i] in self.model.wv:
                
                text[i] = torch.tensor(self.model.wv[text[i]])
            else:
                print(text[i])
            #     text[i] = torch.tensor(self.glove['unk'])

        return text
    
    def __getitem__(self, idx):
        question = self.data[idx]['question']
        column = self.data[idx]['row']
        question_new = deepcopy(question)
        row_new = deepcopy(column)
        label = torch.tensor(self.data[idx]['row_label'])
        question_new = torch.stack(self.get_tokens(question_new))
        row_new = torch.stack(self.get_tokens_new(row_new))
        

        
        return question_new, row_new, label
    


class ValRowsDataset(Dataset):
    def __init__(self, data, glove):
        self.data = data 
        self.glove = glove

    
    def __len__(self):
        return len(self.data)
    
    
    def get_tokens(self, text):
        self.text = text
        for i in range(len(text)):
            if text[i] in self.glove:
                
                text[i] = torch.tensor(self.glove[text[i]])
            else:
                text[i] = torch.tensor(self.glove['unk'])

        return text
    
    def get_tokens_new(self, text):
        # Initialize an empty list to store processed tokens
        processed_text = []
        for token in text:
            if token in self.glove:
                processed_text.append(torch.tensor(self.glove[token]))
            else:
                processed_text.append(torch.tensor(self.glove['unk']))
        # Return the list of processed tokens
        return torch.stack(processed_text)
    
    def __getitem__(self, idx):
        question = self.data[idx]['question']
        rows = self.data[idx]['rows']
        question_new = deepcopy(question)
        rows_new = deepcopy(rows)
        label = torch.tensor(self.data[idx]['row_label'])
        question_new = torch.stack(self.get_tokens(question_new))
        for i in range(len(rows_new)):
            rows_new[i] = self.get_tokens_new(rows_new[i])
        
        
        return question_new, rows_new, label
    


class ValDataset(Dataset):
    def __init__(self, data, glove):
        self.data = data 
        self.glove = glove

    
    def __len__(self):
        return len(self.data)
    
    
    def get_tokens(self, text):
        self.text = text
        for i in range(len(text)):
            if text[i] in self.glove:
                
                text[i] = torch.tensor(self.glove[text[i]])
            else:
                text[i] = torch.tensor(self.glove['unk'])

        return text
    
    def get_tokens_new(self, text):
        # Initialize an empty list to store processed tokens
        processed_text = []
        for token in text:
            if token in self.glove:
                processed_text.append(torch.tensor(self.glove[token]))
            else:
                processed_text.append(torch.tensor(self.glove['unk']))
        # Return the list of processed tokens
        return torch.stack(processed_text)
    
    def __getitem__(self, idx):
        question = self.data[idx]['question']
        column = self.data[idx]['columns']
        question_new = deepcopy(question)
        column_new = deepcopy(column)
        label = torch.tensor(self.data[idx]['col_label'])
        question_new = torch.stack(self.get_tokens(question_new))
        for i in range(len(column_new)):
            column_new[i] = self.get_tokens_new(column_new[i])
        

        
        return question_new, column_new, label





    






