import torch
import torch.nn as nn


    
class QuestionColumnEncoder(nn.Module):
    def __init__(self,embed_size, hidden_size, output_size):
        super(QuestionColumnEncoder, self).__init__()
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True, num_layers = 2)
        self.layer_norm = nn.LayerNorm(hidden_size) 
        self.fc1 = nn.Linear(output_size,output_size//2)
        self.fc2 = nn.Linear(output_size//2,output_size //4)
        self.layer_norm1 = nn.LayerNorm(output_size //2)
        self.layer_norm2 = nn.LayerNorm(output_size//4)



    def forward(self, x):
        output, hn = self.gru(x)
        hidden = self.layer_norm(hn[-1])
        output_hidden = torch.tanh(hidden)
        output_1 = torch.tanh(self.layer_norm1(self.fc1(output_hidden)))
        output_2 = torch.tanh(self.layer_norm2(self.fc2(output_1)))
        return output_2
    
class ColumnEncoder(nn.Module):
    def __init__(self,embed_size, hidden_size, output_size):
        super(ColumnEncoder, self).__init__()
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True, num_layers = 2)
        self.layer_norm = nn.LayerNorm(hidden_size) 
        self.fc1 = nn.Linear(output_size,output_size//2)
        self.fc2 = nn.Linear(output_size//2,output_size //4)
        self.layer_norm1 = nn.LayerNorm(output_size //2)
        self.layer_norm2 = nn.LayerNorm(output_size//4)



    def forward(self, x):
        output, hn = self.gru(x)
        hidden = self.layer_norm(hn[-1])
        output_hidden = torch.tanh(hidden)
        output_1 = torch.tanh(self.layer_norm1(self.fc1(output_hidden)))
        output_2 = torch.tanh(self.layer_norm2(self.fc2(output_1)))
        return output_2


    

class ColumnPredictor(nn.Module):
    def  __init__(self,question_encoder, column_encoder,output_size, device):
        super(ColumnPredictor, self).__init__()
        self.question_encoder = question_encoder
        self.column_encoder = column_encoder
        self.fc = nn.Linear(output_size//2, output_size//4) 
        self.fc1 = nn.Linear(output_size//4, 1) 
        self.layer_norm = nn.LayerNorm(output_size //4) 
        
        self.sigmoid = nn.Sigmoid()
        self.device = device
    
    def forward(self, question, column):
        question = question.to(self.device)
        column = column.to(self.device)
        out1 = self.question_encoder(question)
        out2 = self.column_encoder(column)
        mul = torch.cat((out1, out2), dim=1)
        mul1 = torch.tanh(self.layer_norm(self.fc(mul)))
        mul1 = self.fc1(mul1)
        output =  self.sigmoid(mul1)
        return output
    
