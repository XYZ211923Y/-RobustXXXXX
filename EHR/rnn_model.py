
import rootpath
rootpath.append()
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention,self).__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        # (B, L, H) -> (B , L, 1)
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights


class LSTM(nn.Module):
    
    def __init__(self, options, emb_weights):
        super(LSTM,self).__init__()
        self.options = options
        
        n_diagnosis_codes = options['n_diagnosis_codes']
        visit_size = options['visit_size']
        hidden_size = options['hidden_size']

        n_labels = options['n_labels']
        dropout_rate = options['dropout_rate']
        
        # # code embedding layer
        # self.embed = nn.Linear(n_diagnosis_codes, visit_size)


        # modified embedding layer
        # emb_size = 300
        self.embed = torch.nn.Embedding(n_diagnosis_codes, visit_size)
        self.embed.weight = nn.Parameter(torch.FloatTensor(emb_weights))
        
        # relu layer
        self.relu = nn.ReLU()
        
        # gru layer
        self.lstm = nn.LSTM(visit_size, hidden_size, num_layers = 1, batch_first = False)

        self.attention = SelfAttention(hidden_size)
        
        # dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # fully connected layer
        self.fc = nn.Linear(hidden_size, n_labels)
        
        # softmax
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, weight_of_x):

        # x = torch.LongTensor(x)
        x = self.embed(x) # (n_visits, n_samples, visit_size)
        # multiply by weight here:
        weight_of_x = torch.unsqueeze(weight_of_x, dim=3)
        x = x * weight_of_x
        x = self.relu(x)

        x = torch.mean(x, dim=2)

        h0 = torch.FloatTensor(torch.zeros(1, x.size()[1], x.size()[2])).cuda()
        c0 = torch.FloatTensor(torch.zeros(1, x.size()[1], x.size()[2])).cuda()
        output, h_n = self.lstm(x, (h0, c0)) # output (seq_len, batch, hidden_size)
                                         #h_n (num_layers, batch, hidden_size)

        embedding, attn_weights = self.attention(output.transpose(0, 1))

        # x = self.dropout(embedding) # (n_samples, hidden_size)
        x = embedding
        logit = self.fc(x) # (n_samples, n_labels)

        logit = self.softmax(logit)

        return logit