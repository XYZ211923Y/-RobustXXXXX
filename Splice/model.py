import torch.nn as nn
import torch
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        dropout_rate = 0.5
        input_size = 30
        self.hidden_size = 30
        n_labels = 3
        self.num_layers = 4
        self.lstm = nn.LSTM(input_size, self.hidden_size, num_layers=self.num_layers, batch_first=False)
        self.fc = nn.Linear(self.hidden_size, n_labels)
        self.attention = SelfAttention(self.hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=1)
        self.embed = torch.nn.Embedding(5, 30)
        self.relu = nn.ReLU()

    # overload forward() method
    def forward(self, x, weight_of_x):
        x = x.long()
        x = self.embed(x)
        weight_of_x = torch.unsqueeze(weight_of_x, dim=3)
        x = x * weight_of_x
        x = self.relu(x)

        x = torch.mean(x, dim=2)

        # used for training

        # h0 = torch.randn((self.num_layers, x.size()[1], self.hidden_size)).cuda()
        # c0 = torch.randn((self.num_layers, x.size()[1], self.hidden_size)).cuda()
        # output, h_n = self.lstm(x, (h0, c0))

        output, h_n = self.lstm(x)

        x, attn_weights = self.attention(output.transpose(0, 1))

        x = self.dropout(x)

        logit = self.fc(x)

        logit = self.softmax(logit)

        return logit


class RNN_0(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        dropout_rate = 0.5
        input_size = 5
        hidden_size = 5
        n_labels = 3
        self.num_layers = 4
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=self.num_layers, batch_first=False)
        self.fc = nn.Linear(hidden_size, n_labels)
        self.attention = SelfAttention(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    # overload forward() method
    def forward(self, x, weight_of_x):
        x = x * weight_of_x
        x = self.relu(x)

        # used for training

        # h0 = torch.randn((self.num_layers, x.size()[1], x.size()[2])).cuda()
        # c0 = torch.randn((self.num_layers, x.size()[1], x.size()[2])).cuda()
        # output, h_n = self.lstm(x, (h0, c0))

        output, h_n = self.lstm(x)

        x, attn_weights = self.attention(output.transpose(0, 1))

        x = self.dropout(x)

        logit = self.fc(x)

        logit = self.softmax(logit)
        return logit


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
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