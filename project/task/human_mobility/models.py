import torch
from torch import nn
from project.types.common import NetGen
from project.utils.utils import lazy_config_wrapper


class SimpleRNN(nn.Module):
    def __init__(self, input_size=7, hidden_size=128, num_layers=3, output_size=8):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)
        # Apply dropout to the output of the RNN
        out = self.dropout(out)
        # Decode the hidden state of the last time step
        return self.fc(out[:, -1, :])


class LSTM(nn.Module):
    def __init__(self, input_size=7, hidden_size=128, num_layers=3, output_size=8):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Replace RNN with LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # LSTM outputs (out, (hn, cn))
        # Apply dropout to the output of the LSTM
        out = self.dropout(out)
        # Decode the hidden state of the last time step
        return self.fc(out[:, -1, :])


# Simple wrapper to match the NetGenerator Interface
get_net: NetGen = lazy_config_wrapper(SimpleRNN)
