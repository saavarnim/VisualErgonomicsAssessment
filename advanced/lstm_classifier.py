import torch
import torch.nn as nn

class PostureLSTM(nn.Module):
    """
    Optional secondary architecture for fatigue classification using timeseries 
    sequence of posture angles.
    """
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, num_classes=3):
        super(PostureLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # Classes: 0: Active, 1: Stagnant, 2: Fatigued

    def forward(self, x):
        # x.shape = (batch_size, sequence_length, input_size)
        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        # out[:, -1, :] takes the output of the last sequence step
        out = self.fc(out[:, -1, :])
        return out

# Placeholder for how it would be used:
# model = PostureLSTM()
# # Load weights if available
# # model.load_state_dict(torch.load("lstm_weights.pth"))
# # model.eval()
