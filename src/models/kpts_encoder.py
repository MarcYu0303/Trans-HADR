import torch
import torch.nn as nn

class KeypointsEncoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=32):
        super(KeypointsEncoder, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x
    
if __name__ == '__main__':
    encoder = KeypointsEncoder(input_dim=21*2, hidden_dim=128, output_dim=64)
    x = torch.randn(10, 21*2)
    out = encoder(x)
    print(out.shape)