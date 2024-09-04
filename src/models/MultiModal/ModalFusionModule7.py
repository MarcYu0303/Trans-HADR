import torch
import torch.nn as nn
import torch.nn.functional as F

class ModalFusionModule7(nn.Module):
    def __init__(self, global_dim, local_dim, num_heads=8):
        super(ModalFusionModule7, self).__init__()
        self.global_dim = global_dim
        self.local_dim = local_dim
        self.num_heads = num_heads

        if global_dim != local_dim:
            self.align_dim = nn.Linear(local_dim, global_dim)
        else:
            self.align_dim = None

        self.attention = nn.MultiheadAttention(self.global_dim, num_heads)
        self.alpha_predictor = nn.Sequential(
            nn.Linear(self.global_dim, self.global_dim // 2),
            nn.ReLU(),
            nn.Linear(self.global_dim // 2, 1),
            nn.Sigmoid()
        )
        self.output_layer = nn.Linear(self.global_dim, self.global_dim + self.local_dim)

    def forward(self, global_features, local_features):
        B, T, _ = global_features.size()

        if self.align_dim:
            local_features = self.align_dim(local_features)

        Q = global_features.permute(1, 0, 2)  # (T, B, C)
        K = local_features.permute(1, 0, 2)
        V = global_features.permute(1, 0, 2)

        attn_output, _ = self.attention(Q, K, V)
        
        # Scaled Dot-Product Attention
        d_k = K.size(-1)  # Dimension of keys
        scores = torch.bmm(Q, K.transpose(1, 2)) / (d_k ** 0.5)
        attn_output_alt = torch.bmm(F.softmax(scores, dim=-1), V)

        alpha = self.alpha_predictor(global_features.view(B * T, -1)).view(B, T, 1)
        alpha = alpha.permute(1, 0, 2)

        combined_attn_output = alpha * attn_output + (1 - alpha) * attn_output_alt

        output = self.output_layer(combined_attn_output.permute(1, 0, 2))

        return output
    
if __name__ == '__main__':
    # Example usage
    global_input = torch.rand(2, 33, 256)
    local_input = torch.rand(2, 33, 512)
    block = ModalFusionModule7(global_dim=256, local_dim=512, num_heads=8)
    output = block(global_input, local_input)
    print(global_input.size(), local_input.size(), output.size())
