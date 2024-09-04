import torch
import torch.nn as nn
import torch.nn.functional as F
#传统注意力+高斯核注意力
def gaussian_kernel_attention(Q, K, sigma=1.0):
    # Q, K: (T, B, C)
    T, B, C = Q.size()
    
    # Expanding Q and K for broadcasting
    Q_expanded = Q.unsqueeze(2).expand(-1, -1, T, -1)  # Expands Q to (T, B, T, C)
    K_expanded = K.unsqueeze(1).expand(-1, T, -1, -1)  # Expands K to (T, T, B, C), needs rearranging

    # Correcting the alignment issue:
    K_expanded = K_expanded.permute(0, 2, 1, 3)  # Now K_expanded is (T, B, T, C), aligned with Q_expanded

    # Calculating the squared Euclidean distances
    diff = Q_expanded - K_expanded
    distance = (diff ** 2).sum(dim=-1)  # Summing over the feature dimension C

    # Applying the Gaussian kernel
    gaussian_attention = torch.exp(-distance / (2 * sigma ** 2))
    return gaussian_attention.permute(1, 0, 2)  # Permuting to (B, T, T) for proper alignment in subsequent operations

class ModalFusionModule3(nn.Module):
    def __init__(self, global_dim, local_dim, num_heads=8):
        super(ModalFusionModule3, self).__init__()
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
        gaussian_attention = gaussian_kernel_attention(Q, K)  # Should be (B, T, T)
        
        attn_output_alt = torch.bmm(gaussian_attention, V.permute(1, 0, 2)).permute(1,0,2)  # Apply Gaussian attention

        alpha = self.alpha_predictor(global_features.view(B * T, -1)).view(B, T, 1).permute(1, 0, 2)
        alpha = alpha.expand(-1, -1, V.size(2))  # Ensure alpha covers all channels

        combined_attn_output = alpha * attn_output + (1 - alpha) * attn_output_alt

        output = self.output_layer(combined_attn_output)
        return output


if __name__ == '__main__':
    # Example usage
    global_input = torch.rand(2, 33, 256)
    local_input = torch.rand(2, 33, 256)
    block = ModalFusionModule3(global_dim=256, local_dim=256, num_heads=8)
    output = block(global_input, local_input)
    print(global_input.size(), local_input.size(), output.size())
