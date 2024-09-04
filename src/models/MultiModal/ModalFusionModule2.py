import torch
import torch.nn as nn
import torch.nn.functional as F
#传统注意力+加性注意力
class AdditiveAttention(nn.Module):
    def __init__(self, key_dim, query_dim, hidden_dim):
        super(AdditiveAttention, self).__init__()
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.v = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, key, query):
        # key, query: (T, B, C)
        key_proj = self.key_proj(key)  # (T, B, hidden_dim)
        query_proj = self.query_proj(query)  # (T, B, hidden_dim)
        energy = torch.tanh(key_proj + query_proj.unsqueeze(1))  # Broadcasting (T, T, B, hidden_dim)
        energy = torch.sum(self.v * energy, dim=-1)  # Reduce along hidden dimension (T, T, B)
        attn_weights = F.softmax(energy, dim=1)  # Softmax over key dimension
        return attn_weights.permute(2, 0, 1)  # Ensure shape is (B, T, T)


class ModalFusionModule2(nn.Module):
    def __init__(self, global_dim, local_dim, num_heads=8, hidden_dim=128):
        super(ModalFusionModule2, self).__init__()
        self.global_dim = global_dim
        self.local_dim = local_dim
        self.num_heads = num_heads

        if global_dim != local_dim:
            self.align_dim = nn.Linear(local_dim, global_dim)
        else:
            self.align_dim = None

        self.attention = nn.MultiheadAttention(global_dim, num_heads)
        self.additive_attention = AdditiveAttention(global_dim, global_dim, hidden_dim)
        self.alpha_predictor = nn.Sequential(
            nn.Linear(global_dim, global_dim // 2),
            nn.ReLU(),
            nn.Linear(global_dim // 2, 1),
            nn.Sigmoid()
        )
        self.output_layer = nn.Linear(global_dim, global_dim + local_dim)
        
    def forward(self, global_features, local_features):
        B, T, _ = global_features.size()

        if self.align_dim:
            local_features = self.align_dim(local_features)

        Q = global_features.permute(1, 0, 2)  # (T, B, C)
        K = local_features.permute(1, 0, 2)
        V = global_features.permute(1, 0, 2)  # (T, B, C)

        attn_output, _ = self.attention(Q, K, V)

        attn_weights = self.additive_attention(K, Q)  # (B, T, T)
        V_permuted = V.permute(1, 0, 2)  # (B, T, C) for bmm
        attn_output_alt = torch.bmm(attn_weights, V_permuted)  # Should be (B, T, C)

        attn_output_alt = attn_output_alt.permute(1, 0, 2)  # Permute to (T, B, C) to match attn_output

        alpha = self.alpha_predictor(global_features.view(B * T, -1)).view(B, T, 1)
        alpha = alpha.permute(1, 0, 2).expand(-1, -1, attn_output.size(2))  # Reshape and expand alpha to (T, B, C)

        combined_attn_output = alpha * attn_output + (1 - alpha) * attn_output_alt

        output = self.output_layer(combined_attn_output).permute(1,0,2)
        return output


if __name__ == '__main__':
    # Example usage
    global_input = torch.rand(2, 33, 256)
    local_input = torch.rand(2, 33, 256)
    block = ModalFusionModule2(global_dim=256, local_dim=256, num_heads=8)
    output = block(global_input, local_input)
    print(global_input.size(), local_input.size(), output.size())
