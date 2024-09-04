import torch
import torch.nn as nn
import torch.nn.functional as F
#传统注意力+加性注意力
class ModalFusionModule6(nn.Module):
    def __init__(self, global_dim, local_dim, num_heads=8):
        super(ModalFusionModule6, self).__init__()
        self.global_dim = global_dim
        self.local_dim = local_dim
        self.num_heads = num_heads

        if global_dim != local_dim:
            self.align_dim = nn.Linear(local_dim, global_dim)
        else:
            self.align_dim = None

        self.attention = nn.MultiheadAttention(self.global_dim, num_heads)
        self.query_layer = nn.Linear(self.global_dim, self.global_dim)
        self.key_layer = nn.Linear(self.global_dim, self.global_dim)
        self.value_layer = nn.Linear(self.global_dim, 1)  # 输出单一得分
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
        
        Q_transformed = self.query_layer(Q)  # 变换Q
        K_transformed = self.key_layer(K)    # 变换K
        additive_scores = self.value_layer(torch.tanh(Q_transformed + K_transformed))  # 加性模型
        additive_scores = additive_scores.permute(1, 0, 2)  # 改变形状为 (B, T, 1)
        additive_scores = additive_scores.expand(-1, -1, T)  # 扩展维度以匹配 V 的序列长度

        V_permuted = V.permute(1, 0, 2)  # 改变 V 的形状为 (B, T, C)
        attn_output_alt = torch.bmm(F.softmax(additive_scores, dim=2), V_permuted)  # 应用注意力权重

        alpha = self.alpha_predictor(global_features.view(B * T, -1)).view(B, T, 1)
        alpha = alpha.permute(1, 0, 2)

        combined_attn_output = alpha * attn_output + (1 - alpha) * attn_output_alt.permute(1,0,2)

        output = self.output_layer(combined_attn_output.permute(1, 0, 2))

        return output

if __name__ == '__main__':
    # Example usage
    global_input = torch.rand(2, 33, 256)
    local_input = torch.rand(2, 33, 512)
    block = ModalFusionModule6(global_dim=256, local_dim=512, num_heads=8)
    output = block(global_input, local_input)
    print(global_input.size(), local_input.size(), output.size())
