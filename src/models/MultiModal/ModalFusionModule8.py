import torch
import torch.nn as nn
import torch.nn.functional as F
#门控注意力网络
class ModalFusionModule8(nn.Module):
    def __init__(self, global_dim, local_dim, num_heads=8):
        super(ModalFusionModule8, self).__init__()
        self.global_dim = global_dim
        self.local_dim = local_dim
        self.num_heads = num_heads

        # 维度调整层，将局部特征映射到全局特征的维度
        self.feature_align = nn.Linear(local_dim, global_dim)

        self.attention = nn.MultiheadAttention(self.global_dim, num_heads)
        self.gate_layer = nn.Sequential(
            nn.Linear(self.global_dim, self.global_dim),  # 生成门控信号的维度与全局特征相同
            nn.Sigmoid()
        )
        self.alpha_predictor = nn.Sequential(
            nn.Linear(self.global_dim, self.global_dim // 2),
            nn.ReLU(),
            nn.Linear(self.global_dim // 2, 1),
            nn.Sigmoid()
        )
        self.output_layer = nn.Linear(self.global_dim, self.global_dim + self.local_dim)

    def forward(self, global_features, local_features):
        B, T, _ = global_features.size()

        # 调整局部特征的维度以匹配全局特征
        local_features_aligned = self.feature_align(local_features)

        Q = global_features.permute(1, 0, 2)
        K = local_features_aligned.permute(1, 0, 2)
        V = global_features.permute(1, 0, 2)

        attn_output, _ = self.attention(Q, K, V)
        
        # 生成门控系数，并应用于调整后的局部特征
        gates = self.gate_layer(global_features)
        gated_local_features = gates * local_features_aligned

        # 结合全局特征和门控后的局部特征计算最终的注意力输出
        combined_attn_output = attn_output + gated_local_features.permute(1,0,2)

        output = self.output_layer(combined_attn_output.permute(1, 0, 2))

        return output
    
if __name__ == '__main__':
    # 使用示例
    global_input = torch.rand(2, 33, 256)  # 假设全局特征维度是 512
    local_input = torch.rand(2, 33, 512)   # 假设局部特征维度是 256
    block = ModalFusionModule8(global_dim=256, local_dim=512, num_heads=8)
    output = block(global_input, local_input)
    print(global_input.size(), local_input.size(), output.size())
