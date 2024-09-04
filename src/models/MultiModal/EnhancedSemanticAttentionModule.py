import torch
import torch.nn as nn
import torch.nn.functional as F
#全局特征和局部特征分别做cross-attention，拼接后使用Self-attention进行特征强化（如果local和global维度不匹配，则相互线性映射到匹配的维度）
import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedSemanticAttentionModule(nn.Module):
    def __init__(self, global_dim, local_dim, num_heads=8):
        super(EnhancedSemanticAttentionModule, self).__init__()
        self.global_dim = global_dim
        self.local_dim = local_dim
        self.num_heads = num_heads

        # 使用线性层匹配维度
        self.adjust_global_dim = nn.Linear(global_dim, local_dim)
        self.adjust_local_dim = nn.Linear(local_dim, global_dim)

        # Cross-Attention layers
        self.global_to_local_attention = nn.MultiheadAttention(local_dim, num_heads)
        self.local_to_global_attention = nn.MultiheadAttention(global_dim, num_heads)

        # Self-Attention layer for the concatenated features
        self.self_attention = nn.MultiheadAttention(global_dim + local_dim, num_heads)

        # Optional: Layer normalization
        self.layer_norm = nn.LayerNorm(global_dim + local_dim)

    def forward(self, global_features, local_features):
        global_features = global_features.permute(1, 0, 2)  # Reshape to (T, B, C)
        local_features = local_features.permute(1, 0, 2)

        # 调整全局和局部特征的维度
        adjusted_global_features = self.adjust_global_dim(global_features)
        adjusted_local_features = self.adjust_local_dim(local_features)

        # Cross-attention operations
        global_to_local_attn, _ = self.global_to_local_attention(local_features, adjusted_global_features, adjusted_global_features)
        local_to_global_attn, _ = self.local_to_global_attention(global_features, adjusted_local_features, adjusted_local_features)

        # Concatenate the cross-attention outputs
        concatenated_features = torch.cat((global_to_local_attn, local_to_global_attn), dim=2)

        # Self-attention to enhance the features further
        enhanced_features, _ = self.self_attention(concatenated_features, concatenated_features, concatenated_features)

        # Optional: Layer normalization
        enhanced_features = self.layer_norm(enhanced_features)

        # Reshape output back to (B, T, C)
        output = enhanced_features.permute(1, 0, 2)
        return output

if __name__ == '__main__':
    # Example usage
    global_input = torch.rand(2, 33, 256).cuda()  # Example global features
    local_input = torch.rand(2, 33, 512).cuda()   # Example local features
    block = EnhancedSemanticAttentionModule(global_dim=256, local_dim=512, num_heads=8).cuda()
    output = block(global_input, local_input)
    print(global_input.size(), local_input.size(), output.size())
