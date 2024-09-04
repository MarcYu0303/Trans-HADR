import torch
import torch.nn as nn
import torch.nn.functional as F
#两个可学习的query分别聚合全局和局部特征，然后一个可学习的query聚合前面输出的特征
class CrossAttentionWithLearnableQuery(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super(CrossAttentionWithLearnableQuery, self).__init__()
        self.query = nn.Parameter(torch.randn(1, 1, feature_dim))  # Learnable query
        self.attention = nn.MultiheadAttention(feature_dim, num_heads)

    def forward(self, key_value):
        # key_value shape: (T, B, C)
        T, B, _ = key_value.size()
        query = self.query.expand(T, B, -1)  # Expanding query to match T and B
        output, _ = self.attention(query, key_value, key_value)
        return output

class QueryEnhancedSemanticModule(nn.Module):
    def __init__(self, global_dim, local_dim, num_heads=8):
        super(QueryEnhancedSemanticModule, self).__init__()
        self.global_dim = global_dim
        self.local_dim = local_dim

        # Cross-Attention with learnable query for both global and local features
        self.global_feature_attention = CrossAttentionWithLearnableQuery(self.global_dim, num_heads)
        self.local_feature_attention = CrossAttentionWithLearnableQuery(self.local_dim, num_heads)

        # Cross-Attention for concatenated features
        self.concat_feature_attention = CrossAttentionWithLearnableQuery(self.global_dim + self.local_dim, num_heads)  # Adjusted dimension

    def forward(self, global_features, local_features):
        # Reshape to (T, B, C) for compatibility with MultiheadAttention
        global_features = global_features.permute(1, 0, 2)
        local_features = local_features.permute(1, 0, 2)

        # Generate aggregated global and local features
        agg_global_features = self.global_feature_attention(global_features)
        agg_local_features = self.local_feature_attention(local_features)

        # Concatenate the aggregated features
        concatenated_features = torch.cat((agg_global_features, agg_local_features), dim=2)

        # Aggregate information from concatenated features
        aggregated_output = self.concat_feature_attention(concatenated_features)

        # Reshape output back to (B, T, C)
        output = aggregated_output.permute(1, 0, 2)
        return output
    
if __name__ == '__main__':
    # Example of creating and using the module
    global_input = torch.rand(2, 33, 256).cuda()  # Example global features
    local_input = torch.rand(2, 33, 256).cuda()   # Example local features
    block = QueryEnhancedSemanticModule(global_dim=256, local_dim=256, num_heads=8).cuda()
    output = block(global_input, local_input)
    print(global_input.size(), local_input.size(), output.size())
