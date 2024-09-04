import torch
import torch.nn as nn
import torch.nn.functional as F
#self-attention输出特征生成γ和β，来缩放全局和局部特征
class SelfAttention(nn.Module):
    def __init__(self, feature_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
    
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5)
        attention = F.softmax(attention_scores, dim=-1)
        return torch.matmul(attention, V)
    
class ConditionalFusionModule14(nn.Module):
    def __init__(self, global_dim, local_dim):
        super(ConditionalFusionModule14, self).__init__()
        self.global_dim = global_dim
        self.local_dim = local_dim

        # Dimension alignment for local features, if necessary
        self.local_to_global = nn.Linear(local_dim, global_dim) if global_dim != local_dim else None

        self.self_attention = SelfAttention(global_dim)  # Assuming attention operates on global_dim
        self.condition_network = nn.Linear(global_dim, global_dim)  # Output gamma and beta for each dim

        self.output_layer = nn.Linear(2 * global_dim, local_dim+global_dim)

    def forward(self, global_features, local_features):
        if self.local_to_global:
            local_features = self.local_to_global(local_features)

        # Concatenate global and aligned local features
        combined_features = torch.cat([global_features, local_features], dim=1)

        # Process through self-attention
        attention_output = self.self_attention(combined_features)

        # Generate gamma and beta dynamically for each time step
        gamma_beta = self.condition_network(attention_output)
        gamma, beta = gamma_beta.chunk(2, dim=-1)

        # Make sure gamma and beta match the sequence dimension
        gamma = gamma.reshape(global_features.size(0), global_features.size(1), -1)
        beta = beta.reshape(global_features.size(0), global_features.size(1), -1)

        # Adjust both global and local features
        adjusted_global = (1 + gamma) * global_features + beta
        adjusted_local = (1 + gamma) * local_features + beta

        # Concatenate adjusted features and pass through the output layer
        final_features = torch.cat([adjusted_global, adjusted_local], dim=-1)
        output = self.output_layer(final_features)

        return output
    
if __name__ == '__main__':
    # Example usage
    global_input = torch.rand(2, 33, 256)  # Example global features
    local_input = torch.rand(2, 33, 512)   # Example local features with different dimension
    block = ConditionalFusionModule14(global_dim=256, local_dim=512)
    output = block(global_input, local_input)
    print(global_input.size(), local_input.size(), output.size())
