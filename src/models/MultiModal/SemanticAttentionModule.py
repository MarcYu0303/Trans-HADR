import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticAttentionModule(nn.Module):
    def __init__(self, global_dim, local_dim, num_heads=8):
        super(SemanticAttentionModule, self).__init__()
        self.global_dim = global_dim
        self.local_dim = local_dim
        self.num_heads = num_heads
        
        # Self-Attention for global features
        self.global_attention = nn.MultiheadAttention(global_dim, num_heads)
        
        # Cross-Attention, global features attending to local features
        self.cross_attention = nn.MultiheadAttention(global_dim, num_heads)
        
        # Adjust local feature dimensions to match global dimensions if necessary
        if global_dim != local_dim:
            self.adjust_local_dim = nn.Linear(local_dim, global_dim)
        else:
            self.adjust_local_dim = None
        
        # Final linear layer to combine features
        self.final_linear = nn.Linear(global_dim * 2, global_dim + local_dim)

    def forward(self, global_features, local_features):
        # Reshape to (T, B, C) for MultiheadAttention
        global_features = global_features.permute(1, 0, 2)
        local_features = local_features.permute(1, 0, 2)

        # Adjust local features if their dimensions differ from global features
        if self.adjust_local_dim:
            adjusted_local_features = self.adjust_local_dim(local_features)
        else:
            adjusted_local_features = local_features

        # Self-attention on global features
        global_self_attn, _ = self.global_attention(global_features, global_features, global_features)
        
        # Cross-attention, global features attending to adjusted local features
        global_cross_attn, _ = self.cross_attention(adjusted_local_features, global_features, global_features)
        
        # Concatenate both Attention results
        concatenated_features = torch.cat((global_self_attn, global_cross_attn), dim=2)
        
        # Final linear layer to adjust the concatenated output to the desired output dimension
        output = self.final_linear(concatenated_features)

        # Reshape output back to (B, T, C)
        output = output.permute(1, 0, 2)
        return output
    
if __name__ == '__main__':
    # Example usage
    global_input = torch.rand(2, 33, 256).cuda()  # Example global features
    local_input = torch.rand(2, 33, 256).cuda()   # Example local features
    block = SemanticAttentionModule(global_dim=256, local_dim=256, num_heads=8).cuda()
    output = block(global_input, local_input)
    print(global_input.size(), local_input.size(), output.size())
