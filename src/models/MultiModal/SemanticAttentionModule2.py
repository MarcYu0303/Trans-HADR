import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticAttentionModule2(nn.Module):
    def __init__(self, global_dim, local_dim, num_heads=8):
        super(SemanticAttentionModule2, self).__init__()
        self.global_dim = global_dim
        self.local_dim = local_dim
        self.num_heads = num_heads
        
        # Attention layers
        self.global_attention = nn.MultiheadAttention(self.global_dim, num_heads)
        self.cross_attention = nn.MultiheadAttention(self.global_dim, num_heads)
        
        # Conditionally adjust local dimensions to match global dimensions
        if self.global_dim != self.local_dim:
            self.adjust_local_dim = nn.Linear(self.local_dim, self.global_dim)
        else:
            self.adjust_local_dim = None
        
        # Layer Norms
        self.norm1 = nn.LayerNorm(self.global_dim)
        self.norm2 = nn.LayerNorm(self.global_dim)
        
        # Feedforward layers
        self.feedforward = nn.Sequential(
            nn.Linear(self.global_dim * 2, self.global_dim * 2),
            nn.ReLU(),
            nn.Linear(self.global_dim * 2, self.global_dim + self.local_dim)
        )

    def forward(self, global_features, local_features):
        # Ensure correct shape for MultiheadAttention (T, B, C)
        global_features = global_features.permute(1, 0, 2)
        local_features = local_features.permute(1, 0, 2)

        # Adjust local features dimension if necessary
        if self.adjust_local_dim:
            adjusted_local_features = self.adjust_local_dim(local_features)
        else:
            adjusted_local_features = local_features

        # Self-attention on global features
        global_self_attn, _ = self.global_attention(global_features, global_features, global_features)
        global_self_attn = self.norm1(global_self_attn + global_features)
        
        # Cross-attention, global on adjusted local
        global_cross_attn, _ = self.cross_attention(adjusted_local_features, global_features, global_features)
        global_cross_attn = self.norm2(global_cross_attn + adjusted_local_features)
        
        # Concatenate both Attention results
        concatenated_features = torch.cat((global_self_attn, global_cross_attn), dim=2)
        
        # Process through feedforward network
        output = self.feedforward(concatenated_features)
        
        # Reshape output back to (B, T, C)
        output = output.permute(1, 0, 2)
        return output
    
if __name__ == '__main__':
    # Example of creating the module
    global_input = torch.rand(2, 33, 256).cuda()  # Example global features
    local_input = torch.rand(2, 33, 256).cuda()   # Example local features
    block = SemanticAttentionModule2(global_dim=256, local_dim=256, num_heads=8).cuda()
    output = block(global_input, local_input)
    print(global_input.size(), local_input.size(), output.size())
