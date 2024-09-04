import torch
import torch.nn.functional as F

class CrossAttentionBlock(torch.nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.query_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.key_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.value_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.out_proj = torch.nn.Linear(embed_dim, embed_dim)
        
        self.layer_norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, x1, x2):
        if x1.ndim == 2:
            x1 = x1.unsqueeze(1)
            x2 = x2.unsqueeze(1)
        
        B, L, C = x1.size()
        H = self.num_heads
        D = self.head_dim
        shortcut = x1
        x1 = self.layer_norm(x1)

        # Project inputs to queries, keys, and values
        queries = self.query_proj(x1).view(B, L, H, D).transpose(1, 2)  # (B, H, L, D)
        keys = self.key_proj(x2).view(B, L, H, D).transpose(1, 2)      # (B, H, L, D)
        values = self.value_proj(x2).view(B, L, H, D).transpose(1, 2)  # (B, H, L, D)

        # Compute scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (D ** 0.5)  # (B, H, L, L)
        attn_weights = F.softmax(scores, dim=-1)  # (B, H, L, L)

        # Compute weighted sum of values
        attn_output = torch.matmul(attn_weights, values)  # (B, H, L, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, C)  # (B, L, C)
        attn_output = attn_output + shortcut # Add residual connection

        # Final linear layer
        output = self.out_proj(attn_output)  # (B, L, C)
        output = output + x1  # Add residual connection
        
        return output

class CrossAttentionLayer(torch.nn.Module):
    def __init__(self, in_channel, depth, num_heads):
        super(CrossAttentionLayer, self).__init__()
        self.layers = torch.nn.ModuleList([
            CrossAttentionBlock(in_channel, num_heads) for _ in range(depth)
        ])

    def forward(self, x1, x2):
        for layer in self.layers:
            x1 = layer(x1, x2)
        return x1


class CrossAttention(torch.nn.Module):
    def __init__(self, in_channel, depth, num_heads):
        super(CrossAttention, self).__init__()
        self.rgb_layers = CrossAttentionLayer(in_channel, depth, num_heads)
        self.xyz_layers = CrossAttentionLayer(in_channel, depth, num_heads)
        self.fuse = torch.nn.Linear(in_channel * 2, in_channel)
    
    def forward(self, x):
        rgb_feature_map, xyz_feature_map = x
        rgb_out = self.rgb_layers(rgb_feature_map, xyz_feature_map)
        xyz_out = self.xyz_layers(xyz_feature_map, rgb_feature_map)
        
        x = torch.cat([rgb_out, xyz_out], dim=-1)
        x = self.fuse(x)
        
        return x



if __name__ == "__main__":
    # Example usage
    B, L, C = 2, 10, 64
    x1 = torch.rand(B, L, C)  # Feature set 1
    x2 = torch.rand(B, L, C)  # Feature set 2

    cross_attn = CrossAttention(in_channel=C, depth=2, num_heads=1)
    output = cross_attn((x1, x2))

    print(output.size())  # torch.Size([2, 10, 64])