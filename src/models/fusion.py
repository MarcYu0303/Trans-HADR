import torch
import torch.nn as nn
import torch.nn.functional as F
from models.CrossAttention_1d import CrossAttentionLayer

class DeepFusion(nn.Module):
    def __init__(self, channel=32, embed_dim=128, num_heads=4, H=28, W=28):
        super(DeepFusion, self).__init__()
        self.num_heads = num_heads
        self.H = H
        self.W = W
        self.embed_dim = embed_dim

        # Linear layers to project the dimensions
        self.key_layer = nn.Linear(channel, embed_dim)
        self.value_layer = nn.Linear(channel, embed_dim)
        self.query_layer = nn.Linear(embed_dim, embed_dim)

        # Multihead attention layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
        # Output layer
        self.out_layer = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, rgb_feature, xyz_feature):
        B, C, H, W = rgb_feature.shape
        B, N, E = xyz_feature.shape
        shortcut = xyz_feature
        
        # Flatten modality A to (B, H*W, C)
        rgb_feature = rgb_feature.view(B, C, H * W).permute(0, 2, 1)  # Shape: (B, H*W, C)
        
        # Generate keys and values from modality A
        keys = self.key_layer(rgb_feature)  # Shape: (B, H*W, E)
        values = self.value_layer(rgb_feature)  # Shape: (B, H*W, E)
        
        # Generate queries from modality B
        queries = self.query_layer(xyz_feature)  # Shape: (B, N, E)
        
        # Apply multi-head attention
        attn_output, attn_output_weights = self.multihead_attn(queries, keys, values)
        
        # Final output
        output = self.out_layer(attn_output)
        output = torch.concat([output, shortcut], dim=-1)
        
        return output

class CrossAttentionFusion(nn.Module):
    def __init__(self, in_channel, depth, num_heads, use_layer_norm=True):
        super(CrossAttentionFusion, self).__init__()
        self.rgb_layers = CrossAttentionLayer(in_channel, depth, num_heads)
        self.xyz_layers = CrossAttentionLayer(in_channel, depth, num_heads)
        self.fuse = torch.nn.Linear(in_channel * 2, in_channel)
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.layer_norm = torch.nn.LayerNorm(in_channel)
    
    def forward(self, x):
        rgb_feature_map, xyz_feature_map = x
        rgb_out = self.rgb_layers(rgb_feature_map, xyz_feature_map)
        xyz_out = self.xyz_layers(xyz_feature_map, rgb_feature_map)
        
        x = torch.cat([rgb_out, xyz_out], dim=-1)
        x = self.fuse(x)
        
        if self.use_layer_norm:
            x = self.layer_norm(x)
        
        return x


class AdaptiveFusion(nn.Module):
    def __init__(self, rgb_embed_dim, xyz_embed_dim):
        super(AdaptiveFusion, self).__init__()
        self.embed_dim = rgb_embed_dim + xyz_embed_dim
        self.linear = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
    
    def forward(self, data_dict):
        concated_features = torch.cat([data_dict['intersect_rgb_feat'], data_dict['intersect_voxel_feat']], dim=-1)

        total_miss_sample_num = data_dict['total_miss_sample_num']
        miss_ray_intersect_idx = data_dict['miss_ray_intersect_idx']

        ray_averge = torch.zeros(total_miss_sample_num, self.embed_dim).to(concated_features.device)

        # Use scatter_add to sum up the features for each ray
        ray_sum = torch.zeros(total_miss_sample_num, concated_features.size(-1)).to(concated_features.device)
        ray_count = torch.zeros(total_miss_sample_num, dtype=torch.long).to(concated_features.device)

        ray_sum = ray_sum.scatter_add_(0, miss_ray_intersect_idx.unsqueeze(-1).expand(-1, concated_features.size(-1)), concated_features)
        ray_count = ray_count.scatter_add_(0, miss_ray_intersect_idx, torch.ones_like(miss_ray_intersect_idx))

        # Avoid division by zero
        nonzero_mask = ray_count > 0
        ray_averge[nonzero_mask] = ray_sum[nonzero_mask] / ray_count[nonzero_mask].unsqueeze(-1)
        
        # apply the linear transformation and the sigmoid function
        weights = torch.sigmoid(self.linear(ray_averge))
        
        # Apply weights to features in corresponding ray index
        weighted_features = concated_features * weights[miss_ray_intersect_idx]
        
        return weighted_features
        
class GatedFusion(nn.Module):
    def __init__(self, input_dim):
        super(GatedFusion, self).__init__()
        self.W_rgb = nn.Linear(2 * input_dim, input_dim)
        self.W_xyz = nn.Linear(2 * input_dim, input_dim)
    
    def forward(self, F_xyz, F_rgb):
        # Concatenate F_SA and F_DA along the last dimension
        concat_features = torch.cat((F_xyz, F_rgb), dim=-1)
        
        # Compute gates
        g_rgb = torch.sigmoid(self.W_rgb(concat_features))
        g_xyz = torch.sigmoid(self.W_xyz(concat_features))
        
        # Compute gated fusion
        F_fusion = g_rgb * F_rgb + g_xyz * F_xyz
        
        return torch.cat((F_xyz, F_rgb, F_fusion), dim=-1)    

if __name__ == '__main__':
    # Example usage
    B, C, H, W = 4, 64, 224, 224
    N, E = 100, 128
    num_heads = 4
    rgb_feature = torch.randn(B, C, H, W)
    xyz_feature = torch.randn(B, N, E)
    
    fusion_model = DeepFusion(channel=C, embed_dim=E, num_heads=num_heads, H=H, W=W)
    
    output = fusion_model(rgb_feature, xyz_feature)
    print(output.shape)  # Expected shape: (B, N, E)

    # cross_attention_fusion = CrossAttentionFusion(C, E, num_heads, H, W)
    # output = cross_attention_fusion(rgb_feature, xyz_feature)
    # print(output.shape)  # Expected shape: (B, N, E)
    
    # Example usage
    # N, E = 32, 64  # Batch size: 32, Feature size: 64
    # modality_a = torch.randn(N, E)
    # modality_b = torch.randn(N, E)

    # fusion_module = AdaptiveFusion(rgb_embed_dim=E, xyz_embed_dim=E)
    # output = fusion_module(modality_a, modality_b)
    # print(output.shape)  # Expected shape: (N, 2E)
