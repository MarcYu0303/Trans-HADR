import torch
import torch.nn as nn
import torch.nn.functional as F
#多头注意力生成流场，然后模态融合
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        values = values.reshape(N, -1, self.heads, self.head_dim)
        keys = keys.reshape(N, -1, self.heads, self.head_dim)
        queries = query.reshape(N, -1, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, -1, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out

class FeatureFusionModule2(nn.Module):
    def __init__(self, global_dim, local_dim, num_heads=8):
        super(FeatureFusionModule2, self).__init__()
        self.global_dim = global_dim
        self.local_dim = local_dim

        self.local_dim_adjust = nn.Linear(local_dim, global_dim)

        # Adjust the fc_out in MultiHeadAttention to match the adjusted local dimension
        self.flow_field_generator = MultiHeadAttention(global_dim * 2, num_heads)
        self.flow_field_generator.fc_out = nn.Linear(num_heads * (global_dim * 2 // num_heads), global_dim)

        self.output_network = nn.Sequential(
            nn.Linear(global_dim * 2, global_dim + local_dim),
            nn.ReLU(),
            nn.Linear(global_dim + local_dim, global_dim + local_dim)
        )

    def forward(self, global_features, local_features):
        local_features_adjusted = self.local_dim_adjust(local_features)
        combined_features = torch.cat([global_features, local_features_adjusted], dim=-1)

        flow_field = self.flow_field_generator(combined_features, combined_features, combined_features)

        refined_local_features = local_features_adjusted + flow_field

        final_features = torch.cat([refined_local_features, global_features], dim=-1)
        output = self.output_network(final_features)

        return output
if __name__ == '__main__':
    # Example usage
    global_input = torch.rand(2, 10, 256)
    local_input = torch.rand(2, 10, 512)
    model = FeatureFusionModule2(global_dim=256, local_dim=512)
    output = model(global_input, local_input)
    print(global_input.size(), local_input.size(), output.size())
