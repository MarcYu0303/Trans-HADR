import torch
import torch.nn as nn
import torch.nn.functional as F
#Transformer混淆信息后，再预测γ和β，缩放全局信息
class ModalFusionModule11(nn.Module):
    def __init__(self, global_dim, local_dim, num_heads=8, num_encoder_layers=2):
        super(ModalFusionModule11, self).__init__()
        self.global_dim = global_dim
        self.local_dim = local_dim

        # Dimension alignment for local features, if necessary
        if global_dim != local_dim:
            self.align_dim = nn.Linear(local_dim, global_dim)
        else:
            self.align_dim = None

        # Transformer Encoder for fusing global and local features
        encoder_layer = nn.TransformerEncoderLayer(d_model=global_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Predictors for gamma and beta
        self.gamma_predictor = nn.Sequential(
            nn.Linear(global_dim, global_dim),
            nn.Sigmoid()
        )
        self.beta_predictor = nn.Linear(global_dim, global_dim)

        self.output_layer = nn.Linear(global_dim, global_dim + local_dim)

    def forward(self, global_features, local_features):
        if self.align_dim:
            local_features = self.align_dim(local_features)

        # Combine global and local features
        combined_features = torch.cat([global_features, local_features], dim=1)
        
        # Transformer encoder to fuse features
        combined_features = self.transformer_encoder(combined_features.permute(1, 0, 2)).permute(1, 0, 2)

        # Slice back the global features for gamma and beta application
        fused_global_features = combined_features[:, :global_features.size(1), :]

        # Calculate gamma and beta from fused global features
        gamma = self.gamma_predictor(fused_global_features)
        beta = self.beta_predictor(fused_global_features)

        # Adjust global features using gamma and beta
        adjusted_global_features = (1 + gamma) * global_features + beta
        output = self.output_layer(adjusted_global_features)

        return output
    
if __name__ == '__main__':
    # Example usage
    global_input = torch.rand(2, 33, 256)  # Example global features
    local_input = torch.rand(2, 33, 512)   # Example local features
    block = ModalFusionModule11(global_dim=256, local_dim=512, num_heads=8, num_encoder_layers=2)
    output = block(global_input, local_input)
    print(global_input.size(), local_input.size(), output.size())
