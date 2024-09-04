import torch
import torch.nn as nn
import torch.nn.functional as F
#局部特征+全局特征预测γ和β，缩放全局特征
class ModalFusionModule10(nn.Module):
    def __init__(self, global_dim, local_dim):
        super(ModalFusionModule10, self).__init__()
        self.global_dim = global_dim
        self.local_dim = local_dim

        # Dimension alignment for local features, if necessary
        if global_dim != local_dim:
            self.align_dim = nn.Linear(local_dim, global_dim)
        else:
            self.align_dim = None

        # Simple attention mechanism to process local features
        self.attention = nn.Sequential(
            nn.Linear(global_dim, local_dim),
            nn.Tanh(),
            nn.Linear(local_dim, local_dim),
            nn.ReLU()
        )

        # Predictors for gamma and beta, now taking concatenated global and local features
        self.gamma_predictor = nn.Sequential(
            nn.Linear(global_dim + local_dim, global_dim),
            nn.Sigmoid()  # Ensure gamma is non-negative
        )
        self.beta_predictor = nn.Linear(global_dim + local_dim, global_dim)

        self.output_layer = nn.Linear(self.global_dim, self.global_dim + self.local_dim)
        
    def forward(self, global_features, local_features):
        if self.align_dim:
            local_features = self.align_dim(local_features)
        
        # Process local features through attention
        processed_local_features = self.attention(local_features)

        # Concatenate global and processed local features
        combined_features = torch.cat([global_features, processed_local_features], dim=-1)

        # Calculate gamma and beta from combined features
        gamma = self.gamma_predictor(combined_features)
        beta = self.beta_predictor(combined_features)

        # Adjust global features using gamma and beta
        adjusted_global_features = (1 + gamma) * global_features + beta
        output = self.output_layer(adjusted_global_features)

        return output
    
if __name__ == '__main__':
    # Example usage
    global_input = torch.rand(2, 33, 256)  # Example global features
    local_input = torch.rand(2, 33, 512)   # Example local features with different dimension
    block = ModalFusionModule10(global_dim=256, local_dim=512)
    output = block(global_input, local_input)
    print(global_input.size(), local_input.size(), output.size())
