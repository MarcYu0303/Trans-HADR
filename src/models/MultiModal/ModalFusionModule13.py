import torch
import torch.nn as nn
import torch.nn.functional as F
#根据自适应网络学习到的权重动态地重加权全局特征和局部特征
class AdaptiveFeatureSelection(nn.Module):
    def __init__(self, input_dim):
        super(AdaptiveFeatureSelection, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // 2)
        self.fc2 = nn.Linear(input_dim // 2, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        weights = self.fc1(x)
        weights = F.relu(weights)
        weights = self.fc2(weights)
        weights = self.softmax(weights)
        return weights * x

class ModalFusionModule13(nn.Module):
    def __init__(self, global_dim, local_dim):
        super(ModalFusionModule13, self).__init__()
        self.global_dim = global_dim
        self.local_dim = local_dim

        # Dimension alignment for local features, if necessary
        if global_dim != local_dim:
            self.align_dim = nn.Linear(local_dim, global_dim)
        else:
            self.align_dim = None

        # Feature selection modules for both global and local features
        self.afs_global = AdaptiveFeatureSelection(global_dim)
        self.afs_local = AdaptiveFeatureSelection(local_dim if self.align_dim is None else global_dim)

        # Predictors for gamma and beta for both global and local features
        self.gamma_global = nn.Sequential(
            nn.Linear(global_dim, global_dim),
            nn.Sigmoid()
        )
        self.beta_global = nn.Linear(global_dim, global_dim)
        self.gamma_local = nn.Sequential(
            nn.Linear(global_dim, global_dim),
            nn.Sigmoid()
        )
        self.beta_local = nn.Linear(global_dim, global_dim)

        # Output layer will now handle the concatenated adjusted features
        self.output_layer = nn.Linear(2 * global_dim, local_dim+global_dim)

    def forward(self, global_features, local_features):
        if self.align_dim:
            local_features = self.align_dim(local_features)

        # Apply Adaptive Feature Selection
        global_features = self.afs_global(global_features)
        local_features = self.afs_local(local_features)

        # Calculate gamma and beta for global and local features
        gamma_global = self.gamma_global(global_features)
        beta_global = self.beta_global(global_features)
        gamma_local = self.gamma_local(local_features)
        beta_local = self.beta_local(local_features)

        # Adjust features using gamma and beta
        adjusted_global_features = (1 + gamma_global) * global_features + beta_global
        adjusted_local_features = (1 + gamma_local) * local_features + beta_local

        # Concatenate the adjusted features
        combined_features = torch.cat([adjusted_global_features, adjusted_local_features], dim=-1)
        output = self.output_layer(combined_features)

        return output
    
if __name__ == '__main__':
    # Example usage
    global_input = torch.rand(2, 33, 256)  # Example global features
    local_input = torch.rand(2, 33, 512)   # Example local features, aligned dimensions
    block = ModalFusionModule13(global_dim=256, local_dim=512)
    output = block(global_input, local_input)
    print(global_input.size(), local_input.size(), output.size())
