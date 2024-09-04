import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerFeatureProcessor(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers):
        super(TransformerFeatureProcessor, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.confidence = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.transformer_encoder(x)
        confidence_scores = self.confidence(out).squeeze(-1)
        return out, confidence_scores

class FeatureFusionModule4(nn.Module):
    def __init__(self, global_dim, local_dim, num_heads=4, num_layers=2):
        super(FeatureFusionModule4, self).__init__()
        self.global_dim = global_dim
        self.local_dim = local_dim
        self.global_processor = TransformerFeatureProcessor(global_dim, num_heads, num_layers)
        self.local_processor = TransformerFeatureProcessor(local_dim, num_heads, num_layers)
        self.relative_confidence_network = nn.Sequential(
            nn.Linear(global_dim + local_dim, 1),
            nn.Sigmoid()
        )
        
        # 融合网络
        self.fusion_network = nn.Sequential(
            nn.Linear(global_dim + local_dim + 3, global_dim + local_dim),
            nn.ReLU(),
            nn.Linear(global_dim + local_dim, global_dim + local_dim)
        )

    def forward(self, global_features, local_features):
        refined_global, global_confidence = self.global_processor(global_features)
        refined_local, local_confidence = self.local_processor(local_features)

        # 拼接最后的特征以及计算相对置信度
        combined_features = torch.cat([refined_global, refined_local], dim=-1)
        relative_confidence = self.relative_confidence_network(combined_features).squeeze(-1)
        # 拼接所有特征和置信度
        fusion_input = torch.cat([
            refined_global,
            refined_local,
            global_confidence.unsqueeze(-1),
            local_confidence.unsqueeze(-1),
            relative_confidence.unsqueeze(-1)  # 复制置信度以匹配特征维数
        ], dim=-1)
        fused_features = self.fusion_network(fusion_input)
        return fused_features
    
if __name__ == '__main__':
    # 示例使用
    batch_size = 2
    seq_len = 10
    global_input = torch.rand(batch_size, seq_len, 256)  # 全局特征
    local_input = torch.rand(batch_size, seq_len, 512)   # 局部特征
    model = FeatureFusionModule4(global_dim=256, local_dim=512)
    output = model(global_input, local_input)
    print(global_input.size(), local_input.size(), output.size())
