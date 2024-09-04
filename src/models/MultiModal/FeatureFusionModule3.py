import torch
import torch.nn as nn
import torch.nn.functional as F
#预测置信度，并拼接融合模态
class FeatureProcessor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FeatureProcessor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        self.confidence = nn.Sequential(
            nn.Linear(input_dim, 1),  # 输出单一置信度分数
            nn.Sigmoid()  # 使用 Sigmoid 激活函数确保输出在0到1之间
        )

    def forward(self, x):
        processed_features = self.network(x)
        confidence_scores = self.confidence(x).squeeze(-1)  # 确保无多余维度
        return processed_features, confidence_scores

class FeatureFusionModule3(nn.Module):
    def __init__(self, global_dim, local_dim):
        super(FeatureFusionModule3, self).__init__()
        self.global_processor = FeatureProcessor(global_dim, global_dim)
        self.local_processor = FeatureProcessor(local_dim, local_dim)
        self.relative_confidence_network = FeatureProcessor(global_dim + local_dim, 1)  # 保持单一置信度输出

        # 融合网络
        self.fusion_network = nn.Sequential(
            nn.Linear(global_dim + local_dim + 3, global_dim + local_dim),  # 3个置信度
            nn.ReLU(),
            nn.Linear(global_dim + local_dim, global_dim + local_dim)
        )

    def forward(self, global_features, local_features):
        refined_global, global_confidence = self.global_processor(global_features)
        refined_local, local_confidence = self.local_processor(local_features)

        # 拼接全局和局部特征
        combined_features = torch.cat([refined_global, refined_local], dim=-1)
        relative_confidence = self.relative_confidence_network(combined_features)[1]  # 获取置信度

        # 拼接所有特征和置信度
        fusion_input = torch.cat([refined_global, refined_local, global_confidence.unsqueeze(-1), local_confidence.unsqueeze(-1), relative_confidence.unsqueeze(-1)], dim=-1)
        fused_features = self.fusion_network(fusion_input)

        return fused_features
    
if __name__ == '__main__':
    # 示例使用
    global_input = torch.rand(2, 10, 256)  # 全局特征
    local_input = torch.rand(2, 10, 512)   # 局部特征
    model = FeatureFusionModule3(global_dim=256, local_dim=512)
    output = model(global_input, local_input)
    print(global_input.size(), local_input.size(), output.size())
