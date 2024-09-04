import torch
import torch.nn as nn
import torch.nn.functional as F
#普通流场生成，然后模态融合
class FeatureFusionModule(nn.Module):
    def __init__(self, global_dim, local_dim):
        super(FeatureFusionModule, self).__init__()
        self.global_dim = global_dim
        self.local_dim = local_dim

        # 网络用于生成流场特征
        self.flow_field_generator = nn.Sequential(
            nn.Linear(global_dim + local_dim, local_dim),  # 拼接后过一个线性层
            nn.ReLU(),
            nn.Linear(local_dim, local_dim)  # 输出与局部特征维度一致
        )

        # 最终输出网络
        self.output_network = nn.Sequential(
            nn.Linear(local_dim + global_dim, local_dim + global_dim),  # 拼接后的处理
            nn.ReLU(),
            nn.Linear(local_dim + global_dim, local_dim + global_dim)  # 最终输出的维度
        )

    def forward(self, global_features, local_features):
        # 拼接全局和局部特征
        combined_features = torch.cat([global_features, local_features], dim=-1)

        # 生成流场特征
        flow_field = self.flow_field_generator(combined_features)

        # 得到精细化的局部特征
        refined_local_features = local_features + flow_field

        # 拼接精细化的局部特征与全局特征
        final_features = torch.cat([refined_local_features, global_features], dim=-1)

        # 通过网络获取最终输出
        output = self.output_network(final_features)

        return output

if __name__ == '__main__':
    # 示例使用
    global_input = torch.rand(2, 33, 256)  # 全局特征 (假设 batch size = 2, sequence length = 10, global_dim = 256)
    local_input = torch.rand(2, 33, 512)   # 局部特征 (假设同样的 batch size 和 sequence length, local_dim = 128)
    model = FeatureFusionModule(global_dim=256, local_dim=512)
    output = model(global_input, local_input)
    print(global_input.size(), local_input.size(), output.size())
