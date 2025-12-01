import torch
from torch import nn
import numpy as np
from ipdb import set_trace as st

class Sine(nn.Module):
    def __init__(self, w0=30):
        super().__init__()
        self.w0 = w0

    def forward(self, input):
        # 论文 Sec. 3.2 提到的 30 因子
        return torch.sin(self.w0 * input)

class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=True, first_omega_0=30, hidden_omega_0=30):
        super().__init__()
        
        self.net = []
        
        # 1. 第一层 (First Layer)
        self.net.append(nn.Linear(in_features, hidden_features))
        self.net.append(Sine(first_omega_0))
        
        # 2. 中间隐藏层 (Hidden Layers)
        # 根据原代码逻辑，num_hidden_layers 是指除第一层外的隐藏层数量
        for _ in range(hidden_layers):
            self.net.append(nn.Linear(hidden_features, hidden_features))
            self.net.append(Sine(hidden_omega_0))
            
        # 3. 输出层 (Output Layer)
        if outermost_linear:
            self.net.append(nn.Linear(hidden_features, out_features))
        else:
            self.net.append(nn.Linear(hidden_features, out_features))
            self.net.append(Sine(hidden_omega_0))
        
        # 将列表转换为 Sequential 容器
        self.net = nn.Sequential(*self.net)
        
        # 4. 初始化权重
        self.net.apply(self.init_weights)
        
        # 对第一层应用特殊的初始化
        # self.net[0] 是第一个 Linear 层
        self.first_layer_init(self.net[0])

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                num_input = m.weight.size(-1)
                # 对应原代码中的 sine_init
                # See supplement Sec. 1.5 for discussion of factor 30
                limit = np.sqrt(6 / num_input) / 30 
                m.weight.uniform_(-limit, limit)
                # 原代码虽未显式初始化 bias，但通常设为 0 或保持默认
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def first_layer_init(self, m):
        if isinstance(m, nn.Linear):
            with torch.no_grad():
                num_input = m.weight.size(-1)
                # 对应原代码中的 first_layer_sine_init
                limit = 1 / num_input
                m.weight.uniform_(-limit, limit)

    def forward(self, coords):
        # 允许输入包含 batch 维度
        output = self.net(coords)
        return output



if __name__ == '__main__':
    # --- 实例化示例 ---

    # 按照你给出的结构：in=2, hidden=256, hidden_layers=3 (导致中间有3个块), out=4
    # 注意：原代码的 num_hidden_layers=3 会生成 1(first) + 3(hidden) + 1(out) = 5 个线性层
    model = Siren(
        in_features=2, 
        hidden_features=256, 
        hidden_layers=3, 
        out_features=4
    )

    input = torch.randn(1, 2)
    output = model(input)
    print(output.shape)

    print(model)
    st()