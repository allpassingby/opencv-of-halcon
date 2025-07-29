import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # 定义 Query, Key 和 Value 的线性变换
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        # 定义输出线性变换
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Q, K, V 变换
        Q = self.query(x)  # (batch_size, seq_len, d_model)
        K = self.key(x)  # (batch_size, seq_len, d_model)
        V = self.value(x)  # (batch_size, seq_len, d_model)

        # 将 d_model 分割为多个头
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_model // self.num_heads)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_model // self.num_heads)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_model // self.num_heads)

        # 转置维度以便计算注意力
        Q = Q.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, d_model // num_heads)
        K = K.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, d_model // num_heads)
        V = V.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, d_model // num_heads)

        # 计算 Q 和 K 的点积，得到 attention scores
        attention_scores = torch.matmul(Q, K.permute(0, 1, 3, 2))  # (batch_size, num_heads, seq_len, seq_len)
        attention_scores = attention_scores / (self.d_model ** 0.5)  # 缩放

        # 使用 softmax 计算注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)

        # 将注意力权重应用于 V
        output = torch.matmul(attention_weights, V)  # (batch_size, num_heads, seq_len, d_model // num_heads)

        # 将多个头拼接在一起
        output = output.permute(0, 2, 1, 3).contiguous()  # (batch_size, seq_len, num_heads, d_model // num_heads)
        output = output.view(batch_size, seq_len, self.d_model)  # (batch_size, seq_len, d_model)

        # 线性变换输出
        output = self.fc_out(output)

        return output


# 简单测试
batch_size = 2
seq_len = 5
d_model = 16
num_heads = 4

# 随机输入一个张量，模拟一个batch的输入序列（如图像特征）
x = torch.rand(batch_size, seq_len, d_model)

# 创建自注意力层
self_attention = SelfAttention(d_model=d_model, num_heads=num_heads)

# 获取输出
output = self_attention(x)

# 打印输出的形状
print("Output shape:", output.shape)  # 应该是 (batch_size, seq_len, d_model)
