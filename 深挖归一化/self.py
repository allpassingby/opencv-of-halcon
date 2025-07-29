import torch
import torch.nn.functional as F
import torch.nn as nn

batch_size = 2
seq_len = 5
d_model = 16
num_heads = 4

# 随机输入一个张量，模拟一个batch的输入序列（如图像特征）
x = torch.rand(batch_size, seq_len, d_model)

class selfAttention(nn.Module):
    def __init__(self,d_model,num_head):
        super(selfAttention, self).__init__()
        self.d_model = d_model
        self.num_head = num_head
        assert d_model % num_head == 0, "d_model must be divisible by num_head"

        self.query = nn.Linear(d_model,d_model)
        self.key = nn.Linear(d_model,d_model)
        self.value = nn.Linear(d_model,d_model)
        self.out = nn.Linear(d_model,d_model)

    def forward(self, x):
        bz = x.shape[0]
        length = x.shape[1]

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(bz, length, self.num_head, self.d_model//self.num_head)
        K = K.view(bz, length, self.num_head, self.d_model//self.num_head)
        V = V.view(bz, length, self.num_head, self.d_model//self.num_head)

        Q=Q.permute(0,2,1,3)
        K=K.permute(0,2,1,3)
        V=V.permute(0,2,1,3)

        attention_scores = torch.matmul(Q,K.permute(0,1,3,2))
        attention_scores =  attention_scores/ (self.d_model ** 0.5)

        attention_weight = F.softmax(attention_scores,dim=-1)

        out = torch.matmul(attention_weight,V)

        out_put = out.permute(0,2,1,3).contiguous()
        out_put = out.view(bz,length,self.d_model)
        out_put = self.out(out_put)

        return out_put

batch_size =1
length = 5
model = 64
num_heads = 4

x = torch.rand(batch_size, seq_len, d_model)

sa = selfAttention(d_model,num_heads)

sa_out = sa(x)

print(sa_out.shape)





