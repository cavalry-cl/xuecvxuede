import torch

x = torch.BoolTensor([[1,1,1,0,0],[1,1,1,1,0]])
m = 4
b = 2
n = 5
y=x.unsqueeze(1).expand(-1,m,-1)
print(y)
