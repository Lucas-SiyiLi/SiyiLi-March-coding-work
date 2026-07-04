import torch
import torch.nn as nn
import torch.nn.functional as f

class bp(nn.Module):
    def __init__(self, dims, seed=0):
        super(bp,self).__init__()
        torch.manual_seed(seed)
        layer =[]
        length = len(dims)
        for i in range(length-1):
            layer.append(nn.Linear(dims[i],dims[i+1],bias=0))
        self.network = nn.Sequential(*layer)
    def forward(self,x):
        output = self.network(x)
        return output
    def grad_compute(self,x,y):
        output = self.forward(x)
        loss = f.mse_loss(output,y)
        self.zero_grad()
        loss.backward()
        #这一句需要学习：p.grad.clone()
        grads = [p.grad.clone() for p in self.parameters()]
        return grads,loss.item()
    def update(self,grads,lr=0.01):
        with torch.no_grad():
            #这一句的目的是不要torch记录以下运算操作—节省内存
            for p,g in zip(self.parameters(),grads):
                p -= lr*g
