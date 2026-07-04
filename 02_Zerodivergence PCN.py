#pcn变体——zero-divergence pcn(结果完全等效于backpropagation—不需要等到完全收敛就进行更新权重)
#与pcn在prediction和weight update环节相同
#区别在inference部分
#inference过程：每一次权重前面节点产生error之后进行权重更新
#需要满足learning rate =1
##p-pcn(同步进行inference和weight update)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
data = 100
batch_size = 8
dims = [2,2,2,2]
x1 = torch.randn(data,dims[0])
y1 = torch.ones(data,dims[-1])
dataset = TensorDataset(x1,y1)
loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
class z_pcn(nn.Module):
    def __init__(self, dims,activation=torch.relu,device="cpu"):
        super(z_pcn,self).__init__()
        self.dims = dims
        self.layer = len(dims)
        self.device = device
        self.activation = activation
        #initialize weight
        self.weight = nn.ParameterList([
            nn.Parameter(torch.randn(self.dims[i],self.dims[i+1])*0.01)
            for i in range(self.layer-1)
        ])
    def prediction(self,inputx,x_list=None):
        if x_list is None:
            x_list = [inputx]
            for i in range(1,self.layer):
                x = torch.randn(inputx.size(0),self.dims[i],requires_grad=True)
                x_list.append(x)
        else:
            pass    
        v_list = []
        for j in range(self.layer-1):
            v = x_list[j]@self.weight[j]
            if j < self.layer - 2:
                v = self.activation(v)
            v_list.append(v)
        return x_list,v_list
    def inference_energy(self,x_list,v_list):
        #x_list,v_list = self.prediction(inputx)
        #这里有个bug，在下面train的函数里面也会调用prediction，所以会存在两个state被记录
        #if target is not None:
        #    x_list[-1].data=target.detach()
        #    x_list[-1].requires_grad = False
        E=0
        for x in range(1,self.layer):
            E +=((x_list[x]-v_list[x-1])**2).mean()
        return E
    def weight_update_energy(self,x_list,v_list):
        #x_list,v_list = self.prediction(inputx)
        E = 0
        for w in range(1,self.layer-1):
            E +=((x_list[w]-v_list[w-1])**2).mean()
        return E
    #review:实际上，在同时进行inference和权重更新的时候，只需要一个算energy的函数即可
    def train_in_zpcn(self,inputx,target):
        #前向初始化prediction
        x_list,v_list = self.prediction(inputx,x_list=None)
        x_list[-1].data = target.detach()
        x_list[-1].requires_grad = False
        #error:说明v只算了一次—之后的每一次x更新后未更新v
        #需要调整prediction函数
        optimizerx = [torch.optim.Adam([x],lr=0.01) for x in x_list[1:-1]]
        optimizerw = [torch.optim.Adam([w],lr=0.01) for w in self.weight]
        #逐层更新
        for i in reversed(range(1,self.layer-1)):
            __,v_list = self.prediction(inputx,x_list=x_list)
            E = self.inference_energy(x_list,v_list)
            optimizerx[i-1].zero_grad()
            optimizerw[i-1].zero_grad()
            E.backward()
            optimizerx[i-1].step()
            optimizerw[i-1].step()
        #update w2
        E2 = self.weight_update_energy(x_list,v_list)
        #注意，这样写旧梯度会残留
        #E.backward()
        optimizerw[-1].zero_grad()
        E2.backward()
        optimizerw[-1].step()    
        return x_list,self.weight,E
    
    def fit(self,loader,epoch=10,liter=20):
        for ep in range(epoch):
            epoch_loss = 0
            for x1,y1 in loader:
                for _ in range(liter):
                    _,_,E = self.train_in_zpcn(inputx=x1,target=y1) 
                epoch_loss += E.item()
            average_energy = epoch_loss/len(loader)
            print(f'epoch{ep},energy{average_energy:.4f}')
#evoke model
model = z_pcn(dims)
model.fit(loader)
#test group
x_input = torch.randn(batch_size,dims[0])
y_target = torch.ones(batch_size,dims[-1])
xlistfinal,wlistfinal,e = model.train_in_zpcn(inputx=x_input,target=y_target)
print("the output in the teat is:")
print(xlistfinal[-1])
