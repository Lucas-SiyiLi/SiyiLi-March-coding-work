#Bidirectional pcn
#generate + discriminate
#todo:unlinear 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

sample_num = 100
batch_size = 8
dims = [2,2,2,2]
x_num = torch.randn(sample_num,dims[0])
y_num = torch.zeros(sample_num,dims[-1])
dataset = TensorDataset(x_num,y_num)
loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

class BiPCN(nn.Module):
    def __init__(self, dims,activation = torch.relu,seed=0):
        super(BiPCN,self).__init__()
        self.dims = dims
        self.layer = len(dims)
        self.activation = activation
        torch.manual_seed(seed)
        self.weight_forward = nn.ParameterList([
            torch.randn(self.dims[i],self.dims[i+1])*0.01
            for i in range(self.layer-1)
        ])
        self.weight_backward =nn.ParameterList([
            torch.randn(self.dims[l+1],self.dims[l])*0.01
            for l in range(self.layer-1)
        ])
    def forward(self,x_list):
        v_forward = []
        for i in range(self.layer-1):
            vi= x_list[i]@self.weight_forward[i]
            v_forward.append(vi)
        return v_forward
    def backward(self,x_list):
        v_backward = []
        for i in range(self.layer-1):
            vi= x_list[i+1]@self.weight_backward[i]
            v_backward.append(vi)
        return v_backward
    def energy_optim(self,x_list):
        #双向误差
        v_forward = self.forward(x_list)
        v_backward = self.backward(x_list)
        E = 0
        for i in range(self.layer-1):
            E += ((x_list[i+1]-v_forward[i])**2+
                  (x_list[i]-v_backward[i])**2).mean()
            #统一index方向
        return E
    def inference_optim(self,x,lr,liter,y=None):
        x_list=[x]
        for i in range(1,self.layer):
            xi = torch.zeros(x.size(0),self.dims[i])
            x_list.append(xi)
        if y is not None:
            y = y.detach() #将y从计算图中分离出来
            y.requires_grad_(False)
            x_list[-1] = y
        optimizerx = torch.optim.Adam(x_list[1:-1],lr=lr)
        for w in range(liter):
            optimizerx.zero_grad()
            E =self.energy_optim(x_list)
            E.backward()
            optimizerx.step()
        return x_list,E
    def inference(self,x,T,lr,y=None):
        # -------- 初始化 --------
        x_list=[x]
        #bpcn：forward initialization
        for i in range(self.layer-1):
            xi = x_list[i]@self.weight_forward[i]
            x_list.append(xi)
        if y is not None:
            x_list[-1] = y
        ## -------- inference --------
        #手写梯度更新—能够体现bipcn在neural implementation上的冗余
        for t in range(T):
            new_x_list = [xi.clone() for xi in x_list]
            v_forward = self.forward(x_list)
            v_backward = self.backward(x_list)
            E_before = self.energy_optim(x_list)
            for l in range(1,self.layer-1):
                #discriminate
                e_up = x_list[l]-v_forward[l-1]
                e_up_back = x_list[l+1]-v_forward[l]
                grad_up = -e_up + e_up_back@self.weight_forward[l].T
                #generate
                e_down = x_list[l]-v_backward[l]
                e_down_back = x_list[l-1]-v_backward[l-1]
                grad_down = -e_down + e_down_back@self.weight_backward[l-1].T
                #update
                dx = grad_up +grad_down
                new_x_list[l] =x_list[l] + lr*dx
            E_after = self.energy_optim(new_x_list)
            x_list = new_x_list
        return x_list,E_before,E_after
    def train_forward(self,x,y,lr):
        x_list,_,_ = self.inference(x,T=20,lr=lr,y=y)
        #E = self.energy(x_list)
        #E.backward()
        E = 0
        for i in range(self.layer-1):
                dx_up = x_list[i].T@(x_list[i+1]-v_forward[i])
                with torch.no_grad():
                    self.weight_forward[i].data += lr*dx_up
        v_forward = self.forward(x_list)         
        for p in range(1,self.layer-1):
            E += F.mse_loss(x_list[p],v_forward[p-1])
        return E
    def train_backward(self,x,y,lr):
        E = 0
        x_list,_,_ = self.inference(x,T=20,lr=lr,y=y)
        for i in range(self.layer-1):
                dx_down = x_list[i+1].T@(x_list[i]-v_backward[i])
                #注意这里要transition
                with torch.no_grad():
                    self.weight_backward[i].data += lr*dx_down
        v_backward = self.backward(x_list)
        for p in range(1,self.layer-1):
            E += F.mse_loss(x_list[p],v_backward[p-1])    
        return E
    def fit(self,x,y,loader,epoch,lr=0.01):
        #forward training
        for ep in range(epoch):
            epoch_loss = 0
            for x,y in loader:
                E = self.train_forward(x,y,lr)
                epoch_loss += E.item()
            average_loss = epoch_loss/len(loader)
            print(f'epoch={ep},average_loss={average_loss:.4f}')
        #backward training
        for ep in range(epoch):
            epoch_loss = 0
            for x,y in loader:
                E = self.train_backward(x,y,lr)
                epoch_loss += E.item()        
            average_loss = epoch_loss/len(loader)
            print(f'epoch={ep},average_loss={average_loss:.4f}')
