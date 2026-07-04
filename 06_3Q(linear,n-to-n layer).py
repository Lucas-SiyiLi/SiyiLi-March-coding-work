#3Q-prediction phase&weight alignment + training phase
#linear version
#from bipcn to 3Q:more plausible neural implementation
#todo:weight alignment testment and unlinear try
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset

dims = [2,2,2]
num_sample = 100
batch_size = 8
x_sum = torch.randn(num_sample,dims[0])
y_sum = torch.ones(num_sample,dims[-1])
dataset = TensorDataset(x_sum,y_sum)
loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)

class model_3Q(nn.Module):
    def __init__(self,dims,seed=0):
        super(model_3Q,self).__init__()
        self.dims = dims
        self.layer = len(dims)
        torch.manual_seed(seed)
        self.weight_forward = nn.ParameterList([
            torch.randn(self.dims[i],self.dims[i+1])*0.01
            for i in range(self.layer-1)
        ])
        self.weight_backward = nn.ParameterList([
            torch.randn(self.dims[i+1],self.dims[i])*0.01
            for i in range(self.layer-1)
        ])
    def prediction_forward(self,x_list):
        v_forward=[]
        for i in range(self.layer-1):
            vf = x_list[i]@self.weight_forward[i]
            v_forward.append(vf)
        return v_forward
    def prediction_backward(self,x_list):
        v_backward = []
        for i in range(self.layer-1):
            vb = x_list[i+1]@self.weight_backward[i]
            v_backward.append(vb)
        return v_backward
    def forward_initialise(self,x):
        x_list = [x]
        for i in range(self.layer-1):
            xi = x_list[i]@self.weight_forward[i]
            x_list.append(xi)
        return x_list
    def energy(self,x_list,v_forward,v_backward):
        E = 0
        for i in range(self.layer-1):
            E += ((x_list[i]-v_backward[i])**2 + (x_list[i+1]-v_forward[i])**2).mean()
        return E
    def prediction(self,x,liter):
        #forward initialise
        x_list = self.forward_initialise(x)
        v_forward = self.prediction_forward(x_list)
        v_backward = self.prediction_backward(x_list)
        #update neuron
        for l in range(liter):
            for i in range(1,self.layer-1):
                x_list[i] = (v_forward[i]+v_backward[i])*(1/2)
            v_forward = self.prediction_forward(x_list)
            v_backward = self.prediction_backward(x_list)
        return x_list
    def weight_alignment(self,x_list,lr,liter=5):       
        #weight alignment
        for w in range(liter):
            #weight_forward
            v_forward = self.prediction_forward(x_list)
            for f in range(self.layer-1):
                E_up = x_list[f].T@(x_list[f+1]-v_forward[f])
                with torch.no_grad():
                    self.weight_forward[f] += lr*E_up
            #weight_backward
            v_backward = self.prediction_backward(x_list)
            for b in range(self.layer-1):
                E_down = x_list[b+1].T@(x_list[b]-v_backward[b])
                with torch.no_grad():
                    self.weight_backward[b] += lr*E_down
    def train_step(self,x,lr,liter,y=None):
        #clamp target
        x_list = self.prediction(x,liter)
        self.weight_alignment(x_list,lr=lr,liter=5)
        if y is not None:
            x_list[-1] = y.detach()
            x_list[-1].requires_grad_(False)
        v_forward = self.prediction_forward(x_list)
        v_backward = self.prediction_backward(x_list)
        #update neuron
        for l in range(liter):
            for i in range(1,self.layer-1):
                x_list[i] = (v_forward[i]+v_backward[i])*(1/2)
            v_forward = self.prediction_forward(x_list)
            v_backward = self.prediction_backward(x_list)
        #weight alignment
        x_list = self.weight_alignment(x_list,lr=lr,liter=5)
        return x_list
    def fit(self,loader,epoch,liter,lr):
        for ep in range(epoch):
            epoch_loss = 0
            for x,y in loader:
                x_list = self.train_step(x,lr,liter,y=y)
                v_foward = self.prediction_forward(x_list)
                v_backward = self.prediction_backward(x_list)
                e = self.energy(x_list,v_foward,v_backward)
                epoch_loss += e.item()
            average_energy = epoch_loss/len(loader)
            print(f'epoch={ep},average_energy={average_energy:.4f}')
