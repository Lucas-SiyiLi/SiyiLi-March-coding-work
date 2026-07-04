import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset

dims = [4,3,2,1]
num_sample = 50
batch_size = 8
x_data = torch.randn(num_sample,dims[0]) 
y_data = torch.ones(num_sample,dims[-1])  
dataset = TensorDataset(x_data,y_data) 
train_loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
class pcn(nn.Module):  
    def __init__(self, dims, activation=torch.relu):
        super().__init__()
        self.dims = dims
        self.n_layers = len(dims)
        self.activation = activation 
        self.weights =  nn.ParameterList([  
            nn.Parameter(torch.randn(dims[i],dims[i+1])*0.1)
            for i in range(self.n_layers - 1)
        ])
    #prediction
    def prediction(self,x_list):
        v_list = [x_list[0]]
        for l in range(self.n_layers -1):
            v = x_list[l]@self.weights[l] #linear output
            if l <self.n_layers-2:
                v = self.activation(v) #no activation in the last layer
            v_list.append(v)
        return v_list
    
    #inference & training inference
    def inference(self,x_batch,y_batch=None,liter=10,lr=0.01):
        energy_trace = []
        x_list = [x_batch]  #固定输入
        for i in range(1, self.n_layers):
            x_next = torch.zeros(x_batch.size(0), self.dims[i], requires_grad=True)
            x_list.append(x_next) 
        if y_batch is not None:
            x_list[-1].data = y_batch.detach()
            x_list[-1].requires_grad = False
        #update neuron activity
        optimizer1 = torch.optim.Adam(x_list[1:-1],lr=lr)
        for j in range(liter):
            optimizer1.zero_grad()
            v_list = self.prediction(x_list)
            E = 0
            error_layer =[]
            for m in range(1,self.n_layers):
                E +=((x_list[m]-v_list[m])**2).mean()
                error_layer.append(E.item())
            E.backward()
            optimizer1.step()
            energy_trace.append(E.item())
        return x_list,v_list,E.item(),energy_trace,error_layer
    
    #update weight
    def update_weight(self,x_list,lr=0.01):
        optimizer2 = torch.optim.Adam(self.weights,lr=lr)
        optimizer2.zero_grad()
        #forward
        v_list = self.prediction(x_list) 
        E = 0
        for m in range(1,self.n_layers):
            E +=((x_list[m]-v_list[m])**2).mean()
        E.backward()
        optimizer2.step()
        return E.item()
    #batch training
    def fit(self,train_loader,n_epochs=10,liter=10,lr=0.01):
        energy_history =[]
        for epoch in range(n_epochs):
             epoch_loss = 0
             for batch_x, batch_y in train_loader:
                x_list, v_list, E1,energy_trace,error_layer = self.inference(batch_x, y_batch=batch_y, liter=liter, lr=lr)
                E2 = self.update_weight(x_list, lr=lr) 
                epoch_loss += E2
             average_energy = epoch_loss/len(train_loader)
             energy_history.append(average_energy)
        print(f"Epoch {epoch}, Avg Energy: {average_energy:.4f}")  
        return energy_history         
#use model
model = pcn(dims)
energy_history = model.fit(train_loader, n_epochs=20, liter=5, lr=0.01)
#训练energy曲线
#图一：每一次训练后的energy记录
plt.plot(energy_history)
plt.xlabel('epoch')
plt.ylabel('energy')
plt.title('pcn training energy')
plt.show()

# 测试推理
x_test = torch.randn(batch_size, dims[0])
y_test = torch.ones(batch_size, dims[-1])
x_list, v_list, E,energy_trace,error_layer= model.inference(x_test, y_batch=y_test, liter=10, lr=0.05)
print("Final output after inference:")
print(x_list[-1])
#图二：一次训练中inference的收敛
plt.plot(energy_trace)
plt.xlabel("inference loop")
plt.ylabel("energy")
plt.title("pcn inference convergence")
plt.show()
#图三：记录inference loop结束之后每一层对应的error(主要由于weight尚未update)
plt.bar(range(len(error_layer)),error_layer)
plt.title("layer error from prediction")
plt.show()
