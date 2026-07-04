class epmodel(nn.Module):
    def __init__(self,dims,beta=1e-3,seed=0):
        super(epmodel,self).__init__()
        torch.manual_seed(seed)
        self.dims = dims
        self.layer = len(dims)
        self.beta = beta
        self.weight = nn.ParameterList([
            nn.Parameter(torch.randn(self.dims[i],self.dims[i+1])*0.1)
            for i in range(self.layer-1)
        ])
    def forward(self,x_list):
        v_list =[]
        for i in range(self.layer-1):
            v = x_list[i]@self.weight[i]
            v_list.append(v)
        return v_list
    def init_states(self,x):
        x_list = []
        for i in range(1,self.layer):
            xi = torch.ones(x.size(0),self.dims[i])
            xi.requires_grad_(True)
            x_list.append(xi)
        return x_list
    def energy(self,x_list):
        v_list = self.forward(x_list)
        E = 0
        for i in range(self.layer-1):
            E += ((x_list[i+1] - v_list[i])**2).mean()
        return v_list,E
    def relax(self,x_list,target=None,beta=0.0,liter=20,lr=0.1):
        for _ in range(liter):
            v_list,E = self.energy(x_list)
            if target is not None and beta !=0.0:
                E += beta*((v_list[-1]-target)**2).mean()
            grad = torch.autograd.grad(E,x_list[1:],create_graph=False)            
        #autograd.grad()直接返回梯度，不存；backward()会把梯度存到.grad里
        #create_graph=False:不需要对梯度再求梯度
            with torch.no_grad():
                for i in range(1,self.layer):
                    x_list[i] -= lr*grad[i-1]
                    x_list[i].requires_grad_(True)
        return x_list,E
    def update_weight(self,x_free,x_clamp,target,lr):
        _,E1= self.relax(x_free)
        grad_free = torch.autograd.grad(E1,self.weight,retain_graph=True)
        #注意这里如果两个梯度如果共用了计算图，需要保留下计算图
        _,E2 = self.relax(x_clamp,target,beta=self.beta)
        grad_clamp = torch.autograd.grad(E2,self.weight)
        with torch.no_grad(): #这一步不被autograd记录
            #手动更新梯度？
            #ep.grad 不是单个loss的梯度
            #optimizer可，backward不可
            weight_up = self.weight
            for w,gc,gf in zip(weight_up,grad_clamp,grad_free):
                w -= lr*(gc-gf)/self.beta
        return weight_up

    def train_combine(self,target,x,lr):
        x_list =self.init_states(x)
        x_free,_ = self.relax([s.clone().detach().requires_grad_(True) for s in x_list])
        x_clamp,_ = self.relax([s.clone().detach().requires_grad_(True) for s in x_free],target=target,beta=self.beta)
        w_update = self.update_weight(x_free,x_clamp,target,lr=lr)
        return w_update
    def fit(self,loader,epoch,lr):
        for ep in range(epoch):
            epoch_loss = 0
            for x,y in(loader):
                xlist = self.init_states(x=x)
                weight = self.train_combine(target=y,x=x,lr=lr)
                vlist =[]
                for i in range(self.layer-1):
                    vi = xlist[i]@weight[i]
                    vlist.append(vi)
            for p in range(self.layer-1):
                epoch_loss += ((xlist[p+1] - vlist[p])**2).mean()
            avr_energy =(epoch_loss.item()/len(loader))
            print(f"epocn:{ep},average_energy:{avr_energy}")
