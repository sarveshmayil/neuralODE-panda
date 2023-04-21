import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os
from torchdiffeq import odeint
from tqdm.notebook import tqdm


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None, act=nn.ReLU):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), act()]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), act()]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ODEfunc(nn.Module):
    def __init__(self, state_dim:int, action_dim:int, hidden_dim:int=100, hidden_depth:int=2) -> None:
        super().__init__()
        self.model = mlp(input_dim=state_dim+action_dim,
                         output_dim=state_dim+action_dim,
                         hidden_dim=hidden_dim,
                         hidden_depth=hidden_depth)

    def forward(self, t, x):
        return self.model(x)


class ODEblock(nn.Module):
    def __init__(self, odefunc, n_steps:int, atol:float=1e-8, rtol:float=1e-8, solver:str='dopri5') -> None:
        """
        :param odefunc: ODE function
        :param n_steps: <int> number of steps for ODE integration
        :param atol: <float> absolute tolerance
        :param rtol: <float> relative tolerance
        :param solver: <str> solver type
                Adaptive: 'dopri8', 'dopri5', 'bosh3', 'fehlberg2', 'adaptive_heun'
                Fixed   : 'euler', 'midpoint', 'rk4', 'explicit_adams', 'implicit_adams'
        """
        super().__init__()
        assert n_steps >= 2

        self.odefunc = odefunc
        self.atol = atol
        self.rtol = rtol
        self.solver = solver
        self.integration_time = torch.linspace(0.0, 1.0, n_steps)

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(
            self.odefunc,
            x,
            self.integration_time,
            atol=self.atol, rtol=self.rtol, method=self.solver
        )
        return out[-1]


class Absolute_ODEnet(nn.Module):
    def __init__(self, state_dim:int, action_dim:int,
                 n_steps:int, solver:str='dopri5',
                 hidden_dim:int=100, hidden_depth:int=2,
                 ) -> None:
        super().__init__()
        self.state_dim = state_dim
        odefunc = ODEfunc(state_dim, action_dim, hidden_dim=hidden_dim, hidden_depth=hidden_depth)
        self.ode_block = ODEblock(odefunc, n_steps, solver=solver)

    def forward(self, state, action):
        inp = torch.cat((state, action), -1)
        out = self.ode_block(inp)
        out = out[:,:self.state_dim]
        return out
        
        
class Residual_ODEnet(nn.Module):
    def __init__(self, state_dim:int, action_dim:int, 
                 n_steps:int, solver:str='dopri5',
                 hidden_dim:int=100, hidden_depth:int=2) -> None:
        super().__init__()
        self.state_dim = state_dim
        odefunc = ODEfunc(state_dim, action_dim, hidden_dim=hidden_dim, hidden_depth=hidden_depth)
        self.ode_block = ODEblock(odefunc, n_steps, solver=solver)

    def forward(self, state, action):
        inp = torch.cat((state, action), -1)
        out = self.ode_block(inp)
        out = state + out[:,:self.state_dim]
        return out
        

def train_nODE_model(model, train_dataloader, val_dataloader, loss_fn, num_epochs=100, lr=1e-3):
    """
    Trains the given model for `num_epochs` epochs. Use SGD as an optimizer.
    You may need to use `train_step` and `val_step`.
    :param model: Pytorch nn.Module.
    :param train_dataloader: Pytorch DataLoader with the training data.
    :param val_dataloader: Pytorch DataLoader with the validation data.
    :param num_epochs: <int> number of epochs to train the model.
    :param lr: <float> learning rate for the weight update.
    :return:
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pbar = tqdm(range(num_epochs))
    train_losses = []
    val_losses = []
    for epoch_i in pbar:
        train_loss_i = train_step(model, train_dataloader, optimizer, loss_fn)
        val_loss_i = val_step(model, val_dataloader, loss_fn)
        pbar.set_description(f'Train Loss: {train_loss_i:.4f} | Val Loss: {val_loss_i:.4f}')
        train_losses.append(train_loss_i)
        val_losses.append(val_loss_i)
    return train_losses, val_losses


def train_step(model, train_loader, optimizer, loss_fn) -> float:
    """
    Performs an epoch train step.
    :param model: Pytorch nn.Module
    :param train_loader: Pytorch DataLoader
    :param optimizer: Pytorch optimizer
    :return: train_loss <float> representing the average loss among the different mini-batches.
    """
    train_loss = 0.0
    model.train() 
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        loss = loss_fn(model, data['state'], data['action'], data['next_state'])
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss/len(train_loader)

def val_step(model, val_loader, loss_fn) -> float:
    """
    Perfoms an epoch of model performance validation
    :param model: Pytorch nn.Module
    :param train_loader: Pytorch DataLoader
    :param optimizer: Pytorch optimizer
    :return: val_loss <float> representing the average loss among the different mini-batches
    """
    val_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            loss = loss_fn(model, data['state'], data['action'], data['next_state'])
            val_loss += loss.item()
    return val_loss/len(val_loader)
    
    
def neuralODE_gridsearch(train_loader, val_loader,
                         loss_fn, save_dir:str,
                         state_dim:int, action_dim:int,
                         residual:bool=False, device:str='cpu',
                         num_epochs:int=100, lr:float=1e-3):
    if residual:
        ODEnet = lambda n, solver, hidden_dim: Residual_ODEnet(state_dim, action_dim, n, solver, hidden_dim) 
    else:
        ODEnet = lambda n, solver, hidden_dim: Absolute_ODEnet(state_dim, action_dim, n, solver, hidden_dim) 
    
    loss_fn = loss_fn.to(device)
    
    solvers = ['dopri5', 'rk4']
    n_steps = [2, 6, 10]
    hidden_dims = [4, 32, 100]
    # solvers = ['rk4']
    # n_steps = [2, 4, 6, 8, 10]
    # hidden_dims = [4, 16, 32, 64, 100]
    
    result_dict = {}
    for solver in solvers:
        result_dict[solver] = np.zeros((len(n_steps), len(hidden_dims), 2))

    for solver in solvers:
        for i, n in enumerate(n_steps):
            for j, hidden_dim in enumerate(hidden_dims):
                print("n_steps=%d, solver=%s, hidden_dim=%d" % (n, solver, hidden_dim))
                model = ODEnet(n, solver, hidden_dim).to(device)
                train_losses, val_losses = train_nODE_model(model, 
                                                            train_loader, 
                                                            val_loader, 
                                                            loss_fn, 
                                                            num_epochs=num_epochs, 
                                                            lr=lr)
                                                            
                print("\ttrain_loss=%.3E, val_loss=%.3E" % (train_losses[-1], val_losses[-1]))
                result_dict[solver][i, j, 0] = train_losses[-1]
                result_dict[solver][i, j, 1] = val_losses[-1]

                save_path = os.path.join(save_dir, 'models/nODE_'+str(n)+'_'+solver+'_'+str(hidden_dim)+'.pt')
                torch.save(model.state_dict(), save_path)

    return result_dict, (solvers, n_steps, hidden_dims)
    