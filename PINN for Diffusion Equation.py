# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 17:48:45 2023

@author: shant
"""

import torch
import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker
from sklearn.model_selection import train_test_split

import numpy as np
import time
from pyDOE import lhs         #Latin Hypercube Sampling
import scipy.io

#Set default dtype to float32
torch.set_default_dtype(torch.float)

#PyTorch random number generator
torch.manual_seed(1234)

# Random number generators in other libraries
np.random.seed(1234)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if device == 'cuda': 
    print(torch.cuda.get_device_name()) 

steps = 20000
lr = 1e-3
layers = np.array([2,32,32,1]) # hidden layers
# To generate new data:
x_min = -1
x_max = 1
t_min = 0
t_max = 1
total_points_x = 200
total_points_t = 100
#Nu: Number of training points # Nf: Number of collocation points (Evaluate PDE)
Nu = 100
Nf = 10000


def plot3D(x,t,y):
  x_plot = x.squeeze(1) 
  t_plot = t.squeeze(1)
  X,T = torch.meshgrid(x_plot,t_plot)
  F_xt = y
  fig,ax=plt.subplots(1,1)
  cp = ax.contourf(T,X, F_xt,20,cmap="rainbow")
  fig.colorbar(cp) # Add a colorbar to a plot
  ax.set_title('F(x,t)')
  ax.set_xlabel('t')
  ax.set_ylabel('x')
  plt.show()
  ax = plt.axes(projection='3d')
  ax.plot_surface(T.numpy(), X.numpy(), F_xt.numpy(),cmap="rainbow")
  ax.set_xlabel('t')
  ax.set_ylabel('x')
  ax.set_zlabel('f(x,t)')
  plt.show()
  

def plot3D_Matrix(x,t,y):
  X,T = x,t
  F_xt = y
  fig,ax = plt.subplots(1,1)
  cp = ax.contourf(T,X, F_xt,20,cmap="rainbow")
  fig.colorbar(cp) # Add a colorbar to a plot
  ax.set_title('F(x,t)')
  ax.set_xlabel('t')
  ax.set_ylabel('x')
  plt.show()
  ax = plt.axes(projection='3d')
  ax.plot_surface(T.numpy(), X.numpy(), F_xt.numpy(),cmap="rainbow")
  ax.set_xlabel('t')
  ax.set_ylabel('x')
  ax.set_zlabel('f(x,t)')
  plt.show()

def f_real(x,t):
  return torch.exp(-t)*(torch.sin(np.pi*x))

class FCN(nn.Module):
    
    #Neural Network
      
    def __init__(self,layers):
        super().__init__() 
        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction ='mean')
        'Initialise neural network as a list using nn.Modulelist'  
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)]) 
        self.iter = 0 # This initializes the iter attribute to keep track of the number of iterations performed during optimization (for the optimizer closure).
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0) # In neural network weight initialization, the gain parameter is a scaling factor that controls the magnitude of the initialized weights. It affects the spread of the initial weight values, and it is particularly important in deep neural networks where the network's depth can impact the training dynamics. The Xavier (Glorot) initialization is a popular weight initialization technique that aims to set the initial weights in a way that helps with the forward and backward propagation of signals in the network. The idea behind Xavier initialization is to set the initial weights such that the variance of the activations and gradients remains roughly the same across different layers.
            nn.init.zeros_(self.linears[i].bias.data)   
            
    def forward(self,x):
        if torch.is_tensor(x) != True:         
            x = torch.from_numpy(x)                
        a = x.float()
        for i in range(len(layers)-2):  
            z = self.linears[i](a)              
            a = self.activation(z)    
        a = self.linears[-1](a)
        return a
    
    'Loss Functions'
    
    #Loss Boundary Condition
    def lossBC(self,x_BC,y_BC): # The inputs x_BC and y_BC represent the boundary points and their corresponding true (ground truth) values, respectively.
      loss_BC=self.loss_function(self.forward(x_BC),y_BC) # The self.forward(x_BC) call evaluates the PINN model at the boundary points, obtaining the model predictions for the boundary conditions.
      return loss_BC
  
    #Loss PDE
    def lossPDE(self,x_PDE): #This function computes the loss associated with the partial differential equation (PDE). The input x_PDE represents the collocation points used to evaluate the PDE. The PDE is represented by the function f(x, t) in the code. The function f(x, t) is obtained by evaluating the model at the collocation points x_PDE, and its first and second partial derivatives with respect to t and x, respectively, are computed using the autograd.grad function. 
      g = x_PDE.clone()
      g.requires_grad = True # Enable differentiation
      f = self.forward(g)
      f_x_t = autograd.grad(f, g, torch.ones([g.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0] #first derivative
      f_xx_tt = autograd.grad(f_x_t, g, torch.ones(g.shape).to(device), create_graph=True)[0]#second derivative
      f_t = f_x_t[:,[1]]# we select the 2nd element for t (the first one is x) (Remember the input X=[x,t]) 
      f_xx = f_xx_tt[:,[0]]# we select the 1st element for x (the second one is t) (Remember the input X=[x,t]) 
      f = f_t-f_xx+ torch.exp(-g[:, 1:])* (torch.sin(np.pi * g[:, 0:1]) - np.pi ** 2 * torch.sin(np.pi * g[:, 0:1]))
      return self.loss_function(f,f_hat)

    def loss(self,x_BC,y_BC,x_PDE):
      loss_bc=self.lossBC(x_BC,y_BC)
      loss_pde=self.lossPDE(x_PDE)
      return loss_bc+loss_pde

    #Optimizer              x_train_Nu,y_train_Nu,x_train_Nf                   
    def closure(self):
      optimizer.zero_grad()  
      loss = self.loss(x_train_Nu,y_train_Nu,x_train_Nf) # This line calculates the overall loss of the model using the loss method defined earlier. It computes the combined loss of the boundary condition (x_train_Nu, y_train_Nu) and the partial differential equation (x_train_Nf). The boundary condition loss is the error between the model's predictions at the training points x_train_Nu and their corresponding target values y_train_Nu. The PDE loss is the error between the computed PDE residuals at the collocation points x_train_Nf and the target residuals f_hat.
      loss.backward() # This line performs the backward pass through the computation graph to compute the gradients of the model's parameters with respect to the loss. It calculates the gradients using the chain rule of calculus, which allows efficient computation of gradients in complex computational graphs.
      self.iter += 1
      if self.iter % 100 == 0:
        loss2 = self.lossBC(x_test,y_test) # This line calculates the boundary condition loss using the test data (x_test, y_test). This gives an indication of how well the model is generalizing to unseen data and how well it is satisfying the boundary conditions.
        print("Training Error:",loss.detach().cpu().numpy(),"---Testing Error:",loss2.detach().cpu().numpy())
      return loss  
  
x = torch.linspace(x_min,x_max,total_points_x).view(-1,1) # The view method is used to reshape the tensor. In this case, it is reshaping the tensor x into a 2-dimensional tensor with one column and an unspecified number of rows (denoted by -1). For example, let's say x_min is -1, x_max is 1, and total_points_x is 200. The torch.linspace function will create a 1D tensor x with 200 equally spaced points between -1 and 1. The view(-1,1) operation will reshape it into a 2D tensor with 200 rows and 1 column.
t = torch.linspace(t_min,t_max,total_points_t).view(-1,1)
# Create the mesh 
X,T = torch.meshgrid(x.squeeze(1),t.squeeze(1))
# Evaluate real function
y_real = f_real(X,T)
#plot3D(x,t,y_real) #f_real was defined previously(function)

print(x.shape,t.shape,y_real.shape)
print(X.shape,T.shape)
     
# Transform the mesh into a 2-column vector
x_test = torch.hstack((X.transpose(1,0).flatten()[:,None],T.transpose(1,0).flatten()[:,None])) # x_test is a 2-dimensional tensor of shape (total_points_x * total_points_t, 2), where each row contains the spatial coordinate (x) and temporal coordinate (t) of a data point.
y_test = y_real.transpose(1,0).flatten()[:,None] # is a 2-dimensional tensor of shape (total_points_x * total_points_t, 1), where each row contains the true solution of the PDE for the corresponding data point.

# Domain bounds
lb = x_test[0] #first value
ub = x_test[-1] #last value 
print(x_test.shape,y_test.shape)
print(lb,ub)
     
left_x = torch.hstack((X[:,0][:,None],T[:,0][:,None]))
left_y = torch.sin(np.pi*left_x[:,0]).unsqueeze(1)
bottom_x = torch.hstack((X[-1,:][:,None],T[-1,:][:,None]))
bottom_y = torch.zeros(bottom_x.shape[0], 1)
top_x = torch.hstack((X[0,:][:,None],T[0,:][:,None]))
top_y = torch.zeros(top_x.shape[0], 1)

x_train = torch.vstack([left_x,bottom_x,top_x])
y_train = torch.vstack([left_y,bottom_y,top_y])


idx = np.random.choice(x_train.shape[0],Nu,replace=False)
x_train_Nu = x_train[idx,:]
y_train_Nu = y_train[idx,:]

#collocation points


x_train_Nf = lb + (ub - lb) + lhs(2,Nf)
x_train_Nf = torch.vstack((x_train_Nf,x_train_Nu))

#send data to device

x_train_Nu = x_train_Nu.float().to(device)
y_train_Nu = y_train_Nu.float().to(device)
x_train_Nf = x_train_Nf.float().to(device)
f_hat = torch.zeros(x_train_Nf.shape[0],1).to(device)

torch.manual_seed(123)
#Store tensors to GPU
x_train_Nu = x_train_Nu.float().to(device)#Training Points (BC)
y_train_Nu = y_train_Nu.float().to(device)#Training Points (BC)
x_train_Nf = x_train_Nf.float().to(device)#Collocation Points
f_hat = torch.zeros(x_train_Nf.shape[0],1).to(device)#to minimize function

x_test = x_test.float().to(device) # the input dataset (complete)
y_test = y_test.float().to(device) # the real solution 


#Create Model
PINN = FCN(layers)
PINN.to(device)
print(PINN)
params = list(PINN.parameters())
optimizer = torch.optim.Adam(PINN.parameters(),lr=lr,amsgrad=False)
start_time = time.time()


for i in range(steps):
    if i == 0:
      print("Training Loss  -----  Test Loss")
    loss = PINN.loss(x_train_Nu,y_train_Nu,x_train_Nf)# use mean squared error
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % (steps/10) == 0:
      with torch.no_grad():
        test_loss=PINN.lossBC(x_test,y_test)
      print(loss.detach().cpu().numpy(),'  ---  ',test_loss.detach().cpu().numpy())
      
y1 = PINN(x_test)
x1 = x_test[:,0]
t1 = x_test[:,1]
arr_x1 = x1.reshape(shape=[100,200]).transpose(1,0).detach().cpu()
arr_T1 = t1.reshape(shape=[100,200]).transpose(1,0).detach().cpu()
arr_y1 = y1.reshape(shape=[100,200]).transpose(1,0).detach().cpu()
arr_y_test = y_test.reshape(shape=[100,200]).transpose(1,0).detach().cpu()

plot3D_Matrix(arr_x1,arr_T1,arr_y1)
plot3D_Matrix(X,T,y_real)