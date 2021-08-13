import torch
import numpy as np
#import foamFileOperation
from matplotlib import pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
#from torchvision import datasets, transforms
import csv
from torch.utils.data import DataLoader, TensorDataset,RandomSampler
from math import exp, sqrt,pi
import time

#Solve 2D Diffusion eqn with source term :  Diff * Laplacian(C) = f_source

def geo_train(device,x_in,y_in,xb,yb,cb,xb_Neumann,yb_Neumann,batchsize,learning_rate,epochs,path,Flag_batch,f_source,Diff,Flag_BC_exact,Lambda_BC  ):
	if (Flag_batch):
	 x = torch.Tensor(x_in).to(device)
	 y = torch.Tensor(y_in).to(device) 
	 #xb = torch.Tensor(xb).to(device) #These are different size as x and cannot go into the same TensorDataset
	 #yb = torch.Tensor(yb).to(device)
	 #cb = torch.Tensor(cb).to(device)  	
	 dataset = TensorDataset(x,y)
	 dataloader = DataLoader(dataset, batch_size=batchsize,shuffle=True,num_workers = 0,drop_last = True )
	else:
	 x = torch.Tensor(x_in).to(device)
	 y = torch.Tensor(y_in).to(device)    
	h_nD = 30
	h_n = 10  #20
	input_n = 2 # this is what our answer is a function of. In the original example 3 : x,y,scale 
	class Swish(nn.Module):
		def __init__(self, inplace=True):
			super(Swish, self).__init__()
			self.inplace = inplace

		def forward(self, x):
			if self.inplace:
				x.mul_(torch.sigmoid(x))
				return x
			else:
				return x * torch.sigmoid(x)
	class Net1(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(Net1, self).__init__()
			self.main = nn.Sequential(
				nn.Linear(2,h_nD),
				nn.Tanh(),
				nn.Linear(h_nD,h_nD),
				nn.Tanh(),
				nn.Linear(h_nD,h_nD),
				nn.Tanh(),

				nn.Linear(h_nD,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			return  output
	class Net2(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(Net2, self).__init__()
			self.main = nn.Sequential(
				nn.Linear(input_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			if (Flag_BC_exact):
				output = output*(x- xStart) *(x- xEnd )*(y- yStart) *(y- yEnd ) + (-0.9*x + 1.) #modify output to satisfy BC automatically #PINN-transfer-learning-BC-20
			return  output

	
	################################################################
	############################################################################
	#### Initiate the network used to represent concentration ##############
	#net1 = Net1().to(device)
	net2 = Net2().to(device)
	
	def init_normal(m):
		if type(m) == nn.Linear:
			nn.init.kaiming_normal_(m.weight)

	# use the modules apply function to recursively apply the initialization
	net2.apply(init_normal)



	############################################################################

	optimizer2 = optim.Adam(net2.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	#optimizer3	= optim.Adam(net3.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
	#optimizer4	= optim.Adam(net4.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)


	############################################################################
	#### Define the governing equation ##############
	def criterion(x,y):

		#print (x)
		x = torch.Tensor(x).to(device)
		y = torch.Tensor(y).to(device)
		#x = torch.FloatTensor(x).to(device)
		#x= torch.from_numpy(x).to(device)

		x.requires_grad = True
		y.requires_grad = True
		
		#net_in = torch.cat((x),1)
		net_in = torch.cat((x,y),1)
		C = net2(net_in)
		C = C.view(len(C),-1)

		
		c_x = torch.autograd.grad(C,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		c_xx = torch.autograd.grad(c_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		c_y = torch.autograd.grad(C,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		c_yy = torch.autograd.grad(c_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		
		loss_1 = f_source - Diff * (c_xx + c_yy)




		# MSE LOSS
		loss_f = nn.MSELoss()

		#Note our target is zero. It is residual so we use zeros_like
		loss = loss_f(loss_1,torch.zeros_like(loss_1)) 

		return loss

	###################################################################
	############################################################################
	#### Define the boundary conditions ##############
	def Loss_BC(xb,yb,xb_Neumann,yb_Neumann ,cb):
		xb = torch.FloatTensor(xb).to(device)
		yb = torch.FloatTensor(yb).to(device)
		cb = torch.FloatTensor(cb).to(device)

		xb_Neumann = torch.FloatTensor(xb_Neumann).to(device)
		yb_Neumann = torch.FloatTensor(yb_Neumann).to(device)

		xb_Neumann.requires_grad = True
		yb_Neumann.requires_grad = True

		net_in = torch.cat((xb, yb), 1)
		out = net2(net_in )


		c_bc = out.view(len(out), -1)

		loss_f = nn.MSELoss()
		loss_bc_Dirichlet = loss_f(c_bc, cb)  #The Dirichlet BC (left and right)


		net_in2 = torch.cat((xb_Neumann, yb_Neumann), 1)
		out_n = net2(net_in2 )
		c_n = out_n.view(len(out_n), -1)

		c_y = torch.autograd.grad(c_n,yb_Neumann,grad_outputs=torch.ones_like(yb_Neumann),create_graph = True,only_inputs=True)[0]

		loss_bc_Neumann = loss_f(c_y, torch.zeros_like(c_y))  #The zeron Neumann BC (top and bottom)

		loss_bc = loss_bc_Dirichlet + loss_bc_Neumann

		return loss_bc


	
	LOSS = []
	tic = time.time()


	############################################################################
	####  Main loop##############
	if(Flag_batch):# This one uses dataloader
		for epoch in range(epochs):
			for batch_idx, (x_in,y_in) in enumerate(dataloader):
				#zero gradient
				#net1.zero_grad()
				##Closure function for LBFGS loop:
				#def closure():
				net2.zero_grad()
				loss_eqn = criterion(x_in,y_in)
				loss_bc = Loss_BC(xb,yb,xb_Neumann,yb_Neumann ,cb)
				loss = loss_eqn + Lambda_BC* loss_bc
				loss.backward()
				#return loss
				#loss = closure()
				#optimizer2.step(closure)
				#optimizer3.step(closure)
				#optimizer4.step(closure)
				optimizer2.step() 
				if batch_idx % 20 ==0:
					print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}\t Loss BC {:.6f}'.format(
						epoch, batch_idx * len(x_in), len(dataloader.dataset),
						100. * batch_idx / len(dataloader), loss.item(),loss_bc.item()))
					LOSS.append(loss.item())
				#if epoch %100 == 0:
				#	torch.save(net2.state_dict(),path+"geo_para_axisy_sigma"+str(sigma)+"_epoch"+str(epoch)+"hard_u.pt")
		#Test with all data
		loss_eqn = criterion(x,y)	
		loss_bc = Loss_BC(xb,yb,xb_Neumann,yb_Neumann,cb)
		loss = loss_eqn + Lambda_BC* loss_bc
		print('**** Final (all batches) \tLoss: {:.10f} \t Loss BC {:.6f}'.format(
					loss.item(),loss_bc.item()))

	else:
		for epoch in range(epochs):
			#zero gradient
			#net1.zero_grad()
			##Closure function for LBFGS loop:
			#def closure():
			net2.zero_grad()
			loss_eqn = criterion(x,y)
			loss_bc = Loss_BC(xb,yb,xb_Neumann,yb_Neumann ,cb)
			if (Flag_BC_exact):
				loss = loss_eqn #+ loss_bc
			else:
				loss = loss_eqn + Lambda_BC * loss_bc
			loss.backward()
			#return loss
			#loss = closure()
			#optimizer2.step(closure)
			#optimizer3.step(closure)
			#optimizer4.step(closure)
			optimizer2.step() 
			if epoch % 10 ==0:
				print('Train Epoch: {} \tLoss: {:.10f} \t Loss BC {:.6f}'.format(
					epoch, loss.item(),loss_bc.item()))
				LOSS.append(loss.item())

	toc = time.time()
	elapseTime = toc - tic
	print ("elapse time in parallel = ", elapseTime)
	###################
	#plot
	net_in = torch.cat((x,y),1)
	output = net2(net_in)  #evaluate model
	C_Result = output.data.numpy()

	plt.figure()
	plt.subplot(2, 1, 1)
	plt.scatter(x.detach().numpy(), y.detach().numpy(), c = C_Result, cmap = 'coolwarm')
	plt.title('PINN results')
	plt.colorbar()
	plt.show()

	return 

	############################################################
	##save loss
	##myFile = open('Loss track'+'stenosis_para'+'.csv','w')#
	##with myFile:
		#writer = csv.writer(myFile)
		#writer.writerows(LOSS)
	#LOSS = np.array(LOSS)
	#np.savetxt('Loss_track_pipe_para.csv',LOSS)

	############################################################

	#save network
	##torch.save(net1.state_dict(),"stenosis_para_axisy_sigma"+str(sigma)+"scale"+str(scale)+"_epoch"+str(epochs)+"boundary.pt")
	#torch.save(net2.state_dict(),path+"geo_para_axisy"+"_epoch"+str(epochs)+"c.pt")
	##torch.save(net3.state_dict(),path+"geo_para_axisy_sigma"+str(sigma)+"_epoch"+str(epochs)+"hard_v.pt")
	##torch.save(net4.state_dict(),path+"geo_para_axisy_sigma"+str(sigma)+"_epoch"+str(epochs)+"hard_P.pt")
	#####################################################################


#######################################################



############################################################################
####  Define parameters ##############

device = torch.device("cpu")


Flag_batch = True #Use batch
Flag_Chebyshev = False #Use Chebyshev pts for more accurcy in BL region; Not implemented in 2D
Flag_BC_exact = False #Not implemented yet in 2D
Lambda_BC  = 10. # the weight used in enforcing the BC


batchsize = 128  #Total number of batches 
learning_rate = 1e-2 

if (not Flag_batch):
	epochs  = 2000
else:
	epochs = 25

f_source = 0 #-1.0
Diff = 0.1

nPt = 100 
xStart = 0.
xEnd = 1.
yStart = 0.
yEnd = 0.5

if(Flag_Chebyshev): #!!!Not a very good idea (makes even the simpler case worse)
 x = np.polynomial.chebyshev.chebpts1(2*nPt)
 x = x[nPt:]
 if(0):#Mannually place more pts at the BL 
    x = np.linspace(0.95, xEnd, nPt)
    x[1] = 0.2
    x[2] = 0.5
 x[0] = 0.
 x[-1] = xEnd
 x = np.reshape(x, (nPt,1))
else:
 x = np.linspace(xStart, xEnd, nPt)
 y = np.linspace(yStart, yEnd, nPt)
 x, y = np.meshgrid(x, y)
 x = np.reshape(x, (np.size(x[:]),1))
 y = np.reshape(y, (np.size(y[:]),1))

print('shape of x',x.shape)
print('shape of y',y.shape)



############################################################################
####  Defnine the Dirichlet BCs	##############
C_BC1 = 0.2  #Left Dirichlet BC
C_BC2 = 1. #Right Dirichlet BC
xleft = np.linspace(xStart, xStart, nPt)
xright = np.linspace(xEnd, xEnd, nPt)
xup = np.linspace(xStart, xEnd, nPt)
xdown = np.linspace(xStart, xEnd, nPt)
yleft = np.linspace(yStart, yEnd, nPt)
yright = np.linspace(yStart, yEnd, nPt)
yup = np.linspace(yEnd, yEnd, nPt)
ydown = np.linspace(yStart, yStart, nPt)
cleft = np.linspace(C_BC1, C_BC1, nPt)
cright = np.linspace(C_BC2, C_BC2, nPt)
cup = np.linspace(C_BC2, C_BC2, nPt)
cdown = np.linspace(C_BC1, C_BC1, nPt)

if(0): #Dirichlet BC everywhere
 xb = np.concatenate((xleft, xright, xup, xdown), 0)
 yb = np.concatenate((yleft, yright, yup, ydown), 0)
 cb = np.concatenate((cleft, cright, cup, cdown), 0)
else: #Only Dirichlet BC left and right 
 xb = np.concatenate((xleft, xright), 0)
 yb = np.concatenate((yleft, yright), 0)
 cb = np.concatenate((cleft, cright), 0)

##Define zero Neumann BC location
xb_Neumann = np.concatenate((xup, xdown), 0) 	
yb_Neumann = np.concatenate((yup, ydown), 0) 

xb= xb.reshape(-1, 1) #need to reshape to get 2D array
yb= yb.reshape(-1, 1) #need to reshape to get 2D array
cb= cb.reshape(-1, 1) #need to reshape to get 2D array
xb_Neumann = xb_Neumann.reshape(-1, 1) #need to reshape to get 2D array
yb_Neumann = yb_Neumann.reshape(-1, 1) #need to reshape to get 2D array



path = "Results/"

#Analytical soln
#A = (C_BC2 - C_BC1) / (exp(Vel/Diff) - 1)
#B = C_BC1 - A
#C_analytical = A*np.exp(Vel/Diff*x[:] ) + B



#path = pre+"aneurysmsigma01scalepara_100pt-tmp_"+str(ii)
geo_train(device,x,y,xb,yb,cb,xb_Neumann,yb_Neumann, batchsize,learning_rate,epochs,path,Flag_batch,f_source,Diff,Flag_BC_exact,Lambda_BC  )
#tic = time.time()

#elapseTime = toc - tic
#print ("elapse time in serial = ", elapseTime)

 








