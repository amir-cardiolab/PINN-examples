import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math


#Modified from:
#https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_linear_regression/


# nonlinear fn examples:
#https://medium.com/@benjamin.phillips22/simple-regression-with-neural-networks-in-pytorch-313f06910379

'''
STEP 1: CREATE MODEL CLASS
'''
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)  

    def forward(self, x):
        out = self.linear(x)
        return out

class NonLinearRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim,output_dim):
        super(NonLinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)  
        self.relu1 = nn.ReLU()
        #self.relu1 = nn.LeakyReLU()
        #self.relu1 = nn.Sigmoid()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim) 
        self.relu2 = nn.ReLU()
        #self.relu2 = nn.Sigmoid()
        #self.relu2 = nn.LeakyReLU()
        self.linear3 = nn.Linear(hidden_dim, output_dim) 

    def forward(self, x):
        out = self.linear(x)
        out = self.relu1(out)
        out = self.linear2(out) 
        out = self.relu2(out)
        out = self.linear3(out)

        return out


'''
STEP 2: INSTANTIATE MODEL CLASS
'''
input_dim = 1
output_dim = 1
hidden_dim = 50    # 10 works for simple fns. For high freq sin need more here and also more layers

Flag_noise = True #add noise in training data

model = NonLinearRegressionModel(input_dim, hidden_dim,output_dim)
#model = LinearRegressionModel(input_dim, output_dim)


#######################
#  USE GPU FOR MODEL  #
#######################

device = torch.device("cpu")# torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

'''
STEP 3: INSTANTIATE LOSS CLASS
'''

criterion = nn.MSELoss()

'''
STEP 4: INSTANTIATE OPTIMIZER CLASS
'''

learning_rate = 0.01 

#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) #SGD diverges easier than Adam!
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  #default is 0.001 for Adam

Flag_normalize = True #Normalize data for better performance !!!!!! MAkes a lot of difference!!!!!!!!!!!!

#Define the Data
N = 70
#x_values = [i for i in range(int(-N/2),int(N/2))]
#x_train = np.array(x_values, dtype=np.float32)
x_train = np.linspace(-5., 5., num=N,dtype=np.float32)
x_train1 = x_train
if (Flag_normalize):
    x_train = x_train/ np.max(abs(x_train)) 



y_values = []
for i in range(N):
    #result = 2* x_train1[i] + 1 + 3*(x_train1[i]**2) 
    #result = (x_train1[i]**2)  # similar to above web example
    #result = math.sin(x_train1[i])
    result = math.sin(3.*x_train1[i]) # increasing the factor (freq) makes training harder
    y_values.append(result) 


y_train1 = np.array(y_values, dtype=np.float32)
if (Flag_normalize):
        y_train1 = y_train1 / np.max(abs(y_train1)) 
 


x_train = x_train.reshape(-1, 1) #If you don't this you will get an error stating you need 2D. Simply just reshape accordingly if you ever face such errors down the road.        
y_train1 = y_train1.reshape(-1, 1)

if(Flag_noise):# Add noise
 sigma = 0.025 * np.max(abs(y_train1))  
 mu = 0.
 noise = sigma * np.random.randn(N,1) + mu
 y_train = y_train1 + noise
 y_train = y_train.astype('float32') 
 #print (y_train)


#print (x_train)
#print (y_train)


'''
STEP 5: TRAIN THE MODEL
'''
epochs = 500
for epoch in range(epochs):
    epoch += 1
    # Convert numpy array to torch Variable

    #######################
    #  USE GPU FOR MODEL  #
    #######################
    inputs = torch.from_numpy(x_train).to(device)
    labels = torch.from_numpy(y_train).to(device)

    # Clear gradients w.r.t. parameters
    optimizer.zero_grad() 

    # Forward to get output
    outputs = model(inputs)

    # Calculate Loss
    loss = criterion(outputs, labels)

    # Getting gradients w.r.t. parameters
    loss.backward()

    # Updating parameters
    optimizer.step()

    # Logging
    print('epoch {}, loss {}'.format(epoch, (loss.item())**0.5 ))



if(1): #plot the results

 plt.clf()

 # Get predictions
 predicted = model(torch.from_numpy(x_train).requires_grad_()).data.numpy()

 # Plot true data
 plt.plot(x_train, y_train, 'go', label='True data', alpha=0.5)

 # Plot predictions
 plt.plot(x_train, predicted, '--', label='Predictions', alpha=0.5)

 # Legend and plot
 plt.legend(loc='best')
 plt.show()



