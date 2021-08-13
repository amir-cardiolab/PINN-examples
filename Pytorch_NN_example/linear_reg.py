import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


#Modified from:
#https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_linear_regression/

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
        self.linear2 = nn.Linear(hidden_dim, output_dim) 

    def forward(self, x):
        out = self.linear(x)
        out = self.relu1(out)
        out = self.linear2(out)

        return out


'''
STEP 2: INSTANTIATE MODEL CLASS
'''
input_dim = 1
output_dim = 1
hidden_dim = 1

Flag_noise = True #add noise in training data

#model = NonLinearRegressionModel(input_dim, hidden_dim,output_dim)
model = LinearRegressionModel(input_dim, output_dim)


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

learning_rate = 0.01 #using 0.02 gives worse results! (higher MSE)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


#Define the Data
N=11
x_values = [i for i in range(N)]
x_train = np.array(x_values, dtype=np.float32)

x_train = x_train.reshape(-1, 1) #If you don't this you will get an error stating you need 2D. Simply just reshape accordingly if you ever face such errors down the road.





y_values = []
for i in x_values:
    result = (2*i + 1) / (N-1)
    y_values.append(result) 

y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)

if(Flag_noise):# Add noise
 sigma = 0.1 #2. 
 mu = 0.
 noise = sigma * np.random.randn(N,1) + mu
 y_train = y_train + noise
 y_train = y_train.astype('float32') 
 #print (y_train)

'''
STEP 5: TRAIN THE MODEL
'''
epochs = 100
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
    print('epoch {}, loss {}'.format(epoch, loss.item()))



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



