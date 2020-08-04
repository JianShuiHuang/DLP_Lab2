# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:54:42 2020

@author: ivis
"""
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

##initialize variable
lr = 1e-03
BatchSize = 64
Epochs = 150
device = torch.device('cpu')

def read_bci_data():
    S4b_train = np.load('S4b_train.npz')
    X11b_train = np.load('X11b_train.npz')
    S4b_test = np.load('S4b_test.npz')
    X11b_test = np.load('X11b_test.npz')

    train_data = np.concatenate((S4b_train['signal'], X11b_train['signal']), axis=0)
    train_label = np.concatenate((S4b_train['label'], X11b_train['label']), axis=0)
    test_data = np.concatenate((S4b_test['signal'], X11b_test['signal']), axis=0)
    test_label = np.concatenate((S4b_test['label'], X11b_test['label']), axis=0)

    train_label = train_label - 1
    test_label = test_label -1
    train_data = np.transpose(np.expand_dims(train_data, axis=1), (0, 1, 3, 2))
    test_data = np.transpose(np.expand_dims(test_data, axis=1), (0, 1, 3, 2))

    mask = np.where(np.isnan(train_data))
    train_data[mask] = np.nanmean(train_data)

    mask = np.where(np.isnan(test_data))
    test_data[mask] = np.nanmean(test_data)

    print(train_data.shape, train_label.shape, test_data.shape, test_label.shape)

    return train_data, train_label, test_data, test_label


class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        
        ##layer 1
        self.Conv2D_1 = nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias = False)
        self.BatchNorm_1 = nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        ##layer 2
        self.Conv2D_2 = nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False)
        self.BatchNorm_2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.LeakyReLU_2 = nn.LeakyReLU()
        self.AvgPool2D_2 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0)
        self.Dropout_2 = nn.Dropout(p=0.25)
        
        ##layer 3
        self.Conv2D_3 = nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False)
        self.BatchNorm_3 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.LeakyReLU_3 = nn.LeakyReLU()
        self.AvgPool2D_3 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0)
        self.Dropout_3 = nn.Dropout(p=0.25)
        
        ##layer 4
        self.Linear_4 = nn.Linear(in_features=736, out_features=2, bias=True)
        
        
    ##forwarding and backpropagation
    def forward(self, x):
        y = self.Conv2D_1(x)
        y = self.BatchNorm_1(y)
            
        y = self.Conv2D_2(y)
        y = self.BatchNorm_2(y)
        y = self.LeakyReLU_2(y)
        y = self.AvgPool2D_2(y)
        y = self.Dropout_2(y)
            
        y = self.Conv2D_3(y)
        y = self.BatchNorm_3(y)
        y = self.LeakyReLU_3(y)
        y = self.AvgPool2D_3(y)
        y = self.Dropout_3(y)
            
        y = y.view(y.size(0), -1)
        y = self.Linear_4(y)
        #y = torch.softmax(self.Linear_4(y), 1)
		
        return y

class DeepConvNet(nn.Module):
    def __init__(self):
        super(DeepConvNet, self).__init__()
        
        ##layer 1
        self.Conv2D1_1 = nn.Conv2d(1, 25, kernel_size=(1, 5), stride=(1, 1), bias=False)
        self.Conv2D2_1 = nn.Conv2d(25, 25, kernel_size=(2, 1), stride=(1, 1), bias=False)
        self.BatchNorm_1 = nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.LeakyReLU_1 = nn.LeakyReLU()
        self.MaxPool2D_1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0)
        self.Dropout_1 = nn.Dropout(p = 0.5)
        
        ##layer 2
        self.Conv2D_2 = nn.Conv2d(25, 50, kernel_size=(1, 5), stride=(1, 1), bias=False)
        self.BatchNorm_2 = nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.LeakyReLU_2 = nn.LeakyReLU()
        self.MaxPool2D_2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0)
        self.Dropout_2 = nn.Dropout(p = 0.5)
        
        ##layer 3
        self.Conv2D_3 = nn.Conv2d(50, 100, kernel_size=(1, 5), stride=(1, 1), bias=False)
        self.BatchNorm_3 = nn.BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.LeakyReLU_3 = nn.LeakyReLU()
        self.MaxPool2D_3 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0)
        self.Dropout_3 = nn.Dropout(p = 0.5)
        
        ##layer 4
        self.Conv2D_4 = nn.Conv2d(100, 200, kernel_size=(1, 5), stride=(1, 1), bias=False)
        self.BatchNorm_4 = nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.LeakyReLU_4 = nn.LeakyReLU()
        self.MaxPool2D_4 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0)
        self.Dropout_4 = nn.Dropout(p = 0.5)
        
        ##layer 5
        self.Linear_5 = nn.Linear(in_features=8600, out_features=2, bias=True)
        
    def forward(self, x):
        y = self.Conv2D1_1(x)
        y = self.Conv2D2_1(y)
        y = self.BatchNorm_1(y)
        y = self.LeakyReLU_1(y)
        y = self.MaxPool2D_1(y)
        y = self.Dropout_1(y)
            
        y = self.Conv2D_2(y)
        y = self.BatchNorm_2(y)
        y = self.LeakyReLU_2(y)
        y = self.MaxPool2D_2(y)
        y = self.Dropout_2(y)
            
        y = self.Conv2D_3(y)
        y = self.BatchNorm_3(y)
        y = self.LeakyReLU_3(y)
        y = self.MaxPool2D_3(y)
        y = self.Dropout_3(y)
            
        y = self.Conv2D_4(y)
        y = self.BatchNorm_4(y)
        y = self.LeakyReLU_4(y)
        y = self.MaxPool2D_4(y)
        y = self.Dropout_4(y)
        
        y = y.view(y.size(0), -1)
        y = self.Linear_5(y)
		#y = torch.softmax(self.Linear_5(y), 1)

        return y

def Train(train_data, train_label, test_data, test_label, model, optimizer):
    ##training
    true = 0
    false = 0
    Loss = nn.CrossEntropyLoss()   #change loss function here
        
    for j in range(len(train_data)//BatchSize + 1):
        l = j * BatchSize
        r = j * BatchSize + BatchSize
        if(r > len(train_data)):
            r = len(train_data)
            
        x_train = Variable(torch.from_numpy(train_data[l:r]))
        y_train = Variable(torch.from_numpy(train_label[l:r]))
            
        prediction = model(x_train.float())
        
        for k in range(l, r):
            if (prediction[k-l][0] > prediction[k-l][1]) and (train_label[k] == 0):
                true = true + 1
            elif (prediction[k-l][0] > prediction[k-l][1]) and (train_label[k] == 1):
                false = false + 1
            elif (prediction[k-l][0] <= prediction[k-l][1]) and (train_label[k] == 1):
                true = true + 1
            else:
                false = false + 1
            
        loss = Loss(prediction, y_train.long())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    return true / (true + false)

def Test(train_data, train_label, test_data, test_label, model, optimizer):
    ##testing
    true = 0
    false = 0
        
    for j in range(len(test_data)//BatchSize + 1):
        l = j * BatchSize
        r = j * BatchSize + BatchSize
        if(r > len(test_data)):
            r = len(test_data)
        
        x_test = Variable(torch.from_numpy(test_data[l:r]))
            
        prediction = model(x_test.float())
            
        for k in range(l, r):
            if (prediction[k-l][0] > prediction[k-l][1]) and (test_label[k] == 0):
                true = true + 1
            elif (prediction[k-l][0] > prediction[k-l][1]) and (test_label[k] == 1):
                false = false + 1
            elif (prediction[k-l][0] <= prediction[k-l][1]) and (test_label[k] == 1):
                true = true + 1
            else:
                false = false + 1
        
    return true / (true + false)
        
def show_AccuracyCurve(train_accuracy, test_accuracy):
    plt.title('Accuracy(%)', fontsize = 18)
    x = []
    for i in range(Epochs):
        x.append(i)
        plt.plot(i, train_accuracy[i], 'bo')
        plt.plot(i, test_accuracy[i], 'ro')
    
    plt.plot(x, train_accuracy, test_accuracy)
    plt.show()
    
    
        

##main function
train = []
test = []    

train_data, train_label, test_data, test_label = read_bci_data()

model = EEGNet()
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr = lr)
for i in range(Epochs):
    train_accuracy = Train(train_data, train_label, test_data, test_label, model, optimizer)
    train.append(train_accuracy)
    test_accuracy = Test(train_data, train_label, test_data, test_label, model, optimizer)
    test.append(test_accuracy)
    print("epochs:", i )
    print('Train Accuracy: ', train_accuracy)
    print('Test Accuracy: ', test_accuracy)
print('Max accuracy: ', max(test))
show_AccuracyCurve(train, test)

train0 = []
test0 = [] 

model0 = DeepConvNet()
model0 = model0.to(device)
optimizer0 = optim.Adam(model0.parameters(), lr = lr)
for i in range(Epochs):
    train_accuracy0 = Train(train_data, train_label, test_data, test_label, model0, optimizer0)
    train0.append(train_accuracy0)
    test_accuracy0 = Test(train_data, train_label, test_data, test_label, model0, optimizer0)
    test0.append(test_accuracy0)
    print("epochs:", i )
    print('Train Accuracy: ', train_accuracy0)
    print('Test Accuracy: ', test_accuracy0)
print('Max accuracy: ', max(test0))
show_AccuracyCurve(train0, test0)



    
    
    
