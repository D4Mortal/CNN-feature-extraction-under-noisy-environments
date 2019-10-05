# Implementation of a Convolutional neural network
#
# program written by Daniel Hao, May 2019


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as D
import matplotlib.pyplot as plt


# Define a convolutional network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv1_drop = nn.Dropout2d(0)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv2_drop = nn.Dropout2d(0.2)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32,64, kernel_size=3)
        self.conv3_drop = nn.Dropout2d(0)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv4_drop = nn.Dropout2d(0.2)
        self.bn4 = nn.BatchNorm2d(64)     
    
        self.fc1 = nn.Linear(64*87*69, 200)
        self.fc2 = nn.Linear(200, 7)


    def forward(self, x):
        
        x = F.leaky_relu(self.conv1(x))
        x = self.conv1_drop(x)
        x = self.bn1(x)
        
        x = F.leaky_relu(self.conv2(x))
        x = F.max_pool2d(self.conv2_drop(x), 2)
        x = self.bn2(x)
        
        x = F.leaky_relu(self.conv3(x))
        x = self.conv3_drop(x)
        x = self.bn3(x)
        
        x = F.leaky_relu(self.conv4(x))
        x = F.max_pool2d(self.conv4_drop(x), 2)
        x = self.bn4(x)
        
#        print(x.shape)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return x
    

# calculate balanced accuracy of the network given a confusion matrix
def balanced_accuracy(confusion, dimension):
    balanced_accuracy = 0
    for i in range(dimension):
        total = 0
        for j in range(dimension):
            total += confusion[i][j]
        
        print('{} accuracy: {:.0f}%'.format(i, 100. * confusion[i][i] / total))
        
        balanced_accuracy += confusion[i][i] / total
    balanced_accuracy = balanced_accuracy / dimension
    return balanced_accuracy

# train the neural network for a single epoch and plot thee loss graph
def train(epoch, train_loader, model, train_losses, train_accuracy):
    
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.5, 
                          weight_decay=0.1)
    
    loss_func = torch.nn.CrossEntropyLoss()
    
    model.train()
    correct = 0
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
            
        optimizer.zero_grad()
        output = model(data)
        
        loss = loss_func(output, target)
        train_loss += loss.item()
        train_losses.append(loss.item())
        
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 4 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader)
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
                  .format(train_loss, correct, len(train_loader.dataset),
                          100. * correct / len(train_loader.dataset)))
    
    train_accuracy.append(100. * correct / len(train_loader.dataset))
    
    plt.figure()
    plt.plot(train_losses)
    plt.legend(['train loss'], loc='upper left')
    plt.show()

# test the neural network
def test(test_loader, model, test_accuracy, train_accuracy):
    predictions = [0,0,0,0,0,0,0]
    confusion_matrix = np.zeros(shape=(7,7))
    
    model.eval()
    loss_func = torch.nn.CrossEntropyLoss()
    
    test_loss = 0
    correct = 0
    
    for data, target in test_loader:
        
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
            
        output = model(data)
        test_loss += loss_func(output, target).item() 
        
        pred = output.data.max(1, keepdim=True)[1] 
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        
        predictions[pred] += 1 
        confusion_matrix[target][pred] += 1

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          .format(test_loss, correct, len(test_loader.dataset),
                  100. * correct / len(test_loader.dataset)))
    
    print(predictions)
    print('')
    print('Confusion matrix for testing:')
    print(confusion_matrix)
    print('Balanced Accuracy: {:.0f}%'
          .format(100. * balanced_accuracy(confusion_matrix, 7)))
    
    plt.bar(["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"], 
            height = predictions) 
    
    test_accuracy.append(100. * correct / len(test_loader.dataset))
    
    plt.figure()
    plt.plot(test_accuracy)
    plt.plot(train_accuracy)
    plt.legend(['test accuracy', 'train accuracy'], loc='upper left')
    plt.show()
    
    return  100 * correct / len(test_loader.dataset)


# calcuate and return the split sizes for k fold cross validation
def k_fold_split_size(k, dataset):
    remainder = len(dataset) % k
    split_size = (len(dataset) - remainder) / k
    splits = []
    for i in range(k - 1):
        splits.append(int(split_size))

    splits.append(int(split_size + remainder))
    
    return splits

# train the network over the defined number of epochs, test and graph a
# accuracy graph after each epoch
def train_test(epochs, train_data, test_data):
    model = Net()
    if torch.cuda.is_available():
        model.cuda()
        
    train_losses = []
    test_accuracy = []
    train_accuracy = []
    
    for epoch in range(1, epochs + 1):
        train(epoch,train_data, model, train_losses, train_accuracy)
        result = test(test_data, model, test_accuracy, train_accuracy)
    return result
 
# Apply K fold cross validation, returns the average results over k runs
def k_fold_crossvalidation(k, dataset, epochs):
    splits = k_fold_split_size(k, dataset)

    data_split = D.random_split(dataset, splits)
    total_accuracy = 0
    
    for i in range(k):
        print('\n\n Cross validation run number: {}\n\n'.format(i + 1))
        test_data = data_split[i]
        
        if i == 0:
            train_data = D.ConcatDataset(data_split[1:])
            
        elif i == k-1:
            train_data = D.ConcatDataset(data_split[:k-1])

        else:
            train_data1 = D.ConcatDataset(data_split[:i])
            train_data2 = D.ConcatDataset(data_split[i + 1:])
            train_data = D.ConcatDataset([train_data1, train_data2])
            
        train_loader = torch.utils.data.DataLoader(
                train_data,
                batch_size=16,
                shuffle=True,
                pin_memory=True
            )


        test_loader = torch.utils.data.DataLoader(
                test_data,
                batch_size=1,
                shuffle=True,
                pin_memory=True
            )
        
        total_accuracy += train_test(epochs, train_loader, test_loader)
    average = total_accuracy / k
    print('Average accuracy from {} fold cross validation with {} epochs is {}%'
                                                  .format(k, epochs, average))
    return average