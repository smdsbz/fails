#!/usr/bin/env python3

# Data representation utilities
import numpy as np
import csv
from torch.utils.data import Dataset, DataLoader

# Model learning utilities
import torch
from torch.autograd import Variable

# Visualization
import matplotlib.pyplot as plt

input_filename = "./trainset.csv"

feature_size = 45


# Stock params dataset
class StockDataset(Dataset):
    '''
    the StockDataset - contains stock datas
    '''

    def __init__(self, input_filename=input_filename):
        '''
        load data from the file
        '''
        self.data_with_date = []    # native python list
        with open(input_filename) as datafile:
            # discard header line, save raw numbers only
            datafile.readline()
            # TODO: Read header names
            # NOTE: Header could contain Chinese characters, tempting to use `numpy`
            #       to read `.csv` would fail!
            self.data_with_date = csv.reader(datafile, delimiter=',')

            # data normalizations
            self.data = [ [ float(item) for item in rec[1:] ] for rec in self.data_with_date ]
            self.data = np.array(self.data)
            for idx in range(self.data.shape[1]):
                min_val = min(self.data[:, idx])
                max_val = max(self.data[:, idx])
                self.data[:, idx] -= min_val
                if max_val - min_val == 0:  # discard constant feature
                    self.data[:, idx] -= self.data[:, idx]
                else:
                    self.data[:, idx] /= max_val - min_val
            # for each in self.data:
            #     print(each)
            #     if input() == 'q': break
        self.len = self.data.shape[0] - 1   # `-1` for the last day will be used
        self.data = torch.from_numpy(self.data).float()


    def __getitem__(self, index):
        '''
        get data for feature and target hypothesis

        NOTE: For stock transactions are evaluated by the diff of today's
              price and the next day's, `y` labels are a little bit tricky
        '''
        if (index > self.len - 1):
            raise IndexError()
        features = self.data[index]     # the first row will be used to
                                        # generate `y` labels
        target = torch.Tensor([
            self.data[index+1][0],  # open price
            self.data[index+1][1],  # highest
            self.data[index+1][2],  # lowest
            self.data[index+1][3]   # close price
        ])
        return features, target


    def __len__(self):
        return self.len


######## Model ########

class Model(torch.nn.Module):
    '''
    my model for stock price predicting
    '''

    def __init__(self):
        super(Model, self).__init__()
        self.l1 = torch.nn.Linear(feature_size, 16)
        self.l2 = torch.nn.Linear(16, 8)
        self.l3 = torch.nn.Linear(8, 4)


    def forward(self, feat):
        hypo = self.l1(feat)
        hypo = self.l2(hypo)
        hypo = self.l3(hypo)
        return hypo


###############################################

dataset = StockDataset(input_filename)
train_loader = DataLoader(dataset=dataset,
                          batch_size=30,
                          shuffle=False)

model = Model()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1E-1)

##############################################

losses = []

for epoch in range(50):
    for index, data in enumerate(train_loader, 0):

        feature, target = data
        feature, target = Variable(feature), Variable(target)

        hypothesis = model(feature)

        loss = criterion(hypothesis, target)
        losses.append(loss.data[0])
        print("{:8} {:8} {:8}".format(epoch, index, loss.data[0]))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

plt.plot(losses)
plt.show()

#############################################


testset = StockDataset("./testset.csv")
test_loader = DataLoader(dataset=testset,
                         batch_size=1,
                         shuffle=False)

accuracy = 0
acc_map = []
for index, data in enumerate(test_loader, 0):

    feature, truth = data
    feature, truth = Variable(feature), Variable(truth)
    prediction = model(feature)

    # predict next day's rise / drop of stock price as criterion
    if ((prediction.data[0][0] - prediction.data[0][3])
            * (truth.data[0][0] - truth.data[0][3])) > 0:
        accuracy += 1
        acc_map.append(1)
    else:
        acc_map.append(0)

print("approx. accuracy ==> {:.2f}% ({} out of {})".format(
    accuracy / len(test_loader), accuracy, len(test_loader)))

# plt.plot(acc_map)
# plt.show()

