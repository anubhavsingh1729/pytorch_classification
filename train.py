import torch as t
from data import ChallengeDataset
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
import model
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# load the data from the csv file and perform a train-test-split
# this can be accomplished using the already imported pandas and sklearn.model_selection modules
# TODO
data = pd.load_csv('./data.csv')
train_data, test_data = train_test_split(data, test_size=0.33, random_state=42)
# set up data loading for the training and validation set each using t.utils.data.DataLoader and ChallengeDataset
# objects

train_dataset = ChallengeDataset(train_data, 'train')
test_dataset = ChallengeDataset(test_data, 'val')

train_dataloader = DataLoader(train_dataset, batch_size=32)
test_dataloader = DataLoader(test_dataset, batch_size=32)

# create an instance of our ResNet model
res = model.Model()

# set up a suitable loss criterion (you can find a pre-implemented loss functions in t.nn)
# set up the optimizer (see t.optim)
# create an object of type Trainer and set its early stopping criterion
criterion = t.nn.BCEWithLogitsLoss()
optimizer = t.optim.Adam(res.parameters(), lr=0.001)
trainer = Trainer(res, criterion, optimizer, train_dataloader, test_dataloader)

# go, go, go... call fit on trainer
res = trainer.fit()

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')
