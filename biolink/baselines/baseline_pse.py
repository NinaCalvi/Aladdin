import pandas as pd
import numpy as np
import json
from biolink.eval import rank, hits_rate
from sklearn.linear_model import LogisticRegression
from torch import nn
import torch

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1290, 1002),
            nn.ReLU(),
            nn.Linear(1002, 963)
        )

    def forward(self, x):
        # convert tensor (128, 1, 28, 28) --> (128, 1*28*28)
        # x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x

test = pd.read_csv('ploypharmacy_facts_test.txt', sep='\t', header=None)
train = pd.read_csv('ploypharmacy_facts_train.txt', sep='\t', header=None)
valid = pd.read_csv('ploypharmacy_facts_valid.txt', sep='\t', header=None)

adj_matrix = np.loadtxt('pse_adj_matrix.txt')

with open('entities_dictionary_pse.json', 'r') as f:
    entities_dict = json.load(f)

with open('se_dict.json', 'r') as f:
    se_dict = json.load(f)


def create_x(row, mat_x):
    first_ = adj_matrix[entities_dict[row[0]]]
    second_ =  adj_matrix[entities_dict[row[2]]]
    conc = np.concatenate((first_, second_))
    mat_x[row.index] = conc

train_y = train[1].apply(lambda x: se_dict[x]).values
train_x = np.zeros((train.shape[0], 1290))
train.apply(lambda x: create_x(x, train_x), axis=1)

train_x=torch.tensor(train_x)
train_y=troch.tensor(train_y)

test_y = test[1].apply(lambda x: se_dict[x]).values
test_x = np.zeros((test.shape[0], 1290))
test.apply(lambda x: create_x(x, test_x), axis=1)

test_y=torch.tensor(test_y)
test_x=torch.tensor(test_x)

# lg = LogisticRegression()
# lg.fit(train_x, train_y)
# print('fitted')
#
# probs = lg.predict_proba(test_x)
# ranks = rank(probs, test_y)
# print('MRR', np.mean(1/ranks))
#

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Device: {device}')


model = MLP()
model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()


mean_train_losses = []
mean_valid_losses = []
valid_acc_list = []
epochs = 100

batch_size = 5000




for epoch in range(epochs):
    mrr_test = 0
    hits = dict()
    hits_at = [1, 3, 10]
    model.train()
    batch_start = 0

    train_losses = []
    valid_losses = []

    perms = torch.randperm(train_x.shape[0])
    y_data = train_y[perms]
    x_data = train_x[perms, :]
    while batch_start < train_x.shape[0]:
        batch_end = min(batch_start + batch_size, train_x.shape[0])
        input_batch = train_x[batch_start:batch_end].to(device)
        y_batch = train_y[batch_start:batch_end].to(device)

        optimizer.zero_grad()

        outputs = model(input_batch)
        loss = loss_fn(outputs, y_batch)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    ###STILL NEED TO ADD MRR ETC ETC ETC

    model.eval()
    counter = 0
    counter_hits = 0
    with torch.no_grad():
        while batch_start < test_x.shape[0]:
            counter += 1
            counter_hits += 2*min(batch_size, batch_end - batch_start)
            batch_end = min(batch_start + batch_size, train_x.shape[0])
            input_batch = test_x[batch_start:batch_end].to(device)
            y_batch = test_y[batch_start:batch_end].to(device)
            outputs = model(input_batch)
            # loss = loss_fn(outputs, labels)
            rank(outputs, y_batch)
            mrr = np.mean(1/rank_object)
            mrr_test += mrr
            hits_rate(rank_object, hits, hits_at)


    mrr_test /= counter
    for n in hits_at:
        hits[n] /= counter_hits

    print('epoch : {}, MRR : {:.4f}, h@1 : {:.4f}, h@3 : {:.2f}%, h@10 : {:.2f}%'\
         .format(epoch+1, mrr_test, hits[1], hits[3], hits[10]))
