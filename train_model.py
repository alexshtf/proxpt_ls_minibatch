import pandas as pd
import torch
from sklearn.preprocessing import minmax_scale
import numpy as np

# load the data, and form the tensor dataset
from opt import LeastSquaresProxPointOptimizer

df = pd.read_csv('boston.csv')
inputs = minmax_scale(df[['RM','LSTAT','PTRATIO']].to_numpy()) # rescale inputs
inputs = np.hstack([inputs, np.ones((inputs.shape[0], 1))])  # add "1" to each sample
labels = minmax_scale(df['MEDV'].to_numpy())
dataset = torch.utils.data.TensorDataset(torch.tensor(inputs), -torch.tensor(labels))

# run a trainign loop
x = torch.empty(4, dtype=torch.float64, requires_grad=False)
torch.nn.init.normal_(x)
optimizer = LeastSquaresProxPointOptimizer(x, 0.3)
for epoch in range(20):
    epoch_loss = 0.
    for A_batch, b_batch in torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=6):
        losses = optimizer.step(A_batch, b_batch)
        epoch_loss += torch.sum(losses).item()

    epoch_loss /= len(dataset)
    print(f'epoch = {epoch}, loss = {epoch_loss}')