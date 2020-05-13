import pandas as pd
import torch
from tqdm import tqdm
from sklearn.preprocessing import minmax_scale
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# load the data, and form the tensor dataset
from opt import LeastSquaresProxPointOptimizer


# load data
df = pd.read_csv('boston.csv')
inputs = minmax_scale(df[['RM','LSTAT','PTRATIO']].to_numpy()) # rescale inputs
inputs = np.hstack([inputs, np.ones((inputs.shape[0], 1))])  # add "1" to each sample
labels = minmax_scale(df['MEDV'].to_numpy())
dataset = torch.utils.data.TensorDataset(torch.tensor(inputs), -torch.tensor(labels))

# setup experiment parameters
batch_sizes = [1, 2, 3, 4, 5, 6]
experiments = range(20)
epochs = range(10)
step_sizes = np.geomspace(0.001, 100, 30)

# run experiments and record results
losses = pd.DataFrame(columns=['batch_size', 'step_size', 'experiment', 'epoch', 'loss'])
total_epochs = len(batch_sizes) * len(experiments) * len(step_sizes) * len(epochs)
with tqdm(total=total_epochs, desc='batch_size = NA, step_size = NA, experiment = NA',
          unit='epochs',
          ncols=160) as pbar:
    for batch_size in batch_sizes:
        for step_size in step_sizes:
            for experiment in experiments:
                x = torch.empty(4, requires_grad=False, dtype=torch.float64)
                torch.nn.init.normal_(x)

                optimizer = LeastSquaresProxPointOptimizer(x, step_size)
                for epoch in epochs:
                    epoch_loss = 0.
                    for A_batch, b_batch in torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size):
                        batch_losses = optimizer.step(A_batch, b_batch)
                        epoch_loss += torch.sum(batch_losses).item()

                    epoch_loss /= len(dataset)
                    losses = losses.append(pd.DataFrame.from_dict(
                        {'batch_size': [batch_size],
                         'step_size': [step_size],
                         'experiment': [experiment],
                         'epoch': [epoch],
                         'loss': [epoch_loss]}), sort=True)

                    pbar.update()
                    pbar.set_description(f'batch_size = {batch_size}, step_size = {step_size}, experiment = {experiment}')


# save and read from CSV - so that we can use the results instead of re-computing them,
# by commenting everything up to the next line.
losses.to_csv('results.txt', header=True, index=False)
losses = pd.read_csv('results.txt', header=0)

best_losses = losses[['batch_size', 'step_size', 'experiment', 'loss']]\
    .groupby(['batch_size', 'step_size', 'experiment'], as_index=False)\
    .min()

sns.set()
plot_losses = best_losses.copy()
plot_losses.loc[:, 'batch_size'] = plot_losses.loc[:, 'batch_size'].astype(str)
ax = sns.lineplot(x='step_size', y='loss', hue='batch_size', data=plot_losses, err_style='band')
ax.set_yscale('log')
ax.set_xscale('log')
plt.show()

plot_losses = best_losses[best_losses['step_size'] >= 0.1]
plot_losses.loc[:, 'batch_size'] = plot_losses.loc[:, 'batch_size'].astype(str)
ax = sns.lineplot(x='step_size', y='loss', hue='batch_size', data=plot_losses, err_style='band')
ax.set_yscale('log')
ax.set_xscale('log')
plt.show()
