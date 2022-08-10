"""
SNN for single-task NMNIST classification.

Extracted from https://github.com/lava-nc/lava-dl/blob/main/tutorials/lava/lib/dl/slayer/nmnist/
@author: Intel Corporation

Rewritten by Paolo G. Cachi
"""

import os
import h5py
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# import slayer from lava-dl
import lava.lib.dl.slayer as slayer

from matplotlib import animation
from src.misc.dataset_nmnist import augment, NMNISTDataset


class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        neuron_params = {
            'threshold': 1.25,
            'current_decay': 0.25,
            'voltage_decay': 0.03,
            'tau_grad': 0.03,
            'scale_grad': 3,
            'requires_grad': True,
        }
        neuron_params_drop = {**neuron_params, 'dropout': slayer.neuron.Dropout(p=0.05), }

        self.blocks = torch.nn.ModuleList([
            slayer.block.cuba.Dense(neuron_params_drop, 34 * 34 * 2, 512, weight_norm=True, delay=True),
            slayer.block.cuba.Dense(neuron_params_drop, 512, 512, weight_norm=True, delay=True),
            slayer.block.cuba.Dense(neuron_params_drop, 512, 128, weight_norm=True, delay=True),
            slayer.block.cuba.Dense(neuron_params, 128, 10, weight_norm=True),
        ])

    def forward(self, spike):
        for block in self.blocks:
            spike = block(spike)
        return spike

    def grad_flow(self, path):
        # helps monitor the gradient flow
        grad = [b.synapse.grad_norm for b in self.blocks if hasattr(b, 'synapse')]

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()

        return grad

    def export_hdf5(self, filename):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))


trained_folder = 'results/base_case'
os.makedirs(trained_folder, exist_ok=True)

# device = torch.device('cpu')
device = torch.device('cuda')

net = Network().to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

training_set = NMNISTDataset(train=True, transform=augment, path='../../data')
testing_set = NMNISTDataset(train=False, path='../../data')

train_loader = DataLoader(dataset=training_set, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=testing_set, batch_size=32, shuffle=True)

error = slayer.loss.SpikeRate(true_rate=0.2, false_rate=0.03, reduction='sum').to(device)

stats = slayer.utils.LearningStats()
assistant = slayer.utils.Assistant(net, error, optimizer, stats, classifier=slayer.classifier.Rate.predict)
epochs = 100

print('Initializing training')
for epoch in range(epochs):
    for i, (input, label) in enumerate(train_loader):  # training loop
        output = assistant.train(input, label)
    print(f'[Epoch {epoch:2d}/{epochs}] {stats}', end='')

    for i, (input, label) in enumerate(test_loader):  # training loop
        output = assistant.test(input, label)
    print(f'\r[Epoch {epoch:2d}/{epochs}] {stats}')

    if stats.testing.best_accuracy:
        torch.save(net.state_dict(), trained_folder + '/network.pt')
    stats.update()
    stats.save(trained_folder + '/')
    net.grad_flow(trained_folder + '/')

stats.plot(figsize=(15, 5))

net.load_state_dict(torch.load(trained_folder + '/network.pt'))
net.export_hdf5(trained_folder + '/network.net')

output = net(input.to(device))
for i in range(5):
    inp_event = slayer.io.tensor_to_event(input[i].cpu().data.numpy().reshape(2, 34, 34, -1))
    out_event = slayer.io.tensor_to_event(output[i].cpu().data.numpy().reshape(1, 10, -1))
    inp_anim = inp_event.anim(plt.figure(figsize=(5, 5)), frame_rate=240)
    out_anim = out_event.anim(plt.figure(figsize=(10, 5)), frame_rate=240)
    inp_anim.save(f'{trained_folder}/inp-{i}.gif', animation.PillowWriter(fps=24), dpi=300)
    out_anim.save(f'{trained_folder}/out-{i}.gif', animation.PillowWriter(fps=24), dpi=300)
