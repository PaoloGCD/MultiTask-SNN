"""
SNN for single-task NMNIST classification.

Extracted from https://github.com/lava-nc/lava-dl/blob/main/tutorials/lava/lib/dl/slayer/nmnist/
@author: Intel Corporation

Rewritten by Paolo G. Cachi
"""

import os
import sys
import time
import h5py
import xmltodict
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# import slayer from lava-dl
import lava.lib.dl.slayer as slayer

from matplotlib import animation
from src.misc.dataset_cifar10dvs import CIFAR10DVSDataset

# Get parameters
experiment_number = 0
parameters_path = "../../params/base_case.xml"
data_path = "../../data/CIFAR10-DVS"
if len(sys.argv) == 3:
    parameters_path = str(sys.argv[1])
    data_path = str(sys.argv[2])
elif len(sys.argv) == 4:
    parameters_path = str(sys.argv[1])
    data_path = str(sys.argv[2])
    experiment_number = int(sys.argv[3])

with open(parameters_path) as fd:
    params = xmltodict.parse(fd.read())

epochs = int(params['params']['epochs'])

result_path = f'results/CIFAR10DVS-lenet-{experiment_number:02d}'
os.makedirs(result_path, exist_ok=True)

with open(result_path + '/parameters.xml', 'w') as param_file:
    param_file.write(xmltodict.unparse(params))

print('EXPERIMENT', experiment_number)


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
            slayer.block.cuba.Conv(neuron_params_drop, 2, 6, 5, weight_norm=True, delay=True, padding=2),
            slayer.block.cuba.Pool(neuron_params_drop, 2, weight_norm=True, delay=True),
            slayer.block.cuba.Conv(neuron_params_drop, 6, 16, 5, weight_norm=True, delay=True, padding=2),
            slayer.block.cuba.Pool(neuron_params_drop, 2, weight_norm=True, delay=True),
            slayer.block.cuba.Flatten(),
            slayer.block.cuba.Dense(neuron_params_drop, 1024, 256, weight_norm=True, delay=True),
            slayer.block.cuba.Dense(neuron_params_drop, 256, 128, weight_norm=True, delay=True),
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
        for idx, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{idx}'))


# device = torch.device('cpu')
device = torch.device('cuda')

net = Network().to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

training_set = CIFAR10DVSDataset(train=True, path=data_path, sample_length=300)
testing_set = CIFAR10DVSDataset(train=False, path=data_path, sample_length=300)

train_loader = DataLoader(dataset=training_set, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=testing_set, batch_size=32, shuffle=True)

error = slayer.loss.SpikeRate(true_rate=0.2, false_rate=0.03, reduction='sum').to(device)

stats = slayer.utils.LearningStats()
assistant = slayer.utils.Assistant(net, error, optimizer, stats, classifier=slayer.classifier.Rate.predict)

print('Initializing training')
for epoch in range(epochs):
    time_start = time.time()
    for i, (input_data, label) in enumerate(train_loader):  # training loop
        output = assistant.train(input_data, label)
    print(f'[Epoch {epoch:2d}/{epochs}] {stats}', end='')

    for i, (input_data, label) in enumerate(test_loader):  # training loop
        output = assistant.test(input_data, label)
    time_total = (time.time() - time_start) / 60.0
    print(f'\r[Epoch {epoch:2d}/{epochs}] {stats}', end='')
    print(f'| Time = {time_total:2.3f}')

    if stats.testing.best_accuracy:
        torch.save(net.state_dict(), result_path + '/network.pt')
    stats.update()
    stats.save(result_path + '/')
    net.grad_flow(result_path + '/')

stats.plot(figsize=(15, 5))

net.load_state_dict(torch.load(result_path + '/network.pt'))
net.export_hdf5(result_path + '/network.net')

# Save input and output samples
input_data, _ = next(iter(train_loader))

output = net(input_data.to(device))
for i in range(5):
    inp_event = slayer.io.tensor_to_event(input_data[i].cpu().data.numpy().reshape(2, 32, 32, -1))
    out_event = slayer.io.tensor_to_event(output[i].cpu().data.numpy().reshape(1, 10, -1))
    inp_anim = inp_event.anim(plt.figure(figsize=(5, 5)), frame_rate=240)
    out_anim = out_event.anim(plt.figure(figsize=(10, 5)), frame_rate=240)
    inp_anim.save(f'{result_path}/inp-{i}.gif', animation.PillowWriter(fps=24), dpi=300)
    out_anim.save(f'{result_path}/out-{i}.gif', animation.PillowWriter(fps=24), dpi=300)
