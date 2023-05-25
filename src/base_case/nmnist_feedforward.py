"""
SNN for single-task NMNIST classification.

Extracted from https://github.com/lava-nc/lava-dl/blob/main/tutorials/lava/lib/dl/slayer/nmnist/
@author: Intel Corporation

Rewritten by Paolo G. Cachi
"""

import os
import time
import sys
import h5py
import xmltodict
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# import slayer from lava-dl
import lava.lib.dl.slayer as slayer

from matplotlib import animation
from src.misc.dataset_nmnist_multitask import augment, NMNISTDataset

# Get parameters
experiment_number = 0
parameters_path = "../../params/base_case.xml"
data_path = "../../data/NMNIST"
gpu_number = 0
if len(sys.argv) == 3:
    parameters_path = str(sys.argv[1])
    data_path = str(sys.argv[2])
elif len(sys.argv) == 4:
    parameters_path = str(sys.argv[1])
    data_path = str(sys.argv[2])
    experiment_number = int(sys.argv[3])
elif len(sys.argv) == 5:
    parameters_path = str(sys.argv[1])
    data_path = str(sys.argv[2])
    experiment_number = int(sys.argv[3])
    gpu_number = int(sys.argv[4])

with open(parameters_path) as fd:
    params = xmltodict.parse(fd.read())

epochs = int(params['params']['epochs'])

result_path = f'results/Base-case-NMNIST-{experiment_number:02d}'
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

    def get_grad_flow(self):
        return [b.synapse.grad_norm for b in self.blocks if hasattr(b, 'synapse')]

    def export_hdf5(self, filename):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for idx, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{idx}'))


# device = torch.device('cpu')
device = torch.device('cuda:%d' % gpu_number)

net = Network().to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

training_set = NMNISTDataset(train=True, transform=augment, path=data_path)
testing_set = NMNISTDataset(train=False, path=data_path)

train_loader = DataLoader(dataset=training_set, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=testing_set, batch_size=32, shuffle=True)

error = slayer.loss.SpikeRate(true_rate=0.2, false_rate=0.03, reduction='sum').to(device)

stats = slayer.utils.LearningStats()
assistant = slayer.utils.Assistant(net, error, optimizer, stats, classifier=slayer.classifier.Rate.predict)

# Debug
print('Initializing debug')
input_data, label, _, _ = next(iter(train_loader))
for epoch in range(200):
    time_start = time.time()
    output = assistant.train(input_data, label)
    if epoch % 1 == 0:
        weight_grad = net.get_grad_flow()
        print("%05d" % epoch, end=' ')
        print("[", "%.12f" % weight_grad[0], "%.12f" % weight_grad[1], "]", end=' ')
    # Debug
    print(f' {output.sum().item():5.0f} {stats.training.loss:8.4f} {stats.training.accuracy:8.4f}')
    stats.update()

print('Initializing training')
for epoch in range(epochs):
    time_start = time.time()
    for i, (input_data, label, _, _) in enumerate(train_loader):  # training loop
        output = assistant.train(input_data, label)
    print(f'[Epoch {epoch:2d}/{epochs}] Train '
          f'loss = {stats.training.loss:0.4f} '
          f'acc = {stats.training.accuracy:0.4f}', end=' ')

    time_train = (time.time() - time_start) / 60.0
    time_test_start = time.time()

    for i, (input_data, label, _, _) in enumerate(test_loader):  # training loop
        output = assistant.test(input_data, label)

    print(f'| Test '
          f'loss = {stats.testing.loss:0.4f} '
          f'acc = {stats.testing.accuracy:0.4f} ', end=' ')

    time_test = (time.time() - time_test_start) / 60.0

    print(f'| Time train = {time_train:2.3f} test = {time_test:2.3f}')

    if stats.testing.best_accuracy:
        torch.save(net.state_dict(), result_path + '/network.pt')
    stats.update()
    stats.save(result_path + '/')
    net.grad_flow(result_path + '/')

stats.plot(figsize=(15, 5))

net.load_state_dict(torch.load(result_path + '/network.pt'))
net.export_hdf5(result_path + '/network.net')

# Save input and output samples
input_data, _, _, _ = next(iter(train_loader))

output = net(input_data.to(device))
for i in range(5):
    inp_event = slayer.io.tensor_to_event(input_data[i].cpu().data.numpy().reshape(2, 34, 34, -1))
    out_event = slayer.io.tensor_to_event(output[i].cpu().data.numpy().reshape(1, 10, -1))
    inp_anim = inp_event.anim(plt.figure(figsize=(5, 5)), frame_rate=240)
    out_anim = out_event.anim(plt.figure(figsize=(10, 5)), frame_rate=240)
    inp_anim.save(f'{result_path}/inp-{i}.gif', animation.PillowWriter(fps=24), dpi=300)
    out_anim.save(f'{result_path}/out-{i}.gif', animation.PillowWriter(fps=24), dpi=300)
