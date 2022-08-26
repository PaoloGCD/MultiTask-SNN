"""
Threshold-controlled two block multi-task SNN for SHD data.

Built on Intel's Lava-dl tutorial implementation https://github.com/lava-nc/lava-dl/blob/main/tutorials/lava/lib/dl/slayer/nmnist/train.ipynb
@author: Paolo G. Cachi
"""

import os
import sys
import time
import h5py
import matplotlib.pyplot as plt
import torch
import xmltodict
from torch.utils.data import DataLoader

# import slayer from lava-dl
import lava.lib.dl.slayer as slayer

from matplotlib import animation

from src.misc import cuba_multitask
from src.misc.dataset_shd_multitask import SHDDataset

# Get parameters
experiment_number = 0
parameters_path = "../../params/shd-base-case.xml"
data_path = "../../data/SHD"
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
threshold_1 = float(params['params']['threshold_1'])
sample_length = int(params['params']['sample_length'])

result_path = f'results/SHD-base-case-{experiment_number:02d}'
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
        neuron_params_drop = {**neuron_params, 'dropout': slayer.neuron.Dropout(p=0.05), 'bias': 0}

        self.blocks = torch.nn.ModuleList([
            cuba_multitask.Dense(neuron_params_drop, 700, 256, weight_norm=True, delay=True),
            cuba_multitask.Dense(neuron_params_drop, 256, 128, weight_norm=True, delay=True),
            cuba_multitask.Dense(neuron_params, 128, 30, weight_norm=True)
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
        block = h.create_group('layer')
        for idx, b in enumerate(self.blocks):
            b.export_hdf5(block.create_group(f'{idx}'))


# device = torch.device('cpu')
device = torch.device('cuda')

net = Network().to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

training_set = SHDDataset(train=True, max_time=1.4, sample_length=sample_length, units=700, path=data_path)
testing_set = SHDDataset(train=False, max_time=1.4, sample_length=sample_length, units=700, path=data_path)

train_loader = DataLoader(dataset=training_set, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=testing_set, batch_size=32, shuffle=True)

error = slayer.loss.SpikeRate(true_rate=0.2, false_rate=0.03, reduction='sum').to(device)

stats = slayer.utils.LearningStats()
assistant = slayer.utils.Assistant(net, error, optimizer, stats, classifier=slayer.classifier.Rate.predict)

print('Training (using: %s)' % device)
for epoch in range(epochs):
    time_start = time.time()
    for i, (input_data, label, _) in enumerate(train_loader):
        output = assistant.train(input_data, label)
    time_train = (time.time() - time_start)/60.0

    train_loss = stats.training.loss
    train_acc = stats.training.accuracy
    print(f'[Epoch {epoch:2d}/{epochs}] Train loss = {train_loss:0.4f} acc = {train_acc:0.4f}', end=' ')

    time_test_start = time.time()

    # start test 1
    for i, (input_data, label, _) in enumerate(test_loader):  # training loop
        output = assistant.test(input_data, label)

    time_test = (time.time() - time_test_start) / 60.0

    test1_loss = stats.testing.loss
    test1_acc = stats.testing.accuracy
    print(f'| Test loss = {test1_loss:0.4f} acc = {test1_acc:0.4f}', end=' ')
    print(f'| Time train = {time_train:2.3f} test = {time_test:2.3f}')

    if stats.testing[0].best_accuracy:
        torch.save(net.state_dict(), result_path + '/network.pt')
    stats.update()
    stats.save(result_path + '/')
    net.grad_flow(result_path + '/')

# stats.plot(figsize=(15, 5))

net.load_state_dict(torch.load(result_path + '/network.pt'))
net.export_hdf5(result_path + '/network.net')

# Save input and output samples
input_data, _, _ = next(iter(train_loader))

# set threshold for test 1
for layer in net.feature_extraction_block:
    layer.neuron.threshold = threshold_1
for layer in net.label_classification_block:
    layer.neuron.threshold = threshold_1

# process data
output = net(input_data.to(device))
for i in range(3):
    inp_event = slayer.io.tensor_to_event(input_data[i].cpu().data.numpy().reshape(1, 28, 25, -1))
    out_event = slayer.io.tensor_to_event(output[i].cpu().data.numpy().reshape(1, 20, -1))
    inp_anim = inp_event.anim(plt.figure(figsize=(5, 5)), frame_rate=240)
    out_anim = out_event.anim(plt.figure(figsize=(10, 5)), frame_rate=240)
    inp_anim.save(f'{result_path}/inp-{i}.gif', animation.PillowWriter(fps=24), dpi=300)
    out_anim.save(f'{result_path}/out-{i}.gif', animation.PillowWriter(fps=24), dpi=300)
