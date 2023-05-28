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
from src.misc.dataset_dvsgesture128 import DVSGesture128Dataset, augment
from src.misc import lif_multitask

# Get parameters
experiment_number = 0
parameters_path = "../../params/base_case.xml"
data_path = "../../data/DVS-Gesture128"
gpu_number = 0
debug = 0
weight_norm = True
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
elif len(sys.argv) == 6:
    parameters_path = str(sys.argv[1])
    data_path = str(sys.argv[2])
    experiment_number = int(sys.argv[3])
    gpu_number = int(sys.argv[4])
    debug = int(sys.argv[5])

with open(parameters_path) as fd:
    params = xmltodict.parse(fd.read())

epochs = int(params['params']['epochs'])

result_path = f'results/DVSGesture128-feedforward-{experiment_number:02d}'
os.makedirs(result_path, exist_ok=True)

with open(result_path + '/parameters.xml', 'w') as param_file:
    param_file.write(xmltodict.unparse(params))

print('EXPERIMENT', experiment_number)


class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        neuron_params = {
            'threshold': 1.25,
            'voltage_decay': 0.5,
            'tau_grad': 0.01,
            'scale_grad': 3,
            'requires_grad': True,
        }
        neuron_batch_norm = {**neuron_params, 'norm': torch.nn.BatchNorm3d, }
        neuron_params_drop = {**neuron_params, 'dropout': slayer.neuron.Dropout(p=0.5), }
        neuron_params_norm_drop = {**neuron_params, 'dropout': slayer.neuron.Dropout(p=0.5),
                                   'norm': torch.nn.BatchNorm3d}

        self.blocks = torch.nn.ModuleList([

            lif_multitask.Conv(neuron_batch_norm, 2, 128, 3, weight_norm=weight_norm, delay=False, padding=1),
            torch.nn.MaxPool3d((2, 2, 1), stride=(2, 2, 1)),
            lif_multitask.Conv(neuron_batch_norm, 128, 128, 3, weight_norm=weight_norm, delay=False, padding=1),
            torch.nn.MaxPool3d((2, 2, 1), stride=(2, 2, 1)),
            lif_multitask.Conv(neuron_params_norm_drop, 128, 128, 3, weight_norm=weight_norm, delay=False, padding=1),
            torch.nn.MaxPool3d((2, 2, 1), stride=(2, 2, 1)),
            lif_multitask.Conv(neuron_params_norm_drop, 128, 128, 3, weight_norm=weight_norm, delay=False, padding=1),
            torch.nn.MaxPool3d((2, 2, 1), stride=(2, 2, 1)),
            lif_multitask.Conv(neuron_params_norm_drop, 128, 128, 3, weight_norm=weight_norm, delay=False, padding=1),
            torch.nn.MaxPool3d((2, 2, 1), stride=(2, 2, 1)),
            slayer.block.cuba.Flatten(),

            lif_multitask.Dense(neuron_params_drop, 2048, 512, weight_scale=30, weight_norm=weight_norm, delay=False),
            lif_multitask.Dense(neuron_params_drop, 512, 110, weight_scale=30, weight_norm=weight_norm, delay=False),
            torch.nn.AvgPool2d((10, 1), (10, 1))
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

    def print_grad_flow(self):
        print('[', end=' ')
        for block in self.blocks:
            if hasattr(block, 'synapse'):
                print("%14.12f" % block.synapse.grad_norm, end=' ')
        print(']')

    def print_neuron_current_decay(self):
        print('[', end=' ')
        for block in self.blocks:
            if hasattr(block, 'neuron'):
                print(
                    "%6.4f" % block.neuron.current_decay.grad.item() if block.neuron.current_decay.grad is not None else 0.0,
                    end=' ')
        print(']')

    def print_neuron_voltage_decay(self):
        print('[', end=' ')
        for block in self.blocks:
            if hasattr(block, 'neuron'):
                print(
                    "%6.4f" % block.neuron.voltage_decay.grad.item() if block.neuron.voltage_decay.grad is not None else 0.0,
                    end=' ')
        print(']')

    def export_hdf5(self, filename):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for idx, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{idx}'))


# device = torch.device('cpu')
device = torch.device('cuda:%d' % gpu_number)
# device = torch.device('mps')

print(device)

net = Network().to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=64)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=64)

training_set = DVSGesture128Dataset(train=True,
                                    path=data_path,
                                    sub_path='/20frames128sizeTimeValueT',
                                    time_steps=20,
                                    output_size=128,
                                    codification='time',
                                    keep_value=True,
                                    transform=augment)
testing_set = DVSGesture128Dataset(train=False,
                                   path=data_path,
                                   sub_path='/20frames128sizeTimeValueT',
                                   time_steps=20,
                                   output_size=128,
                                   codification='time',
                                   keep_value=True)

train_loader = DataLoader(dataset=training_set, batch_size=16, shuffle=True)
test_loader = DataLoader(dataset=testing_set, batch_size=16, shuffle=True)

error = slayer.loss.SpikeRate(true_rate=1.0, false_rate=0.03, reduction='sum').to(device)

stats = slayer.utils.LearningStats()
classifier = slayer.classifier.Rate.predict
assistant = slayer.utils.Assistant(net, error, optimizer, stats, classifier=classifier)

# Debug
if debug:
    print('Initializing debug')
    input_data, label = next(iter(train_loader))
    net.print_grad_flow()
    # net.print_neuron_current_decay()
    net.print_neuron_voltage_decay()
    for epoch in range(200):
        time_start = time.time()
        output = assistant.train(input_data, label)
        time_train = time.time() - time_start
        scheduler.step()

        print(f'Epoch {epoch:03d}: {output.sum().item():5.0f} {stats.training.loss:8.4f} {stats.training.accuracy:8.4f}'
              f' [{time_train:5.2f}]', end=' ')

        if epoch % 1 == 0:
            net.print_grad_flow()
            # net.print_neuron_current_decay()
            # net.print_neuron_voltage_decay()

        stats.update()

print('Initializing training')
for epoch in range(epochs):
    time_start = time.time()
    for i, (input_data, label) in enumerate(train_loader):
        output = assistant.train(input_data, label)

    print(f'[Epoch {epoch:3d}/{epochs}] Train '
          f'loss = {stats.training.loss:7.4f} '
          f'acc = {stats.training.accuracy:7.4f}', end=' ')

    scheduler.step()

    target_label_list = []
    predicted_label_list = []
    for i, (input_data, label) in enumerate(test_loader):
        output = assistant.test(input_data, label)
        predicted_label = classifier(output)
        target_label_list.extend(label.tolist())
        predicted_label_list.extend(predicted_label.tolist())
    time_total = (time.time() - time_start) / 60.0

    print(f'| Test '
          f'loss = {stats.testing.loss:7.4f} '
          f'acc = {stats.testing.accuracy:7.4f} ', end=' ')

    print(f'| Time = {time_total:2.3f}')

    if stats.testing.best_accuracy:
        torch.save(net.state_dict(), result_path + '/network.pt')
        torch.save(target_label_list, result_path + '/target_label.pt')
        torch.save(predicted_label_list, result_path + '/predicted_label.pt')
    stats.update()
    stats.save(result_path + '/')
    net.grad_flow(result_path + '/')

print()
print('max train acc', stats.training.max_accuracy)
print('max test acc', stats.testing.max_accuracy)

stats.plot(figsize=(15, 5))

net.load_state_dict(torch.load(result_path + '/network.pt'))
net.export_hdf5(result_path + '/network.net')

# Save input and output samples
input_data, _ = next(iter(train_loader))

output = net(input_data.to(device))
for i in range(5):
    inp_event = slayer.io.tensor_to_event(input_data[i].cpu().data.numpy().reshape(2, 128, 128, -1))
    out_event = slayer.io.tensor_to_event(output[i].cpu().data.numpy().reshape(1, 10, -1))
    inp_anim = inp_event.anim(plt.figure(figsize=(5, 5)), frame_rate=240)
    out_anim = out_event.anim(plt.figure(figsize=(10, 5)), frame_rate=240)
    inp_anim.save(f'{result_path}/inp-{i}.gif', animation.PillowWriter(fps=24), dpi=300)
    out_anim.save(f'{result_path}/out-{i}.gif', animation.PillowWriter(fps=24), dpi=300)

