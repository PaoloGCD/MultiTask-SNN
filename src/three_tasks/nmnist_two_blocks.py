"""
Threshold-controlled two-blocks three-tasks SNN.

Built on Intel's Lava-dl tutorial implementation https://github.com/lava-nc/lava-dl/blob/main/tutorials/lava/lib/dl/slayer/nmnist/train.ipynb
@author: Paolo G. Cachi
"""

import os
import sys
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import xmltodict
from torch.utils.data import DataLoader

# import slayer from lava-dl
import lava.lib.dl.slayer as slayer

from matplotlib import animation

from src.misc import stats_multitask, assistant_2blocks, cuba_multitask
from src.misc.dataset_nmnist_multitask import augment, NMNISTDataset

# Get parameters
experiment_number = 0
parameters_path = "../../params/three-tasks-two-blocks.xml"
data_path = "../../data/NMNIST"
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
threshold_2 = float(params['params']['threshold_2'])
threshold_3 = float(params['params']['threshold_3'])

result_path = f'results/Three-tasks-NMNIST-two-blocks-{experiment_number:02d}'
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

        self.feature_extraction_block = torch.nn.ModuleList([
            cuba_multitask.Dense(neuron_params_drop, 34 * 34 * 2, 512, weight_norm=True, delay=True),
            cuba_multitask.Dense(neuron_params_drop, 512, 512, weight_norm=True, delay=True)
        ])

        self.label_classification_block = torch.nn.ModuleList([
            cuba_multitask.Dense(neuron_params_drop, 512, 128, weight_norm=True, delay=True),
            cuba_multitask.Dense(neuron_params, 128, 14, weight_norm=True)
        ])

    def forward(self, spike):
        for block in self.feature_extraction_block:
            spike = block(spike)
        for block in self.label_classification_block:
            spike = block(spike)
        return spike

    def grad_flow(self, path):
        # helps monitor the gradient flow
        grad0 = [b.synapse.grad_norm for b in self.feature_extraction_block if hasattr(b, 'synapse')]
        grad1 = [b.synapse.grad_norm for b in self.label_classification_block if hasattr(b, 'synapse')]
        grad = grad0 + grad1

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()

        return grad

    def export_hdf5(self, filename):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        block = h.create_group('block_feature_extraction')
        for idx, b in enumerate(self.feature_extraction_block):
            b.export_hdf5(block.create_group(f'{idx}'))
        block = h.create_group('block_label_classification')
        for idx, b in enumerate(self.label_classification_block):
            b.export_hdf5(block.create_group(f'{idx}'))


# device = torch.device('cpu')
device = torch.device('cuda')

net = Network().to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

training_set = NMNISTDataset(train=True, transform=augment, path=data_path)
testing_set = NMNISTDataset(train=False, path=data_path)

train_loader = DataLoader(dataset=training_set, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=testing_set, batch_size=32, shuffle=True)

error = slayer.loss.SpikeRate(true_rate=0.2, false_rate=0.03, reduction='sum').to(device)

stats = stats_multitask.LearningStatsHandler(number_training_blocks=1, number_tasks=3)
assistant = assistant_2blocks.Assistant(net, error, optimizer, stats=stats, classifier=slayer.classifier.Rate.predict)

print('Training (using: %s)' % device)
for epoch in range(epochs):
    time_start = time.time()
    for i, (input_data, label1, label2, label3) in enumerate(train_loader):  # training loop
        # set threshold according to task
        random_number = np.random.rand()
        if random_number < 0.33:
            # set threshold to value 1
            for layer in net.feature_extraction_block:
                layer.neuron.threshold = threshold_1
            for layer in net.label_classification_block:
                layer.neuron.threshold = threshold_1
            label = label1
        elif random_number < 0.66:
            # set threshold to value 2
            for layer in net.feature_extraction_block:
                layer.neuron.threshold = threshold_2
            for layer in net.label_classification_block:
                layer.neuron.threshold = threshold_2
            label = label2
        else:
            # set threshold to value 3
            for layer in net.feature_extraction_block:
                layer.neuron.threshold = threshold_3
            for layer in net.label_classification_block:
                layer.neuron.threshold = threshold_3
            label = label3
        # train
        output = assistant.train(input_data, label)
    time_train = (time.time() - time_start)/60.0

    train_loss = stats.training.loss[0]
    train_acc = stats.training.accuracy[0]
    print(f'[Epoch {epoch:2d}/{epochs}] Train loss = {train_loss:0.4f} acc = {train_acc:0.4f}', end=' ')

    time_test_start = time.time()

    # set biases for test 1
    for layer in net.feature_extraction_block:
        layer.neuron.threshold = threshold_1
    for layer in net.label_classification_block:
        layer.neuron.threshold = threshold_1

    # start test 1
    for i, (input_data, label1, label2, label3) in enumerate(test_loader):  # training loop
        output = assistant.test(input_data, label1, 0)

    test1_loss = stats.testing[0].loss[0]
    test1_acc = stats.testing[0].accuracy[0]
    print(f'| Test1 loss = {test1_loss:0.4f} acc = {test1_acc:0.4f}', end=' ')

    # set biases for test 2
    for layer in net.feature_extraction_block:
        layer.neuron.threshold = threshold_2
    for layer in net.label_classification_block:
        layer.neuron.threshold = threshold_2

    for i, (input_data, label1, label2, label3) in enumerate(test_loader):  # training loop
        output = assistant.test(input_data, label2, 1)
    time_test = (time.time() - time_test_start)/60.0

    test2_loss = stats.testing[1].loss[0]
    test2_acc = stats.testing[1].accuracy[0]
    print(f'| Test2 loss = {test2_loss:0.4f} acc = {test2_acc:0.4f}', end=' ')

    # set biases for test 3
    for layer in net.feature_extraction_block:
        layer.neuron.threshold = threshold_3
    for layer in net.label_classification_block:
        layer.neuron.threshold = threshold_3

    for i, (input_data, label1, label2, label3) in enumerate(test_loader):  # training loop
        output = assistant.test(input_data, label3, 2)
    time_test = (time.time() - time_test_start) / 60.0

    test3_loss = stats.testing[2].loss[0]
    test3_acc = stats.testing[2].accuracy[0]
    print(f'| Test3 loss = {test3_loss:0.4f} acc = {test3_acc:0.4f}', end=' ')
    print(f'| Time train = {time_train:2.3f} test = {time_test:2.3f}')

    if stats.testing[0].best_accuracy[0]:
        torch.save(net.state_dict(), result_path + '/network.pt')
    stats.update()
    stats.save(result_path + '/')
    net.grad_flow(result_path + '/')

# stats.plot(figsize=(15, 5))

net.load_state_dict(torch.load(result_path + '/network.pt'))
net.export_hdf5(result_path + '/network.net')

# Save input and output samples
input_data, _, _, _ = next(iter(train_loader))

# set biases for test 1
for layer in net.feature_extraction_block:
    layer.neuron.threshold = threshold_1
for layer in net.label_classification_block:
    layer.neuron.threshold = threshold_1

# process data
output = net(input_data.to(device))
for i in range(3):
    inp_event = slayer.io.tensor_to_event(input_data[i].cpu().data.numpy().reshape(2, 34, 34, -1))
    out_event = slayer.io.tensor_to_event(output[i].cpu().data.numpy().reshape(1, 14, -1))
    inp_anim = inp_event.anim(plt.figure(figsize=(5, 5)), frame_rate=240)
    out_anim = out_event.anim(plt.figure(figsize=(10, 5)), frame_rate=240)
    inp_anim.save(f'{result_path}/inp-{i}.gif', animation.PillowWriter(fps=24), dpi=300)
    out_anim.save(f'{result_path}/out1-{i}.gif', animation.PillowWriter(fps=24), dpi=300)

# set biases for test 2
for layer in net.feature_extraction_block:
    layer.neuron.threshold = threshold_2
for layer in net.label_classification_block:
    layer.neuron.threshold = threshold_2

# process data
output = net(input_data.to(device))
for i in range(3):
    out_event = slayer.io.tensor_to_event(output[i].cpu().data.numpy().reshape(1, 14, -1))
    out_anim = out_event.anim(plt.figure(figsize=(10, 5)), frame_rate=240)
    out_anim.save(f'{result_path}/out2{i}.gif', animation.PillowWriter(fps=24), dpi=300)

# set biases for test 3
for layer in net.feature_extraction_block:
    layer.neuron.threshold = threshold_3
for layer in net.label_classification_block:
    layer.neuron.threshold = threshold_3

# process data
output = net(input_data.to(device))
for i in range(3):
    out_event = slayer.io.tensor_to_event(output[i].cpu().data.numpy().reshape(1, 14, -1))
    out_anim = out_event.anim(plt.figure(figsize=(10, 5)), frame_rate=240)
    out_anim.save(f'{result_path}/out3{i}.gif', animation.PillowWriter(fps=24), dpi=300)
