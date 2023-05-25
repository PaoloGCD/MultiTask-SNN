"""
Threshold-controlled three-blocks three-tasks SNN.

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

import lava.lib.dl.slayer as slayer

from matplotlib import animation

from src.misc import stats_multitask, assistant_3blocks, cuba_multitask

from src.misc.dataset_nmnist_multitask import augment, NMNISTDataset

# Get parameters
experiment_number = 0
parameters_path = "../../params/three-tasks-three-blocks.xml"
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
loss_rate = float(params['params']['loss_rate'])
threshold_1 = float(params['params']['threshold_1'])
threshold_2 = float(params['params']['threshold_2'])
threshold_3 = float(params['params']['threshold_3'])

result_path = f'results/Three-tasks-NMNIST-three-blocks-{experiment_number:02d}'
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
            'bias': 0
        }
        neuron_params_drop = {**neuron_params, 'dropout': slayer.neuron.Dropout(p=0.05)}

        self.feature_extraction_block = torch.nn.ModuleList([
            cuba_multitask.Dense(neuron_params_drop, 34 * 34 * 2, 512, weight_norm=True, delay=True),
            cuba_multitask.Dense(neuron_params_drop, 512, 512, weight_norm=True, delay=True)
        ])

        self.label_classification_block = torch.nn.ModuleList([
            cuba_multitask.Dense(neuron_params_drop, 512, 128, weight_norm=True, delay=True),
            cuba_multitask.Dense(neuron_params, 128, 14, weight_norm=True)
        ])

        self.label_task_block = torch.nn.ModuleList([
            cuba_multitask.Dense(neuron_params, 512, 3, weight_norm=True)
        ])

    def forward(self, spike):
        for block in self.feature_extraction_block:
            spike = block(spike)
        spike_label = spike
        spike_task = spike
        for block in self.label_classification_block:
            spike_label = block(spike_label)
        for block in self.label_task_block:
            spike_task = block(spike_task)
        return spike_label, spike_task

    def grad_flow(self, path):
        # helps monitor the gradient flow
        grad0 = [b.synapse.grad_norm for b in self.feature_extraction_block if hasattr(b, 'synapse')]
        grad1 = [b.synapse.grad_norm for b in self.label_classification_block if hasattr(b, 'synapse')]
        grad2 = [b.synapse.grad_norm for b in self.label_task_block if hasattr(b, 'synapse')]
        grad = grad0+grad1+grad2
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
        block = h.create_group('block_task_classification')
        for idx, b in enumerate(self.label_task_block):
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

stats = stats_multitask.LearningStatsHandler(number_output_blocks=2, number_tests=3)
assistant = assistant_3blocks.Assistant(net, error, optimizer, loss_rate=loss_rate, stats=stats, classifier=slayer.classifier.Rate.predict)

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
            task = torch.zeros(input_data.shape[0], dtype=torch.int64)
        elif random_number < 0.66:
            # set threshold to value 2
            for layer in net.feature_extraction_block:
                layer.neuron.threshold = threshold_2
            for layer in net.label_classification_block:
                layer.neuron.threshold = threshold_2
            label = label2
            task = torch.ones(input_data.shape[0], dtype=torch.int64)
        else:
            # set threshold to value 3
            for layer in net.feature_extraction_block:
                layer.neuron.threshold = threshold_3
            for layer in net.label_classification_block:
                layer.neuron.threshold = threshold_3
            label = label3
            task = torch.ones(input_data.shape[0], dtype=torch.int64)*2
        # train
        output_label, output_task = assistant.train(input_data, label, task)
    time_train = (time.time() - time_start)/60.0

    train_classifier_loss, train_task_loss = stats.training.loss
    train_classifier_acc, train_task_acc = stats.training.accuracy

    print(f'[Epoch {epoch:2d}/{epochs}] Train '
          f'loss = {train_classifier_loss:0.4f} / {train_task_loss:0.4f} '
          f'acc = {train_classifier_acc:0.4f} / {train_task_acc:0.4f}', end=' ')

    time_test_start = time.time()

    # set threshold for test 1
    for layer in net.feature_extraction_block:
        layer.neuron.threshold = threshold_1
    for layer in net.label_classification_block:
        layer.neuron.threshold = threshold_1

    for i, (input_data, label1, label2, label3) in enumerate(test_loader):  # training loop
        label_task_1 = torch.zeros(input_data.shape[0], dtype=torch.int64)
        output = assistant.test(input_data, label1, label_task_1, 0)

    test1_classifier_loss, test1_task_loss = stats.testing[0].loss
    test1_classifier_acc, test1_task_acc = stats.testing[0].accuracy
    print(f'| Test1 '
          f'loss = {test1_classifier_loss:0.4f} / {test1_task_loss:0.4f} '
          f'acc = {test1_classifier_acc:0.4f} / {test1_task_acc:0.4f}', end=' ')

    # set threshold to 2
    for layer in net.feature_extraction_block:
        layer.neuron.threshold = threshold_2
    for layer in net.label_classification_block:
        layer.neuron.threshold = threshold_2

    for i, (input_data, label1, label2, label3) in enumerate(test_loader):  # training loop
        label_task_2 = torch.ones(input_data.shape[0], dtype=torch.int64)
        output = assistant.test(input_data, label2, label_task_2, 1)

    time_test = (time.time() - time_test_start)/60.0

    test2_classifier_loss, test2_task_loss = stats.testing[1].loss
    test2_classifier_acc, test2_task_acc = stats.testing[1].accuracy
    print(f'| Test2 '
          f'loss = {test2_classifier_loss:0.4f} / {test2_task_loss:0.4f} '
          f'acc = {test2_classifier_acc:0.4f} / {test2_task_acc:0.4f}', end=' ')

    # set threshold for test 3
    for layer in net.feature_extraction_block:
        layer.neuron.threshold = threshold_3
    for layer in net.label_classification_block:
        layer.neuron.threshold = threshold_3

    for i, (input_data, label1, label2, label3) in enumerate(test_loader):  # training loop
        label_task_3 = torch.ones(input_data.shape[0], dtype=torch.int64)*2
        output = assistant.test(input_data, label3, label_task_3, 2)

    test3_classifier_loss, test3_task_loss = stats.testing[2].loss
    test3_classifier_acc, test3_task_acc = stats.testing[2].accuracy
    print(f'| Test3 '
          f'loss = {test3_classifier_loss:0.4f} / {test3_task_loss:0.4f} '
          f'acc = {test3_classifier_acc:0.4f} / {test3_task_acc:0.4f}', end=' ')

    print(f'| Time = {time_train+time_test:2.3f}')

    if stats.testing[0].best_accuracy:
        torch.save(net.state_dict(), result_path + '/network.pt')
    stats.update()
    stats.save(result_path + '/')
    net.grad_flow(result_path + '/')

# stats.plot(figsize=(15, 5))

net.load_state_dict(torch.load(result_path + '/network.pt'))
net.export_hdf5(result_path + '/network.net')

# Save input and output samples
input_data, _, _, _ = next(iter(train_loader))

# set threshold for test 1
for layer in net.feature_extraction_block:
    layer.neuron.threshold = threshold_1
for layer in net.label_classification_block:
    layer.neuron.threshold = threshold_1

# process data
output_label, output_task = net(input_data.to(device))
for i in range(3):
    inp_event = slayer.io.tensor_to_event(input_data[i].cpu().data.numpy().reshape(2, 34, 34, -1))
    out_event = slayer.io.tensor_to_event(output_label[i].cpu().data.numpy().reshape(1, 14, -1))
    inp_anim = inp_event.anim(plt.figure(figsize=(5, 5)), frame_rate=240)
    out_anim = out_event.anim(plt.figure(figsize=(10, 5)), frame_rate=240)
    inp_anim.save(f'{result_path}/inp-{i}.gif', animation.PillowWriter(fps=24), dpi=300)
    out_anim.save(f'{result_path}/out1-{i}.gif', animation.PillowWriter(fps=24), dpi=300)

# set threshold for test 2
for layer in net.feature_extraction_block:
    layer.neuron.threshold = threshold_2
for layer in net.label_classification_block:
    layer.neuron.threshold = threshold_2

# process data
output_label, output_task = net(input_data.to(device))
for i in range(3):
    out_event = slayer.io.tensor_to_event(output_label[i].cpu().data.numpy().reshape(1, 14, -1))
    out_anim = out_event.anim(plt.figure(figsize=(10, 5)), frame_rate=240)
    out_anim.save(f'{result_path}/out2-{i}.gif', animation.PillowWriter(fps=24), dpi=300)

# set threshold for test 3
for layer in net.feature_extraction_block:
    layer.neuron.threshold = threshold_3
for layer in net.label_classification_block:
    layer.neuron.threshold = threshold_3

# process data
output_label, output_task = net(input_data.to(device))
for i in range(3):
    out_event = slayer.io.tensor_to_event(output_label[i].cpu().data.numpy().reshape(1, 14, -1))
    out_anim = out_event.anim(plt.figure(figsize=(10, 5)), frame_rate=240)
    out_anim.save(f'{result_path}/out3-{i}.gif', animation.PillowWriter(fps=24), dpi=300)
