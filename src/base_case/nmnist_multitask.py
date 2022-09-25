"""
Threshold-controlled three block multi-task SNN.

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

from src.misc import stats_multitask, cuba_multitask

from src.misc.dataset_nmnist_multitask import augment, NMNISTDataset

# Get parameters
experiment_number = 0
parameters_path = "../../params/nmnist_multitask.xml"
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

result_path = f'results/Threshold-NMNIST-three-blocks-{experiment_number:02d}'
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

        self.task1_classification_block = torch.nn.ModuleList([
            cuba_multitask.Dense(neuron_params_drop, 512, 128, weight_norm=True, delay=True),
            cuba_multitask.Dense(neuron_params, 128, 10, weight_norm=True)
        ])

        self.task2_classification_block = torch.nn.ModuleList([
            cuba_multitask.Dense(neuron_params_drop, 512, 128, weight_norm=True, delay=True),
            cuba_multitask.Dense(neuron_params, 128, 2, weight_norm=True)
        ])

    def forward(self, spike):
        for block in self.feature_extraction_block:
            spike = block(spike)
        spike_task1 = spike
        spike_task2 = spike
        for block in self.task1_classification_block:
            spike_task1 = block(spike_task1)
        for block in self.task2_classification_block:
            spike_task2 = block(spike_task2)
        return spike_task1, spike_task2

    def grad_flow(self, path):
        # helps monitor the gradient flow
        grad0 = [b.synapse.grad_norm for b in self.feature_extraction_block if hasattr(b, 'synapse')]
        grad1 = [b.synapse.grad_norm for b in self.task1_classification_block if hasattr(b, 'synapse')]
        grad2 = [b.synapse.grad_norm for b in self.task2_classification_block if hasattr(b, 'synapse')]
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
        block = h.create_group('task1_classification_block')
        for idx, b in enumerate(self.task1_classification_block):
            b.export_hdf5(block.create_group(f'{idx}'))
        block = h.create_group('task2_classification_block')
        for idx, b in enumerate(self.task2_classification_block):
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

stats = stats_multitask.LearningStatsHandler(number_training_blocks=2, number_tasks=1)
classifier = slayer.classifier.Rate.predict

print('Training (using: %s)' % device)
for epoch in range(epochs):
    
    # Train
    time_start = time.time()
    for i, (input_data, label1, label2, _) in enumerate(train_loader):  # training loop
        net.train()

        input_data = input_data.to(device)
        target_label1 = label1.to(device)
        target_label2 = (label2-10).to(device)

        output_task1, output_task2 = net(input_data)

        loss_task1 = error(output_task1, target_label1)
        loss_task2 = error(output_task2, target_label2)

        loss = loss_task1 + loss_task2

        stats.training.num_samples += input_data.shape[0]
        stats.training.loss_sum[0] += loss_task1.cpu().data.item() * output_task1.shape[0]
        stats.training.loss_sum[1] += loss_task2.cpu().data.item() * output_task2.shape[0]
        if classifier is not None:
            stats.training.correct_samples[0] += torch.sum(torch.eq(classifier(output_task1), target_label1)).cpu().data.item()
            stats.training.correct_samples[1] += torch.sum(torch.eq(classifier(output_task2), target_label2)).cpu().data.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    time_train = (time.time() - time_start)/60.0

    train_classifier_loss1, train_classifier_loss2 = stats.training.loss
    train_classifier_acc1, train_classifier_acc2 = stats.training.accuracy

    print(f'[Epoch {epoch:2d}/{epochs}] Train '
          f'loss = {train_classifier_loss1:0.4f} / {train_classifier_loss2:0.4f} '
          f'acc = {train_classifier_acc1:0.4f} / {train_classifier_acc2:0.4f}', end=' ')

    # Test
    time_test_start = time.time()

    for i, (input_data, label1, label2, _) in enumerate(test_loader):
        net.eval()

        with torch.no_grad():
            input_data = input_data.to(device)
            target_label1 = label1.to(device)
            target_label2 = (label2-10).to(device)

            output_task1, output_task2 = net(input_data)

            loss_task1 = error(output_task1, target_label1)
            loss_task2 = error(output_task2, target_label2)

            loss = loss_task1 + loss_task2

            stats.testing[0].num_samples += input_data.shape[0]
            stats.testing[0].loss_sum[0] += loss_task1.cpu().data.item() * output_task1.shape[0]
            stats.testing[0].loss_sum[1] += loss_task2.cpu().data.item() * output_task2.shape[0]
            if classifier is not None:
                stats.testing[0].correct_samples[0] += torch.sum(torch.eq(classifier(output_task1), target_label1)).cpu().data.item()
                stats.testing[0].correct_samples[1] += torch.sum(torch.eq(classifier(output_task2), target_label2)).cpu().data.item()

    time_test = (time.time() - time_test_start)/60.0

    test1_classifier_loss1, test1_classifier_loss2 = stats.testing[0].loss
    test1_classifier_acc1, test1_classifier_acc2 = stats.testing[0].accuracy
    print(f'| Test1 '
          f'loss = {test1_classifier_loss1:0.4f} / {test1_classifier_loss2:0.4f} '
          f'acc = {test1_classifier_acc1:0.4f} / {test1_classifier_acc2:0.4f}', end=' ')

    print(f'| Time = {time_train+time_test:2.3f}')

    if stats.testing[0].best_accuracy:
        torch.save(net.state_dict(), result_path + '/network.pt')
    stats.update()
    stats.save(result_path + '/')
    net.grad_flow(result_path + '/')

net.load_state_dict(torch.load(result_path + '/network.pt'))
net.export_hdf5(result_path + '/network.net')

# Save input and output samples
input_data, _, _, _ = next(iter(train_loader))

# process data
output_task1, output_task2 = net(input_data.to(device))
for i in range(3):
    inp_event = slayer.io.tensor_to_event(input_data[i].cpu().data.numpy().reshape(2, 34, 34, -1))
    out1_event = slayer.io.tensor_to_event(output_task1[i].cpu().data.numpy().reshape(1, 10, -1))
    out2_event = slayer.io.tensor_to_event(output_task2[i].cpu().data.numpy().reshape(1, 2, -1))
    inp_anim = inp_event.anim(plt.figure(figsize=(5, 5)), frame_rate=240)
    out1_anim = out1_event.anim(plt.figure(figsize=(10, 5)), frame_rate=240)
    out2_anim = out1_event.anim(plt.figure(figsize=(10, 5)), frame_rate=240)
    inp_anim.save(f'{result_path}/inp.gif', animation.PillowWriter(fps=24), dpi=300)
    out1_anim.save(f'{result_path}/out1.gif', animation.PillowWriter(fps=24), dpi=300)
    out2_anim.save(f'{result_path}/out2.gif', animation.PillowWriter(fps=24), dpi=300)
