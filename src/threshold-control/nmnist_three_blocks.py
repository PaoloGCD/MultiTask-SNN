import os
import time
import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

import lava.lib.dl.slayer as slayer

from matplotlib import animation

from src.misc import stats_multitask, assistant_multitask, cuba_multitask

from src.misc.dataset_nmnist_multitask import augment, NMNISTDataset


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
            cuba_multitask.Dense(neuron_params, 128, 12, weight_norm=True)
        ])

        self.label_task_block = torch.nn.ModuleList([
            cuba_multitask.Dense(neuron_params, 512, 2, weight_norm=True)
        ])

    def forward(self, spike):
        for layer in self.feature_extraction_block:
            spike = layer(spike)
        spike_label = spike
        spike_task = spike
        for layer in self.label_classification_block:
            spike_label = layer(spike_label)
        for layer in self.label_task_block:
            spike_task = layer(spike_task)
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
        layer = h.create_group('block_feature_extraction')
        for i, b in enumerate(self.feature_extraction_block):
            b.export_hdf5(layer.create_group(f'{i}'))
        layer = h.create_group('block_label_classification')
        for i, b in enumerate(self.label_classification_block):
            b.export_hdf5(layer.create_group(f'{i}'))
        layer = h.create_group('block_task_classification')
        for i, b in enumerate(self.label_task_block):
            b.export_hdf5(layer.create_group(f'{i}'))


trained_folder = 'Trained'
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

stats = stats_multitask.LearningStats()
assistant = assistant_multitask.Assistant(net, error, optimizer, stats, classifier=slayer.classifier.Rate.predict)
epochs = 100

print('Training using:', device)
threshold_1 = 1.25
threshold_2 = 5
loss_rate = 0.5
for epoch in range(epochs):
    time_start = time.time()
    for i, (input, label1, label2) in enumerate(train_loader):  # training loop
        # set bias according to task
        if np.random.rand() < 0.5:
            # set biases to 1
            for layer in net.feature_extraction_block:
                layer.neuron.threshold = threshold_1
            for layer in net.label_classification_block:
                layer.neuron.threshold = threshold_1
            label = label1
            task = torch.zeros(input.shape[0], dtype=torch.int64)
        else:
            # set biases to 0
            for layer in net.feature_extraction_block:
                layer.neuron.threshold = threshold_2
            for layer in net.label_classification_block:
                layer.neuron.threshold = threshold_2
            label = label2
            task = torch.ones(input.shape[0], dtype=torch.int64)
        # train
        output_label, output_task = assistant.train(input, label, task, loss_rate)
    time_train = (time.time() - time_start)/60.0

    train_classifier_loss = stats.training.classifier_loss
    train_task_loss = stats.training.task_loss

    train_classifier_acc = stats.training.classifier_accuracy
    train_task_acc = stats.training.task_accuracy

    print(f'[Epoch {epoch:2d}/{epochs}] Train loss = {train_classifier_loss:0.4f} / {train_task_loss:0.4f} acc = {train_classifier_acc:0.4f} / {train_task_acc:0.4f}', end=' ')

    time_test_start = time.time()

    # set biases for test 1
    for layer in net.feature_extraction_block:
        layer.neuron.threshold = threshold_1
    for layer in net.label_classification_block:
        layer.neuron.threshold = threshold_1

    for i, (input, label1, label2) in enumerate(test_loader):  # training loop
        label_task_1 = torch.zeros(input.shape[0], dtype=torch.int64)
        output = assistant.test(input, label1, label_task_1, 1)

    test1_classifier_loss = stats.testing1.classifier_loss
    test1_task_loss = stats.testing1.task_loss

    test1_classifier_acc = stats.testing1.classifier_accuracy
    test1_task_acc = stats.testing1.task_accuracy
    print(f'| Test1 loss = {test1_classifier_loss:0.4f} / {test1_task_loss:0.4f} acc = {test1_classifier_acc:0.4f} / {test1_task_acc}', end=' ')

    # set biases to 2
    for layer in net.feature_extraction_block:
        layer.neuron.threshold = threshold_2
    for layer in net.label_classification_block:
        layer.neuron.threshold = threshold_2

    for i, (input, label1, label2) in enumerate(test_loader):  # training loop
        label_task_2 = torch.ones(input.shape[0], dtype=torch.int64)
        output = assistant.test(input, label2, label_task_2, 2)

    time_test = (time.time() - time_test_start)/60.0

    test2_classifier_loss = stats.testing2.classifier_loss
    test2_task_loss = stats.testing2.task_loss

    test2_classifier_acc = stats.testing2.classifier_accuracy
    test2_task_acc = stats.testing2.task_accuracy
    print(f'| Test2 loss = {test2_classifier_loss:0.4f} / {test2_task_loss:0.4f} acc = {test2_classifier_acc:0.4f} / {test2_task_acc}', end=' ')
    print(f'| Time = {time_train+time_test:2.3f}')

    # if epoch % 20 == 0:  # cleanup display
    #     print('\r', ' ' * len(f'\r[Epoch {epoch:2d}/{epochs}] {stats}'))
    #     stats_str = str(stats).replace("| ", "\n")
    #     print(f'[Epoch {epoch:2d}/{epochs}]\n{stats_str}')

    if stats.testing1.best_classifier_accuracy:
        torch.save(net.state_dict(), trained_folder + '/network.pt')
    stats.update()
    stats.save(trained_folder + '/')
    net.grad_flow(trained_folder + '/')

# stats.plot(figsize=(15, 5))

net.load_state_dict(torch.load(trained_folder + '/network.pt'))
net.export_hdf5(trained_folder + '/network.net')

# set threshold for test 1
for layer in net.feature_extraction_block:
    layer.neuron.threshold = threshold_1
for layer in net.label_classification_block:
    layer.neuron.threshold = threshold_1
output_label, output_task = net(input.to(device))
for i in range(3):
    inp_event = slayer.io.tensor_to_event(input[i].cpu().data.numpy().reshape(2, 34, 34, -1))
    out_event = slayer.io.tensor_to_event(output_label[i].cpu().data.numpy().reshape(1, 12, -1))
    inp_anim = inp_event.anim(plt.figure(figsize=(5, 5)), frame_rate=240)
    out_anim = out_event.anim(plt.figure(figsize=(10, 5)), frame_rate=240)
    inp_anim.save(f'{trained_folder}/inp-{i}.gif', animation.PillowWriter(fps=24), dpi=300)
    out_anim.save(f'{trained_folder}/out1-{i}.gif', animation.PillowWriter(fps=24), dpi=300)

# set threshold for test 2
for layer in net.feature_extraction_block:
    layer.neuron.threshold = threshold_2
for layer in net.label_classification_block:
    layer.neuron.threshold = threshold_2
output_label, output_task = net(input.to(device))
for i in range(3):
    out_event = slayer.io.tensor_to_event(output_label[i].cpu().data.numpy().reshape(1, 12, -1))
    out_anim = out_event.anim(plt.figure(figsize=(10, 5)), frame_rate=240)
    out_anim.save(f'{trained_folder}/out2-{i}.gif', animation.PillowWriter(fps=24), dpi=300)



