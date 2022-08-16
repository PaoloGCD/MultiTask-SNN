
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Multitask assistant utility for automatically load network from network
description.

Built on Intel's Lava-dl implementation https://github.com/lava-nc/lava-dl
@author: Paolo G. Cachi"""

import torch


class Assistant:
    """Assistant that bundles training, validation and testing workflow.

    Parameters
    ----------
    net : torch.nn.Module
        network to train.
    error : object or lambda
        an error object or a lambda function that evaluates error.
        It is expected to take ``(output, target)`` | ``(output, label)``
        as it's argument and return a scalar value.
    optimizer : torch optimizer
        the learning optimizer.
    stats : slayer.utils.stats
        learning stats logger. If None, stats will not be logged.
        Defaults to None.
    classifier : slayer.classifier or lambda
        classifier object or lambda function that takes output and
        returns the network prediction. None means regression mode.
        Classification steps are bypassed.
        Defaults to None.

    Attributes
    ----------
    net
    error
    optimizer
    stats
    classifier
    device : torch.device or None
        the main device memory where network is placed. It is not at start and
        gets initialized on the first call.
    """
    def __init__(
        self,
        net, error, optimizer, loss_rate=0,
        stats=None, classifier=None
    ):
        self.net = net
        self.error = error
        self.optimizer = optimizer
        self.classifier = classifier
        self.stats = stats
        self.device = None
        self.loss_rate = loss_rate

    def reduce_lr(self, factor=10 / 3):
        """Reduces the learning rate of the optimizer by ``factor``.

        Parameters
        ----------
        factor : float
            learning rate reduction factor. Defaults to 10/3.

        Returns
        -------

        """
        for param_group in self.optimizer.param_groups:
            print('\nLearning rate reduction from', param_group['lr'])
            param_group['lr'] /= factor

    def train(self, input_data, target_label, target_task):
        """Training assistant.

        Parameters
        ----------
        input_data : torch tensor
            input tensor.
        target_label : torch tensor
            ground truth or label for classification output.
        target_task : torch tensor
            ground truth or label for task output.

        Returns
        -------
        output
            network's output.

        """
        self.net.train()

        if self.device is None:
            for p in self.net.parameters():
                self.device = p.device
                break
        device = self.device

        input_data = input_data.to(device)
        target_label = target_label.to(device)
        target_task = target_task.to(device)

        output_label, output_task = self.net(input_data)

        loss_label = self.error(output_label, target_label)
        loss_task = self.error(output_task, target_task)

        loss = (1-self.loss_rate)*loss_label + self.loss_rate*loss_task

        if self.stats is not None:
            self.stats.training.num_samples += input_data.shape[0]
            self.stats.training.loss_sum[0] += loss_label.cpu().data.item() * output_label.shape[0]
            self.stats.training.loss_sum[1] += loss_task.cpu().data.item() * output_task.shape[0]
            if self.classifier is not None:
                self.stats.training.correct_samples[0] += torch.sum(torch.eq(self.classifier(output_label), target_label)).cpu().data.item()
                self.stats.training.correct_samples[1] += torch.sum(torch.eq(self.classifier(output_task), target_task)).cpu().data.item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return output_label, output_task

    def test(self, input_data, target_label, target_task, test_number):
        """Testing assistant.

        Parameters
        ----------
        input_data : torch tensor
            input tensor.
        target_label : torch tensor
            ground truth or label for classification block.
        target_task : torch tensor
            ground truth or label for task block.
        test_number : int
            number of performed test

        Returns
        -------
        output
            network's output.
        count : optional
            spike count if ``count_log`` is enabled

        """
        self.net.eval()

        if self.device is None:
            for p in self.net.parameters():
                self.device = p.device
                break
        device = self.device

        with torch.no_grad():
            input_data = input_data.to(device)
            target_label = target_label.to(device)
            target_task = target_task.to(device)

            output_label, output_task = self.net(input_data)

            loss_label = self.error(output_label, target_label)
            loss_task = self.error(output_task, target_task)

            if self.stats is not None:  # loss
                self.stats.testing[test_number].num_samples += input_data.shape[0]
                self.stats.testing[test_number].loss_sum[0] += loss_label.cpu().data.item() * output_label.shape[0]
                self.stats.testing[test_number].loss_sum[1] += loss_task.cpu().data.item() * output_task.shape[0]
                if self.classifier is not None:   # classification
                    self.stats.testing[test_number].correct_samples[0] += torch.sum(torch.eq(self.classifier(output_label), target_label)).cpu().data.item()
                    self.stats.testing[test_number].correct_samples[1] += torch.sum(torch.eq(self.classifier(output_task), target_task)).cpu().data.item()

            return output_label, output_task
