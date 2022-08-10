
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
        net, error, optimizer,
        stats=None, classifier=None
    ):
        self.net = net
        self.error = error
        self.optimizer = optimizer
        self.classifier = classifier
        self.stats = stats
        self.device = None

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

    def train(self, input, target_label, target_task, loss_rate):
        """Training assistant.

        Parameters
        ----------
        input : torch tensor
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

        input = input.to(device)
        target_label = target_label.to(device)
        target_task = target_task.to(device)

        output_label, output_task = self.net(input)

        loss_label = self.error(output_label, target_label)
        loss_task = self.error(output_task, target_task)

        loss = (1-loss_rate)*loss_label + loss_rate*loss_task

        if self.stats is not None:
            self.stats.training.num_samples += input.shape[0]
            self.stats.training.loss_sum += loss.cpu().data.item() * output_label.shape[0]
            self.stats.training.loss_classifier_sum += loss_label.cpu().data.item() * output_label.shape[0]
            self.stats.training.loss_task_sum += loss_task.cpu().data.item() * output_label.shape[0]
            if self.classifier is not None:   # classification
                self.stats.training.correct_classifier_samples += torch.sum(self.classifier(output_label) == target_label).cpu().data.item()
                self.stats.training.correct_task_samples += torch.sum(self.classifier(output_task) == target_task).cpu().data.item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return output_label, output_task

    def test(self, input, target_label, target_task, test_number):
        """Testing assistant.

        Parameters
        ----------
        input : torch tensor
            input tensor.
        target : torch tensor
            ground truth or label.

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
            input = input.to(device)
            target_label = target_label.to(device)
            target_task = target_task.to(device)

            output_label, output_task = self.net(input)

            loss_label = self.error(output_label, target_label)
            loss_task = self.error(output_task, target_task)

            loss = loss_label + loss_task

            if self.stats is not None and test_number == 1:
                self.stats.testing1.num_samples += input.shape[0]
                self.stats.testing1.loss_sum += loss.cpu().data.item() * output_label.shape[0]
                self.stats.testing1.loss_classifier_sum += loss_label.cpu().data.item() * output_label.shape[0]
                self.stats.testing1.loss_task_sum += loss_task.cpu().data.item() * output_label.shape[0]
                if self.classifier is not None:   # classification
                    self.stats.testing1.correct_classifier_samples += torch.sum(
                        self.classifier(output_label) == target_label).cpu().data.item()
                    self.stats.testing1.correct_task_samples += torch.sum(
                        self.classifier(output_task) == target_task).cpu().data.item()
            if self.stats is not None and test_number == 2:
                self.stats.testing2.num_samples += input.shape[0]
                self.stats.testing2.loss_sum += loss.cpu().data.item() * output_label.shape[0]
                self.stats.testing2.loss_classifier_sum += loss_label.cpu().data.item() * output_label.shape[0]
                self.stats.testing2.loss_task_sum += loss_task.cpu().data.item() * output_label.shape[0]
                if self.classifier is not None:  # classification
                    self.stats.testing2.correct_classifier_samples += torch.sum(
                        self.classifier(output_label) == target_label).cpu().data.item()
                    self.stats.testing2.correct_task_samples += torch.sum(
                        self.classifier(output_task) == target_task).cpu().data.item()

            return output_label, output_task
