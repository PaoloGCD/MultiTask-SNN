# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Module for managing, visualizing, and displaying learning statistics."""

import numpy as np
import matplotlib.pyplot as plt


class LearningStat:
    """Learning stat manager

    Attributes
    ----------
    loss_sum: float
        accumulated loss sum.
    correct_samples: int
        accumulated correct samples.
    num_samples: int
        number of samples accumulated.
    min_loss: float
        best loss recorded.
    max_accuracy: float
        best accuracy recorded.
    loss_log: list of float
        log of loss stats.
    accuracy_log: list of float
        log of accuracy stats.
    best_loss: bool
        does current epoch have best loss? It is updated only
        after `stats.update()`.
    best_accuracy: bool
        does current epoch have best accuracy? It is updated only
        after `stats.update()`.
    """
    def __init__(self):
        self.loss_sum = 0
        self.correct_samples = 0
        self.num_samples = 0
        self.min_loss = None
        self.max_accuracy = None
        self.loss_log = []
        self.accuracy_log = []
        self.best_loss = False
        self.best_accuracy = False

    def reset(self):
        """Reset stat."""
        self.loss_sum = 0
        self.correct_samples = 0
        self.num_samples = 0

    @property
    def loss(self):
        """Current loss."""
        if self.num_samples > 0:
            return self.loss_sum / self.num_samples
        else:
            return 0

    @property
    def accuracy(self):
        """Current accuracy."""
        if self.num_samples > 0 and self.correct_samples > 0:
            return self.correct_samples / self.num_samples
        else:
            return 0

    @property
    def valid_loss_log(self):
        """ """
        return self.loss_log != [None] * len(self.loss_log)

    @property
    def valid_accuracy_log(self):
        """ """
        return self.accuracy_log != [None] * len(self.accuracy_log)

    def update(self):
        """Update stat."""
        self.loss_log.append(self.loss)
        if self.min_loss is None or self.loss < self.min_loss:
            self.min_loss = self.loss
            self.best_loss = True
        else:
            self.best_loss = False

        self.accuracy_log.append(self.accuracy)
        if self.max_accuracy is None or self.accuracy > self.max_accuracy:
            self.max_accuracy = self.accuracy
            self.best_accuracy = True
        else:
            self.best_accuracy = False

    def __str__(self):
        """String method.
        """
        if self.loss is None:
            return ''
        elif self.accuracy is None:
            if self.min_loss is None:
                return f'loss = {self.loss:11.5f}'
            else:
                return f'loss = {self.loss:11.5f} '\
                    f'(min = {self.min_loss:11.5f})'
        else:
            if self.min_loss is None:
                if self.max_accuracy is None:
                    return f'loss = {self.loss:11.5f}{" "*24}'\
                        f'accuracy = {self.accuracy:7.5f}'
            else:
                return f'loss = {self.loss:11.5f} '\
                    f'(min = {self.min_loss:11.5f}){" "*4}'\
                    f'accuracy = {self.accuracy:7.5f} '\
                    f'(max = {self.max_accuracy:7.5f})'


class LearningStats:
    """Manages training, validation and testing stats.

    Attributes
    ----------
    training : LearningStat
        `LearningStat` object to manage training statistics.
    testing : LearningStat
        `LearningStat` object to manage testing statistics.
    validation : LearningStat
        `LearningStat` object to manage validation statistics.
    """
    def __init__(self):
        self.lines_printed = 0
        self.training = LearningStat()
        self.testing1 = LearningStat()
        self.testing2 = LearningStat()
        self.validation = LearningStat()

    def update(self):
        """Update all the stats. Typically called at the end of epoch."""
        self.training.update()
        self.training.reset()
        self.testing1.update()
        self.testing1.reset()
        self.testing2.update()
        self.testing2.reset()
        self.validation.update()
        self.validation.reset()

    def new_line(self):
        """Forces stats printout on new line."""
        self.lines_printed = 0

    def __str__(self):
        """String summary of stats.
        """
        val_str = str(self.validation)
        if len(val_str) > 0:
            val_str = ' | Valid ' + val_str

        test1_str = str(self.testing1)
        if len(test1_str) > 0:
            test_str = ' | Test1  ' + test1_str

        test2_str = str(self.testing2)
        if len(test2_str) > 0:
            test_str = ' | Test1  ' + test2_str

        return f'Train {str(self.training)}{val_str}{test1_str}{test2_str}'

    def print(
        self, epoch,
        iter=None, time_elapsed=None, header=None, dataloader=None
    ):
        """Dynamic print method for stats.

        Parameters
        ----------
        epoch : int
            current epoch
        iter : int or None
            iteration number in epoch. Defaults to None.
        time_elapsed : float or None
            elapsed time. Defaults to None.
        header : list or None
            List of strings to print before statistics. It can be used to
            customize additional prints. None means no header.
            Defaults to None.
        dataloader : torch.dataloader or None
            torch dataloader. If not None, shows progress in the epoch.
            Defaults to None.

        """
        # move cursor up by self.lines_printed
        print(f'\033[{self.lines_printed}A')

        self.lines_printed = 1
        epoch_str = f'Epoch {epoch:4d}'
        iter_str = '' if iter is None else f': i = {iter:5d} '
        if time_elapsed is None:
            profile_str = ''
        else:
            profile_str = f', {time_elapsed*1000:12.4f} ms elapsed'

        if dataloader is None or iter is None:
            progress_str = ''
        else:
            iter_sig_digits = int(np.ceil(np.log10(len(dataloader.dataset))))
            progress_str = f'{iter*dataloader.batch_size:{iter_sig_digits}}' \
                + f'/{len(dataloader.dataset)} '\
                  f'({100.0*iter/len(dataloader):.0f}%)'
            iter_str = ': '

        if header is not None:
            for h in header:
                print('\033[2K' + str(h))
                self.lines_printed += 1

        print(epoch_str + iter_str + progress_str + profile_str + " " * 8)
        self.lines_printed += 1
        for line in self.__str__().split('| '):
            print(line)
            self.lines_printed += 1

    def plot(self, figures=(1, 2), figsize=None, path=None):
        """Plots the training curves.

        Parameters
        ----------
        figures : tuple of ints
            figures to plot loss and accuracy. Defaults to (1, 2).
        figsize : tuple of ints or None
            custom width and height of the figure. None means default size.
            Defaults to None.
        path : str
            If not None, saves the plot to the path specified.
            Defaults to None.

        """
        def figure_init(fig_id):
            """
            """
            plt.figure(fig_id, figsize=figsize)
            plt.cla()
            return True

        loss_plot_exists = False
        if self.training.valid_loss_log:
            loss_plot_exists = figure_init(figures[0])
            plt.semilogy(self.training.loss_log, label='Training')
        if self.validation.valid_loss_log:
            if loss_plot_exists is False:
                loss_plot_exists = figure_init(figures[0])
            plt.semilogy(self.validation.loss_log, label='Validation')
        if self.testing1.valid_loss_log:
            if loss_plot_exists is False:
                loss_plot_exists = figure_init(figures[0])
            plt.semilogy(self.testing1.loss_log, label='Testing')
        if self.testing2.valid_loss_log:
            if loss_plot_exists is False:
                loss_plot_exists = figure_init(figures[0])
            plt.semilogy(self.testing2.loss_log, label='Testing')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        if path is not None:
            plt.savefig(path + 'loss.png')

        if self.training.valid_accuracy_log is False and \
            self.validation.valid_accuracy_log is False and \
                self.testing1.valid_accuracy_log is False and \
                    self.testing2.valid_accuracy_log is False:
            return
        acc_plot_exists = False
        if self.training.valid_accuracy_log:
            acc_plot_exists = figure_init(figures[1])
            plt.plot(self.training.accuracy_log, label='Training')
        if self.validation.valid_accuracy_log:
            if acc_plot_exists is False:
                acc_plot_exists = figure_init(figures[1])
            plt.plot(self.validation.accuracy_log, label='Validation')
        if self.testing1.valid_accuracy_log:
            if acc_plot_exists is False:
                acc_plot_exists = figure_init(figures[1])
            plt.plot(self.testing1.accuracy_log, label='Testing')
        if self.testing2.valid_accuracy_log:
            if acc_plot_exists is False:
                acc_plot_exists = figure_init(figures[1])
            plt.plot(self.testing2.accuracy_log, label='Testing')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        if path is not None:
            plt.savefig(path + 'accuracy.png')

    def save(self, path=''):
        """Saves learning stats to file

        Parameters
        ----------
        path : str
            Folder path to save the stats. Defaults to ''.

        """
        with open(path + 'loss.txt', 'wt') as loss:
            header = ''
            if self.training.valid_loss_log:
                header += ' Train       '
            if self.validation.valid_loss_log:
                header += ' Valid       '
            if self.testing1.valid_loss_log:
                header += ' Test1       '
            if self.testing2.valid_loss_log:
                header += ' Test2       '

            loss.write(f'#{header}\r\n')

            for tr, va, te1, te2 in zip(
                self.training.loss_log,
                self.validation.loss_log,
                self.testing1.loss_log,
                self.testing2.loss_log
            ):
                entry = '' if tr is None else f'{tr:12.6f} '
                entry += '' if va is None else f'{va:12.6f} '
                entry += '' if te1 is None else f'{te1:12.6f} '
                entry += '' if te2 is None else f'{te2:12.6f} '
                loss.write(f'{entry}\r\n')

        if self.training.valid_accuracy_log is False and \
            self.validation.valid_accuracy_log is False and \
                self.testing1.valid_accuracy_log is False and \
                    self.testing2.valid_accuracy_log is False:
            return
        with open(path + 'accuracy.txt', 'wt') as accuracy:
            header = ''
            if self.training.valid_loss_log:
                header += ' Train       '
            if self.validation.valid_loss_log:
                header += ' Valid       '
            if self.testing1.valid_loss_log:
                header += ' Test1        '
            if self.testing2.valid_loss_log:
                header += ' Test2        '

            accuracy.write(f'#{header}\r\n')

            for tr, va, te1, te2 in zip(
                self.training.accuracy_log,
                self.validation.accuracy_log,
                self.testing1.accuracy_log,
                self.testing2.accuracy_log
            ):
                entry = '' if tr is None else f'{tr:12.6f} '
                entry += '' if va is None else f'{va:12.6f} '
                entry += '' if te1 is None else f'{te1:12.6f} '
                entry += '' if te2 is None else f'{te2:12.6f} '
                accuracy.write(f'{entry}\r\n')

    def load(self, path=''):
        """
        """
        pass
