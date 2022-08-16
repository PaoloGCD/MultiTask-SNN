# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Modified module for managing, visualizing, and displaying learning statistics.

Built on Intel's Lava-dl implementation https://github.com/lava-nc/lava-dl
@author: Paolo G. Cachi"""

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

    def __init__(self, number_output_blocks=1):
        self.num_samples = 0
        self.output_blocks = number_output_blocks
        self.correct_samples = [0 for _ in range(self.output_blocks)]
        self.loss_sum = [0 for _ in range(self.output_blocks)]
        self.min_loss = [None for _ in range(self.output_blocks)]
        self.max_accuracy = [None for _ in range(self.output_blocks)]
        self.best_loss = [False for _ in range(self.output_blocks)]
        self.best_accuracy = [False for _ in range(self.output_blocks)]
        self.loss_log = []
        self.accuracy_log = []

    def reset(self):
        """Reset stat."""
        self.loss_sum = [0 for _ in range(self.output_blocks)]
        self.correct_samples = [0 for _ in range(self.output_blocks)]
        self.num_samples = 0

    @property
    def loss(self):
        """Current loss."""
        if self.num_samples > 0:
            return [loss/self.num_samples for loss in self.loss_sum]
        else:
            return [0 for _ in self.loss_sum]

    @property
    def accuracy(self):
        """Current accuracy."""
        if self.num_samples > 0:
            return [samples/self.num_samples for samples in self.correct_samples]
        else:
            return [0 for _ in self.correct_samples]

    def update(self):
        """Update stat."""
        current_loss = self.loss
        self.loss_log.append(current_loss)
        for i, loss in enumerate(current_loss):
            if self.min_loss[i] is None or loss < self.min_loss[i]:
                self.min_loss[i] = loss
                self.best_loss[i] = True
            else:
                self.best_loss[i] = False

        current_accuracy = self.accuracy
        self.accuracy_log.append(current_accuracy)
        for i, accuracy in enumerate(current_accuracy):
            if self.max_accuracy[i] is None or accuracy > self.max_accuracy[i]:
                self.max_accuracy[i] = accuracy
                self.best_accuracy[i] = True
            else:
                self.best_accuracy[i] = False

    @property
    def valid_loss_log(self):
        """ """
        return self.loss_log != [[None]*self.output_blocks] * len(self.loss_log)

    @property
    def valid_accuracy_log(self):
        """ """
        return self.accuracy_log != [[None]*self.output_blocks] * len(self.accuracy_log)

    def __str__(self):
        """String method.
        """
        if self.loss is [None for _ in self.loss_sum]:
            return ''

        loss_string = 'loss = ['

        for i, loss in enumerate(self.loss):
            loss_string += f'{loss:11.5f}'
            if self.min_loss[i] is not None:
                loss_string += f'(min = {self.min_loss[i]:11.5f}), '
            else:
                loss_string += f', '
        loss_string = loss_string[:-2] + ']'

        accuracy_string = ''

        if self.accuracy is not [None for _ in self.loss_sum]:
            accuracy_string = 'accuracy = ['

            for i, accuracy in enumerate(self.accuracy):
                accuracy_string += f'{accuracy:7.5f}'
                if self.max_accuracy[i] is not None:
                    accuracy_string += f'(max = {self.max_accuracy[i]}), '
                else:
                    accuracy_string += ', '

            accuracy_string = accuracy_string[:-2] + ']'

        return loss_string + " "*4 + accuracy_string + " "*4


class LearningStatsHandler:
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
    def __init__(self, number_tasks=1, number_training_blocks=1):
        self.number_tasks = number_tasks
        self.number_training_blocks = number_training_blocks

        self.lines_printed = 0

        self.training = LearningStat(number_training_blocks)
        self.validation = [LearningStat(number_training_blocks) for i in range(number_tasks)]
        self.testing = [LearningStat(number_training_blocks) for i in range(number_tasks)]

    def update(self):
        """Update all the stats. Typically called at the end of epoch."""
        self.training.update()
        self.training.reset()

        for validation in self.validation:
            validation.update()
            validation.reset()

        for testing in self.testing:
            testing.update()
            testing.reset()

    def new_line(self):
        """Forces stats printout on new line."""
        self.lines_printed = 0

    def __str__(self):
        """String summary of stats.
        """
        val_str = ''
        for i, validation in enumerate(self.validation):
            string = str(validation)
            if len(string) > 0:
                val_str += ' | Valid%d ' % i + string

        test_str = ''
        for i, test in enumerate(self.testing):
            string = str(test)
            if len(string) > 0:
                test_str += ' | Test%d  ' % i + string

        return f'Train {str(self.training)}{val_str}{test_str}'

    def print(
        self, epoch,
        iteration=None, time_elapsed=None, header=None, data_loader=None
    ):
        """Dynamic print method for stats.

        Parameters
        ----------
        epoch : int
            current epoch
        iteration : int or None
            iteration number in epoch. Defaults to None.
        time_elapsed : float or None
            elapsed time. Defaults to None.
        header : list or None
            List of strings to print before statistics. It can be used to
            customize additional prints. None means no header.
            Defaults to None.
        data_loader : torch.dataloader or None
            torch dataloader. If not None, shows progress in the epoch.
            Defaults to None.

        """
        # move cursor up by self.lines_printed
        print(f'\033[{self.lines_printed}A')

        self.lines_printed = 1
        epoch_str = f'Epoch {epoch:4d}'
        iter_str = '' if iteration is None else f': i = {iteration:5d} '
        if time_elapsed is None:
            profile_str = ''
        else:
            profile_str = f', {time_elapsed*1000:12.4f} ms elapsed'

        if data_loader is None or iteration is None:
            progress_str = ''
        else:
            iter_sig_digits = int(np.ceil(np.log10(len(data_loader.dataset))))
            progress_str = f'{iteration*data_loader.batch_size:{iter_sig_digits}}' \
                + f'/{len(data_loader.dataset)} '\
                  f'({100.0*iteration/len(data_loader):.0f}%)'
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

    # def plot(self, figures=(1, 2), fig_size=None, path=None):
    #     """Plots the training curves.
    #
    #     Parameters
    #     ----------
    #     figures : tuple of ints
    #         figures to plot loss and accuracy. Defaults to (1, 2).
    #     fig_size : tuple of ints or None
    #         custom width and height of the figure. None means default size.
    #         Defaults to None.
    #     path : str
    #         If not None, saves the plot to the path specified.
    #         Defaults to None.
    #
    #     """
    #     def figure_init(fig_id):
    #         """
    #         """
    #         plt.figure(fig_id, figsize=fig_size)
    #         plt.cla()
    #         return True
    #
    #     loss_plot_exists = False
    #     if self.training.valid_loss_log:
    #         loss_plot_exists = figure_init(figures[0])
    #         plt.semilogy(self.training.loss_log, label=['Training%d' % d for d in range(self.number_training_blocks)])
    #     if self.validation.valid_loss_log:
    #         if loss_plot_exists is False:
    #             loss_plot_exists = figure_init(figures[0])
    #         plt.semilogy(self.validation.loss_log, label='Validation')
    #     if self.testing.valid_loss_log:
    #         if loss_plot_exists is False:
    #             loss_plot_exists = figure_init(figures[0])
    #         plt.semilogy(self.testing.loss_log, label='Testing')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.legend()
    #     if path is not None:
    #         plt.savefig(path + 'loss.png')
    #
    #     if self.training.valid_accuracy_log is False and \
    #         self.validation.valid_accuracy_log is False and \
    #             self.testing.valid_accuracy_log is False:
    #         return
    #     acc_plot_exists = False
    #     if self.training.valid_accuracy_log:
    #         acc_plot_exists = figure_init(figures[1])
    #         plt.plot(self.training.accuracy_log, label='Training')
    #     if self.validation.valid_accuracy_log:
    #         if acc_plot_exists is False:
    #             acc_plot_exists = figure_init(figures[1])
    #         plt.plot(self.validation.accuracy_log, label='Validation')
    #     if self.testing.valid_accuracy_log:
    #         if acc_plot_exists is False:
    #             acc_plot_exists = figure_init(figures[1])
    #         plt.plot(self.testing.accuracy_log, label='Testing')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Accuracy')
    #     plt.legend()
    #     if path is not None:
    #         plt.savefig(path + 'accuracy.png')

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
                for i in range(self.number_training_blocks):
                    header += ' Train-%d     ' % i

            for val_idx, validation in enumerate(self.validation):
                if validation.valid_loss_log:
                    for block_idx in range(self.number_training_blocks):
                        header += ' Valid_%d-%d   ' % (val_idx, block_idx)

            for test_idx, testing in enumerate(self.testing):
                if testing.valid_loss_log:
                    for block_idx in range(self.number_training_blocks):
                        header += ' Test_%d-%d    ' % (test_idx, block_idx)

            loss.write(f'#{header}\r\n')

            entry = ''
            for idx in range(len(self.training.loss_log)):
                for block_idx in range(self.number_training_blocks):
                    training_value = self.training.loss_log[idx][block_idx]
                    entry += '' if training_value is None else f'{training_value:12.6f} '

                for val_idx, validation in enumerate(self.validation):
                    for block_idx in range(self.number_training_blocks):
                        validation_value = validation.loss_log[idx][block_idx]
                        entry += '' if validation_value is None else f'{validation_value:12.6f} '

                for test_idx, test in enumerate(self.testing):
                    for block_idx in range(self.number_training_blocks):
                        test_value = test.loss_log[idx][block_idx]
                        entry += '' if test_value is None else f'{test_value:12.6f} '

                loss.write(f'{entry}\r\n')

        if self.training.valid_accuracy_log is False:
            return

        for validation in self.validation:
            if validation.valid_accuracy_log is False:
                return

        for test in self.testing:
            if test.valid_accuracy_log is False:
                return

        with open(path + 'accuracy.txt', 'wt') as accuracy:
            header = ''
            if self.training.valid_accuracy_log:
                for i in range(self.number_training_blocks):
                    header += ' Train-%d     ' % i

            for val_idx, validation in enumerate(self.validation):
                if validation.valid_accuracy_log:
                    for block_idx in range(self.number_training_blocks):
                        header += ' Valid_%d-%d   ' % (val_idx, block_idx)

            for test_idx, testing in enumerate(self.testing):
                if testing.valid_accuracy_log:
                    for block_idx in range(self.number_training_blocks):
                        header += ' Test_%d-%d    ' % (test_idx, block_idx)

            accuracy.write(f'#{header}\r\n')

            entry = ''
            for idx in range(len(self.training.accuracy_log)):
                for block_idx in range(self.number_training_blocks):
                    training_value = self.training.accuracy_log[idx][block_idx]
                    entry += '' if training_value is None else f'{training_value:12.6f} '

                for val_idx, validation in enumerate(self.validation):
                    for block_idx in range(self.number_training_blocks):
                        validation_value = validation.accuracy_log[idx][block_idx]
                        entry += '' if validation_value is None else f'{validation_value:12.6f} '

                for test_idx, test in enumerate(self.testing):
                    for block_idx in range(self.number_training_blocks):
                        test_value = test.accuracy_log[idx][block_idx]
                        entry += '' if test_value is None else f'{test_value:12.6f} '

                accuracy.write(f'{entry}\r\n')

    def load(self, path=''):
        """
        """
        pass
