# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier:  BSD-3-Clause

"""Multitask CUBA-LIF layer blocks

Built on Intel's Lava-dl implementation https://github.com/lava-nc/lava-dl
@author: Paolo G. Cachi"""

import numpy as np
import torch

from lava.lib.dl.slayer.axon import delay
from lava.lib.dl.slayer.axon import Delay
from lava.lib.dl.slayer.synapse import layer as synapse

import src.misc.neuron_cuba_multitask as neuron_cuba_multitask


class AbstractCuba(torch.nn.Module):
    """Abstract block class for Current Based Leaky Integrator neuron. This
    should never be instantiated on it's own.
    """
    def __init__(self, *args, **kwargs):
        super(AbstractCuba, self).__init__(*args, **kwargs)
        if self.neuron_params is not None:
            self.neuron = neuron_cuba_multitask.Neuron(**self.neuron_params)
        delay = kwargs['delay'] if 'delay' in kwargs.keys() else False
        self.delay = Delay(max_delay=62) if delay is True else None
        del self.neuron_params


class AbstractDense(torch.nn.Module):
    """Abstract dense block class. This should never be instantiated on its own.

    Parameters
    ----------
    neuron_params : dict, optional
        a dictionary of neuron parameter. Defaults to None.
    in_neurons : int
        number of input neurons.
    out_neurons : int
        number of output neurons.
    weight_scale : int, optional
        weight initialization scaling. Defaults to 1.
    weight_norm : bool, optional
        flag to enable weight normalization. Defaults to False.
    pre_hook_fx : optional
        a function pointer or lambda that is applied to synaptic weights before
        synaptic operation. None means no transformation. Defaults to None.
    delay : bool, optional
        flag to enable axonal delay. Defaults to False.
    delay_shift : bool, optional
        flag to simulate spike propagation delay from one layer to next.
        Defaults to True.
    mask : bool array, optional
        boolean synapse mask that only enables relevant synapses. None means no
        masking is applied. Defaults to None.
    count_log : bool, optional
        flag to return event count log. If True, an additional value of average
        event rate is returned. Defaults to False.
    """
    def __init__(
        self, neuron_params, in_neurons, out_neurons,
        weight_scale=1, weight_norm=False, pre_hook_fx=None,
        delay=False, delay_shift=True, mask=None, count_log=False,
    ):
        super(AbstractDense, self).__init__()
        # neuron parameters
        self.neuron_params = neuron_params
        # synapse parameters
        self.synapse_params = {
            'in_neurons': in_neurons,
            'out_neurons': out_neurons,
            'weight_scale': weight_scale,
            'weight_norm': weight_norm,
            'pre_hook_fx': pre_hook_fx,
        }

        self.count_log = count_log

        if mask is None:
            self.mask = None
        else:
            self.register_buffer(
                'mask',
                mask.reshape(mask.shape[0], mask.shape[1], 1, 1, 1)
            )

        # These variables must be initialized by another abstract function
        self.neuron = None
        self.synapse = None
        self.delay = None
        self.delay_shift = delay_shift

    def forward(self, x):
        """
        """
        if self.mask is not None:
            if self.synapse.complex is True:
                self.synapse.real.weight.data *= self.mask
                self.synapse.imag.weight.data *= self.mask
            else:
                self.synapse.weight.data *= self.mask

        z = self.synapse(x)
        x = self.neuron(z)
        if self.delay_shift is True:
            x = delay(x, 1)
        if self.delay is not None:
            x = self.delay(x)

        if self.count_log is True:
            return x, torch.mean(x > 0)
        else:
            return x

    @property
    def shape(self):
        """Shape of the block.
        """
        return self.neuron.shape

    def export_hdf5(self, handle):
        """Hdf5 export method for the block.

        Parameters
        ----------
        handle : file handle
            hdf5 handle to export block description.
        """
        def weight(s):
            return s.pre_hook_fx(
                s.weight, descale=True
            ).reshape(s.weight.shape[:2]).cpu().data.numpy()

        def delay(d):
            return torch.floor(d.delay).flatten().cpu().data.numpy()

        # dense descriptors
        handle.create_dataset(
            'type', (1, ), 'S10', ['dense'.encode('ascii', 'ignore')]
        )
        handle.create_dataset('shape', data=np.array(self.neuron.shape))
        handle.create_dataset('inFeatures', data=self.synapse.in_channels)
        handle.create_dataset('outFeatures', data=self.synapse.out_channels)

        if self.synapse.weight_norm_enabled:
            self.synapse.disable_weight_norm()
        if hasattr(self.synapse, 'imag'):   # complex synapse
            handle.create_dataset(
                'weight/real',
                data=weight(self.synapse.real)
            )
            handle.create_dataset(
                'weight/imag',
                data=weight(self.synapse.imag)
            )
        else:
            handle.create_dataset('weight', data=weight(self.synapse))

        # bias
        has_norm = False
        if hasattr(self.neuron, 'norm'):
            if self.neuron.norm is not None:
                has_norm = True
        if has_norm is True:
            handle.create_dataset(
                'bias',
                data=self.neuron.norm.bias.cpu().data.numpy().flatten()
            )

        # delay
        if self.delay is not None:
            handle.create_dataset('delay', data=delay(self.delay))

        # neuron
        for key, value in self.neuron.device_params.items():
            handle.create_dataset(f'neuron/{key}', data=value)
        if has_norm is True:
            if hasattr(self.neuron.norm, 'weight_exp'):
                handle.create_dataset(
                    'neuron/weight_exp',
                    data=self.neuron.norm.weight_exp
                )


class Dense(AbstractCuba, AbstractDense):
    def __init__(self, *args, **kwargs):
        super(Dense, self).__init__(*args, **kwargs)
        self.synapse = synapse.Dense(**self.synapse_params)
        if 'pre_hook_fx' not in kwargs.keys():
            self.synapse.pre_hook_fx = self.neuron.quantize_8bit
        del self.synapse_params