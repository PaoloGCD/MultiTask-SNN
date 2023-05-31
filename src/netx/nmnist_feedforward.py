import sys
import time
import logging

from typing import Tuple
import numpy as np

from src.misc.dataset_nmnist import NMNISTDataset

from lava.magma.core.process.process import AbstractProcess
from lava.magma.core.model.py.model import PyLoihiProcessModel
from lava.magma.core.model.sub.model import AbstractSubProcessModel
from lava.magma.core.process.ports.ports import InPort, OutPort
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType

from lava.magma.core.decorator import implements, requires
from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.resources import CPU, Loihi2NeuroCore

from lava.magma.core.run_configs import Loihi2HwCfg, Loihi2SimCfg
from lava.magma.core.run_conditions import RunSteps

from lava.proc import io

from lava.lib.dl import netx

from lava.utils.system import Loihi2
if Loihi2.is_loihi2_available:
    from lava.proc import embedded_io as eio


# Input adapter #############################################################
class InputAdapter(AbstractProcess):
    """Input adapter process.

    Parameters
    ----------
    shape : tuple of ints
        Shape of input.
    """
    def __init__(self, shape: Tuple[int, ...]) -> None:
        super().__init__(shape=shape)
        self.inp = InPort(shape=shape)
        self.out = OutPort(shape=shape)


@implements(proc=InputAdapter, protocol=LoihiProtocol)
@requires(CPU)
class PyInputAdapterModel(PyLoihiProcessModel):
    """Input adapter model for CPU."""
    inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def run_spk(self):
        self.out.send(self.inp.recv())


@implements(proc=InputAdapter, protocol=LoihiProtocol)
@requires(Loihi2NeuroCore)
class NxInputAdapterModel(AbstractSubProcessModel):
    """Input adapter model for Loihi 2."""
    def __init__(self, proc: AbstractProcess) -> None:
        self.inp: PyInPort = LavaPyType(np.ndarray, np.int32)
        self.out: PyOutPort = LavaPyType(np.ndarray, np.int32)
        shape = proc.proc_params.get('shape')
        self.adapter = eio.spike.PyToNxAdapter(shape=shape)
        proc.inp.connect(self.adapter.inp)
        self.adapter.out.connect(proc.out)


# Output adapter #############################################################
class OutputAdapter(AbstractProcess):
    """Output adapter process.

    Parameters
    ----------
    shape : Tuple[int, ...]
        Shape of output.
    """
    def __init__(self, shape: Tuple[int, ...]) -> None:
        super().__init__(shape=shape)
        self.inp = InPort(shape=shape)
        self.out = OutPort(shape=shape)


@implements(proc=OutputAdapter, protocol=LoihiProtocol)
@requires(CPU)
class PyOutputAdapterModel(PyLoihiProcessModel):
    """Output adapter model for CPU."""
    inp: PyInPort = LavaPyType(PyInPort.VEC_DENSE, float)
    out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, float)

    def run_spk(self):
        self.out.send(self.inp.recv())


@implements(proc=OutputAdapter, protocol=LoihiProtocol)
@requires(Loihi2NeuroCore)
class NxOutputAdapterModel(AbstractSubProcessModel):
    """Output adapter model for Loihi 2."""
    def __init__(self, proc: AbstractProcess) -> None:
        self.inp: PyInPort = LavaPyType(np.ndarray, np.int32)
        self.out: PyOutPort = LavaPyType(np.ndarray, np.int32)
        shape = proc.proc_params.get('shape')
        self.adapter = eio.spike.NxToPyAdapter(shape=shape)
        proc.inp.connect(self.adapter.inp)
        self.adapter.out.connect(proc.out)


class CustomHwRunConfig(Loihi2HwCfg):
    """Custom Loihi2 hardware run config."""
    def __init__(self):
        super().__init__(select_sub_proc_model=True)

    def select(self, proc, proc_models):
        # customize run config
        if isinstance(proc, InputAdapter):
            return NxInputAdapterModel
        if isinstance(proc, OutputAdapter):
            return NxOutputAdapterModel
        return super().select(proc, proc_models)


class CustomSimRunConfig(Loihi2SimCfg):
    """Custom Loihi2 simulation run config."""
    def __init__(self):
        super().__init__(select_tag='fixed_pt')

    def select(self, proc, proc_models):
        # customize run config
        if isinstance(proc, InputAdapter):
            return PyInputAdapterModel
        if isinstance(proc, OutputAdapter):
            return PyOutputAdapterModel
        return super().select(proc, proc_models)


if __name__ == '__main__':

    print("Loading dataset")
    start_program_time = time.time()

    Loihi2.preferred_partition = 'oheogulch'
    loihi2_is_available = Loihi2.is_loihi2_available

    if Loihi2.is_loihi2_available:
        print(f'Running on {Loihi2.partition}')
        compression = io.encoder.Compression.DENSE
    else:
        print("Loihi2 compiler is not available in this system. "
              "Running on CPU backend.")
        compression = io.encoder.Compression.DENSE

    # Get set up parameters
    experiment_number = 0
    parameters_path = "../../params/base_case.xml"
    data_path = "../../data/NMNIST"
    if len(sys.argv) == 3:
        parameters_path = str(sys.argv[1])
        data_path = str(sys.argv[2])
    elif len(sys.argv) == 4:
        parameters_path = str(sys.argv[1])
        data_path = str(sys.argv[2])
        experiment_number = int(sys.argv[3])

    result_path = f'results/Base-case-NMNIST-{experiment_number:02d}'

    print("Loading dataset")
    start_time = time.time()

    # Running parameters
    num_samples = 10
    num_steps = 350
    total_time_steps = num_samples*num_steps

    # Get test samples
    testing_set = NMNISTDataset(train=False, path=data_path, random=True, num_samples=num_samples)

    sample, true_label = testing_set[0]
    sample_shape = sample.shape
    all_samples = np.zeros((sample_shape[0], total_time_steps))
    all_true_labels = np.zeros(num_samples).astype('int')
    for i in range(num_samples):
        sample, true_label = testing_set[i]
        all_samples[:, i*num_steps:(i*num_steps)+sample_shape[1]] = sample
        all_true_labels[i] = true_label

    print("Dataset loaded in", time.time() - start_time)
    print("Building network")
    start_time = time.time()

    net = netx.hdf5.Network(net_config=result_path + '/network.net')

    source = io.source.RingBuffer(data=all_samples)
    sink = io.sink.RingBuffer(shape=net.out.shape, buffer=total_time_steps)
    inp_adapter = InputAdapter(shape=net.inp.shape)
    out_adapter = OutputAdapter(shape=net.out.shape)

    source.s_out.connect(inp_adapter.inp)
    inp_adapter.out.connect(net.inp)
    net.out.connect(out_adapter.inp)
    out_adapter.out.connect(sink.a_in)

    print("Network configured in", time.time() - start_time)
    print("Running...")
    start_time = time.time()

    if loihi2_is_available:
        run_config = CustomHwRunConfig()
    else:
        run_config = CustomSimRunConfig()

    net._log_config.level = logging.INFO
    net.run(condition=RunSteps(num_steps=total_time_steps), run_cfg=run_config)

    print("Run finished in", time.time() - start_time)
    print("Calculating results")
    start_time = time.time()

    all_predicted_labels = np.zeros(num_samples).astype('int')
    output = sink.data.get()
    for i in range(num_samples):
        output_per_sample = output[:, i*num_steps:(i+1)*num_steps]
        output_per_sample_sum = output_per_sample.sum(axis=1)
        predicted_label = output_per_sample_sum.argmax()
        all_predicted_labels[i] = predicted_label

    # print(all_true_labels)
    # print(all_predicted_labels)

    correct_labels = (all_true_labels == all_predicted_labels).astype('int').sum()
    accuracy = correct_labels/num_samples

    print("Results calculated in", time.time() - start_time)
    print("Total time ", time.time() - start_program_time)

    print()
    print("Accuracy", accuracy)

    net.stop()


