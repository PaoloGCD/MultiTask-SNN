import os
import pickle
import torchvision
import numpy as np

import torch
from torch.utils.data import Dataset


def image2spike_train(x, number_steps=300, max_rate=20):
    batch_size = x.shape[0]
    sample_size = x.shape[1]
    firing_times = []
    units_fired = []
    for sample in range(batch_size):
        if sample % 1000 == 0:
            print(sample)
        sample_firing_times = []
        sample_units_fired = []
        for value in range(sample_size):
            if value > 0:
                binomial_dist = 1 - np.random.binomial(1, float(1000. - x[sample, value] * max_rate) / 1000.0, size=(number_steps,))
                times = np.where(binomial_dist == 1)[0]
                unit = [value for x in times]
                sample_firing_times.extend(times)
                sample_units_fired.extend(unit)
        firing_times.append(sample_firing_times)
        units_fired.append(sample_units_fired)

    return firing_times, units_fired


class MNISTDataset(Dataset):

    def __init__(
            self, path='data',
            train=True,
            sample_length=300,
            max_frequency=20,
            transform=None, download=True
    ):
        super(MNISTDataset, self).__init__()

        self.sample_length = sample_length

        self.cache_dir = path

        if train:
            spiking_path = os.path.join(self.cache_dir, 'spiking_train.txt')
        else:
            spiking_path = os.path.join(self.cache_dir, 'spiking_test.txt')

        if not os.path.exists(spiking_path):
            if train:
                dataset = torchvision.datasets.MNIST(self.cache_dir, train=True, transform=None, target_transform=None,
                                                     download=download)
            else:
                dataset = torchvision.datasets.MNIST(self.cache_dir, train=False, transform=None, target_transform=None,
                                                     download=download)

            # Standardize data
            x_data = np.array(dataset.data, dtype=np.float)
            x_data = x_data.reshape(x_data.shape[0], -1) / 255

            y_data = np.array(dataset.targets, dtype=np.int)

            x_firing_times, x_units_fired = image2spike_train(x_data, sample_length, max_frequency)

            spiking_data = {"firing_times": x_firing_times, "units_fired": x_units_fired, "labels": y_data.tolist(),
                            "num_units": x_data.shape[1]}

            with open(spiking_path, "wb") as fp:
                pickle.dump(spiking_data, fp)

        with open(spiking_path, "rb") as fp:
            spiking_data = pickle.load(fp)

        self.firing_times = spiking_data["firing_times"]
        self.units_fired = spiking_data["units_fired"]
        self.labels_ = np.array(spiking_data["labels"], dtype=np.int)

        self.number_units = spiking_data["num_units"]

        self.transform = transform

    def __getitem__(self, idx):
        times = self.firing_times[idx]
        units = self.units_fired[idx]

        if self.transform is not None:
            units = self.transform(units)

        coo = [[], []]

        coo[0].extend(units)
        coo[1].extend(times)

        i = torch.LongTensor(coo)
        v = torch.FloatTensor(np.ones(len(coo[0])))

        X_batch = torch.sparse_coo_tensor(i, v, torch.Size([self.number_units, self.sample_length])).to_dense()
        y_batch = torch.tensor(self.labels_[idx])

        return X_batch, y_batch, y_batch % 2 + 10, (y_batch < 5) + 12

    def __len__(self):
        return len(self.firing_times)
