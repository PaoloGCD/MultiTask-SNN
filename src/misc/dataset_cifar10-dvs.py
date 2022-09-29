import struct
import glob
import numpy as np

from torch.utils.data import Dataset
import torch


# The functions used in this file to download the dataset are based on
# code from the keras library. Specifically, from the following file:
# https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/utils/data_utils.py


class CIFAR10DVSDataset(Dataset):

    def __init__(
            self, path='data/CIFAR10-DVS',
            sample_length=300
    ):
        super(CIFAR10DVSDataset, self).__init__()

        self.path = path
        self.sample_length = sample_length

        self.samples = glob.glob(f'{path}/*/*.aedat')

        self.label_names = {
            'airplane': 0,
            'automobile': 1,
            'bird': 2,
            'cat': 3,
            'deer': 4,
            'dog': 5,
            'frog': 6,
            'horse': 7,
            'ship': 8,
            'truck': 9
        }

        self.bytes_per_event = 8
        self.retina_size_x = 128
        self.x_mask = int('fe', 16)
        self.y_mask = int('7f00', 16)
        self.pol_mask = int('1', 16)
        self.x_shift = 1
        self.y_shift = 8

    def __getitem__(self, idx):

        filename = self.samples[idx]
        label = self.label_names[filename.split('/')[-2]]

        with open(filename, 'rb') as input_file:
            for _ in range(5):
                head_line = input_file.readline()
            data_bytes = input_file.read()

        number_events = int(len(data_bytes) / self.bytes_per_event)
        number_packets = number_events * 2
        data_int = struct.unpack('>%di' % number_packets, data_bytes)

        address_packets = np.asarray(data_int[::2])
        time_packets = np.asarray(data_int[1::2])

        x_address = self.retina_size_x - 1 - np.right_shift(np.bitwise_and(address_packets, self.x_mask), self.x_shift)
        y_address = np.right_shift(np.bitwise_and(address_packets, self.y_mask), self.y_shift)
        polarity = 1 - np.bitwise_and(address_packets, self.pol_mask)

        time_packets = time_packets - time_packets[0]
        time_bins = np.linspace(0, time_packets[-1], num=self.sample_length)
        times = np.digitize(time_packets, time_bins) - 1

        coo = [[], [], [], []]

        coo[0].extend(polarity.tolist())
        coo[1].extend(x_address.tolist())
        coo[2].extend(y_address.tolist())
        coo[3].extend(times.tolist())

        i = torch.LongTensor(coo)
        v = torch.FloatTensor(np.ones(len(coo[0])))

        X_batch = torch.sparse_coo_tensor(i, v, torch.Size([2, 128, 128, self.sample_length])).to_dense()
        y_batch = torch.tensor(label)

        return X_batch, y_batch

    def __len__(self):
        return len(self.samples)
