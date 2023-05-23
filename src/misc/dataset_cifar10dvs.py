import os
import sys
import struct
import glob
import numpy as np
from multiprocessing import Pool

from torch.utils.data import Dataset
import torch
import pickle

import lava.lib.dl.slayer as slayer


# The functions used in this file to download the dataset are based on
# code from the keras library. Specifically, from the following file:
# https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/utils/data_utils.py


def augment(event):
    x_shift = 16
    y_shift = 16
    theta = 10
    xjitter = np.random.randint(2 * x_shift) - x_shift
    yjitter = np.random.randint(2 * y_shift) - y_shift
    ajitter = (np.random.rand() - 0.5) * theta / 180 * 3.141592654
    sin_theta = np.sin(ajitter)
    cos_theta = np.cos(ajitter)
    event.x = event.x * cos_theta - event.y * sin_theta + xjitter
    event.y = event.x * sin_theta + event.y * cos_theta + yjitter
    return event


class CIFAR10DVSDataset(Dataset):

    def __init__(
            self,
            train=True,
            path='data/CIFAR10-DVS',
            sub_path='/data',
            input_size=128,
            output_size=48,
            time_steps=300,
            codification='time',
            keep_value=True,
            transform=None
    ):
        super(CIFAR10DVSDataset, self).__init__()

        self.data_path = path + sub_path
        self.time_steps = time_steps

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

        self.input_size = input_size
        self.output_size = output_size

        self.transform = transform

        if train:
            data_index_filename = self.data_path + '/train_data_index.pkl'
            label_filename = self.data_path + '/train_label.pkl'
        else:
            data_index_filename = self.data_path + '/test_data_index.pkl'
            label_filename = self.data_path + '/test_label.pkl'

        if not os.path.exists(data_index_filename):

            os.makedirs(self.data_path, exist_ok=True)

            raw_sample_files = glob.glob(f'{path}/*/*.aedat')
            all_labels = np.array([self.label_names[filename.split('/')[-2]] for filename in raw_sample_files])

            print("Preparing data")
            for idx, sample_file in enumerate(raw_sample_files):
                with open(sample_file, 'rb') as input_file:
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

                x_address = (x_address * output_size / input_size).astype('int64')
                y_address = (y_address * output_size / input_size).astype('int64')

                times = np.arange(number_events)
                if codification == 'time':
                    time_packets = time_packets - time_packets[0]
                    time_bins = np.linspace(0, time_packets[-1] + 1, num=self.time_steps + 1)
                    times = np.digitize(time_packets, time_bins) - 1
                elif codification == 'event':
                    elements_per_bin = int((number_events+self.time_steps-1)/self.time_steps)
                    times = (np.arange(number_events)/elements_per_bin).astype(int)

                coo = [[], [], [], []]

                coo[0].extend(polarity.tolist())
                coo[1].extend(x_address.tolist())
                coo[2].extend(y_address.tolist())
                coo[3].extend(times.tolist())

                i = torch.LongTensor(coo)
                v = torch.FloatTensor(np.ones(len(coo[0])))

                data_tensor = torch.sparse_coo_tensor(i, v, torch.Size([2, output_size, output_size, time_steps]))
                data_tensor = data_tensor.to_dense()

                if not keep_value:
                    data_tensor = (data_tensor > 0.5).float()

                data_event = slayer.io.tensor_to_event(data_tensor)

                with open(self.data_path + '/%d.pt' % idx, 'wb') as file:
                    pickle.dump(data_event, file)

                # torch.save(data_tensor, self.data_path + '/%d.pt' % idx)

                # data_indices = torch.nonzero(data_tensor).t()
                # torch.save(data_indices, self.data_path + '/%d.pt' % idx)

                # data_values = data_tensor[data_indices[0], data_indices[1], data_indices[2], data_indices[3]]
                # data_values = data_values.reshape(1, -1)
                # torch.save(torch.cat((data_indices, data_values), 0), self.data_path + '/%d.pt' % idx)

                if (idx % 100) == 0:
                    print(idx)

            print("Selecting training and testing sets")
            train_data_index = []
            test_data_index = []
            for i in range(10):
                class_train_index = np.arange(len(raw_sample_files))[all_labels == i][:900]
                class_test_index = np.arange(len(raw_sample_files))[all_labels == i][900:]
                train_data_index.extend(class_train_index.tolist())
                test_data_index.extend(class_test_index.tolist())

            with open(self.data_path + '/train_data_index.pkl', 'wb') as file:
                pickle.dump(train_data_index, file)

            with open(self.data_path + '/test_data_index.pkl', 'wb') as file:
                pickle.dump(test_data_index, file)

            train_label = all_labels[train_data_index]
            test_label = all_labels[test_data_index]

            with open(self.data_path + '/train_label.pkl', 'wb') as file:
                pickle.dump(train_label.tolist(), file)

            with open(self.data_path + '/test_label.pkl', 'wb') as file:
                pickle.dump(test_label.tolist(), file)
            
            if train:
                self.data = train_data_index
                self.labels = train_label
            else:
                self.data = test_data_index
                self.labels = test_label

        else:
            with open(data_index_filename, 'rb') as file:
                self.data = pickle.load(file)   

            with open(label_filename, 'rb') as file:
                self.labels = pickle.load(file)

        print("loaded", len(self.data), " samples")

    def __getitem__(self, idx):

        data_index = self.data[idx]
        label = self.labels[idx]

        # coo = torch.load(self.data_path + '/%d.pt' % data_index)
        # v = torch.FloatTensor(np.ones(len(coo[0])))
        #
        # X_batch = torch.sparse_coo_tensor(coo, v, torch.Size([2, 32, 32, self.sample_length])).to_dense()
        # X_batch = torch.load(self.data_path + '/%d.pt' % data_index)

        with open(self.data_path + '/%d.pt' % data_index, 'rb') as file:
            event = pickle.load(file)

        if self.transform is not None:
            event = self.transform(event)

        # spike = event.fill_tensor(
        #     torch.zeros(2, self.output_size, self.output_size, self.time_steps),
        #     sampling_time=1,
        # )

        # Event to tensor
        spike_tensor = torch.zeros(2, self.output_size, self.output_size, self.time_steps)

        t_start = 0
        x_event = np.round(event.x).astype(int)
        c_event = np.round(event.c).astype(int)
        t_event = np.round(event.t / 1).astype(int) - t_start
        payload = event.p
        binning_mode = 'SUM'

        y_event = np.round(event.y).astype(int)
        valid_ind = np.argwhere(
            (x_event < spike_tensor.shape[2])
            & (y_event < spike_tensor.shape[1])
            & (c_event < spike_tensor.shape[0])
            & (t_event < spike_tensor.shape[3])
            & (x_event >= 0)
            & (y_event >= 0)
            & (c_event >= 0)
            & (t_event >= 0)
        )

        if binning_mode.upper() == 'OR':
            spike_tensor[
                c_event[valid_ind],
                y_event[valid_ind],
                x_event[valid_ind],
                t_event[valid_ind]
            ] = payload[valid_ind]
        elif binning_mode.upper() == 'SUM':
            spike_tensor[
                c_event[valid_ind],
                y_event[valid_ind],
                x_event[valid_ind],
                t_event[valid_ind]
            ] += payload[valid_ind]
        else:
            raise Exception(
                'Unsupported binning_mode. It was {binning_mode}'
            )

        y_batch = torch.tensor(label)

        return spike_tensor, y_batch

    def __len__(self):
        return len(self.data)
