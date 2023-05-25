import os
import struct
import numpy as np

from torch.utils.data import Dataset
import torch
import pickle

import lava.lib.dl.slayer as slayer


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


class DVSGesture128Dataset(Dataset):

    def __init__(
            self,
            train=True,
            path='data/DVS-Gesture128',
            sub_path='/data',
            input_size=128,
            output_size=48,
            time_steps=300,
            codification='time',
            keep_value=True,
            transform=None
    ):
        super(DVSGesture128Dataset, self).__init__()
        self.processed_data_path = path + sub_path
        self.raw_data_path = path + '/raw_data'
        self.time_steps = time_steps

        self.bytes_per_event = 36
        self.retina_size_x = 128
        self.x_mask = int('1fff', 16)
        self.y_mask = int('1fff', 16)
        self.pol_mask = int('1', 16)
        self.x_shift = 17
        self.y_shift = 2
        self.pol_shift = 1

        self.input_size = input_size
        self.output_size = output_size

        self.transform = transform

        if train:
            processed_samples_filename = self.processed_data_path + '/train_data_samples.pkl'
            processed_labels_filename = self.processed_data_path + '/train_labels.pkl'
            raw_samples_filename = self.raw_data_path + '/train_raw_samples.pkl'
            raw_labels_filename = self.raw_data_path + '/train_raw_labels.pkl'
            all_sample_files = path + '/DvsGesture/trials_to_train.txt'
        else:
            processed_samples_filename = self.processed_data_path + '/test_data_samples.pkl'
            processed_labels_filename = self.processed_data_path + '/test_labels.pkl'
            raw_samples_filename = self.raw_data_path + '/test_raw_samples.pkl'
            raw_labels_filename = self.raw_data_path + '/test_raw_labels.pkl'
            all_sample_files = path + '/DvsGesture/trials_to_test.txt'

        if not os.path.exists(processed_samples_filename):

            if not os.path.exists(raw_samples_filename):
                print("Creating raw data samples")
                os.makedirs(self.raw_data_path, exist_ok=True)

                with open(all_sample_files, 'r') as all_sample_files:
                    all_sample_files_list = all_sample_files.read().splitlines()
                number_files = len(all_sample_files_list)

                sample_counter = 0
                all_labels_list = []
                all_samples_list = []
                for idx, sample_file in enumerate(all_sample_files_list):

                    if sample_file == '':
                        continue
                    print('Processing raw sample [', idx, '/', number_files, ']')
                    with open(path + '/DvsGesture/' + sample_file, 'rb') as sample_file_data:
                        for _ in range(5):
                            _ = sample_file_data.readline()

                        x_address = []
                        y_address = []
                        time_stamps = []
                        polarity = []

                        while True:
                            header = sample_file_data.read(28)
                            if not header or len(header) == 0:
                                break

                            # read header
                            e_type = struct.unpack('H', header[0:2])[0]
                            e_source = struct.unpack('H', header[2:4])[0]
                            e_size = struct.unpack('I', header[4:8])[0]
                            e_offset = struct.unpack('I', header[8:12])[0]
                            e_tsoverflow = struct.unpack('I', header[12:16])[0]
                            e_capacity = struct.unpack('I', header[16:20])[0]
                            e_number = struct.unpack('I', header[20:24])[0]
                            e_valid = struct.unpack('I', header[24:28])[0]

                            data_length = e_capacity * e_size
                            data = sample_file_data.read(data_length)
                            counter = 0

                            if e_type == 1:
                                while data[counter:counter + e_size]:
                                    aer_data = struct.unpack('I', data[counter:counter + 4])[0]
                                    timestamp = struct.unpack('I', data[counter + 4:counter + 8])[0] | e_tsoverflow << 31
                                    x = (aer_data >> 17) & 0x00007FFF
                                    y = (aer_data >> 2) & 0x00007FFF
                                    pol = (aer_data >> 1) & 0x00000001
                                    counter = counter + e_size
                                    x_address.append(x)
                                    y_address.append(y)
                                    time_stamps.append(timestamp)
                                    polarity.append(pol)
                            else:
                                # non-polarity event packet, not implemented
                                pass

                        x_address = np.asarray(x_address)
                        y_address = np.asarray(y_address)
                        time_stamps = np.asarray(time_stamps)
                        polarity = np.asarray(polarity)

                        events_file = path + '/DvsGesture/' + sample_file[:-6] + '_labels.csv'
                        events_csv = np.loadtxt(events_file, dtype=np.uint32, delimiter=',', skiprows=1)

                        labels = events_csv[:, 0] - 1
                        all_labels_list.extend(labels.tolist())

                        for i in range(labels.shape[0]):
                            time_init = events_csv[i, 1]
                            time_end = events_csv[i, 2]

                            sample_index = (time_stamps >= time_init) * (time_stamps < time_end)

                            sample = {
                                'x': x_address[sample_index],
                                'y': y_address[sample_index],
                                't': time_stamps[sample_index] - time_init,
                                'p': polarity[sample_index]
                            }

                            if train:
                                with open(self.raw_data_path + '/train_%d.pt' % sample_counter, 'wb') as file:
                                    pickle.dump(sample, file)
                                all_samples_list.append('/train_%d.pt' % sample_counter)
                            else:
                                with open(self.raw_data_path + '/test_%d.pt' % sample_counter, 'wb') as file:
                                    pickle.dump(sample, file)
                                all_samples_list.append('/test_%d.pt' % sample_counter)

                            sample_counter += 1

                with open(raw_samples_filename, 'wb') as file:
                    pickle.dump(all_samples_list, file)

                with open(raw_labels_filename, 'wb') as file:
                    pickle.dump(all_labels_list, file)

            print("Creating processed data samples")
            os.makedirs(self.processed_data_path, exist_ok=True)

            with open(raw_samples_filename, 'rb') as all_raw_samples_file:
                raw_samples = pickle.load(all_raw_samples_file)
            number_samples = len(raw_samples)

            with open(raw_labels_filename, 'rb') as all_raw_labels_file:
                raw_labels = pickle.load(all_raw_labels_file)

            processed_samples_list = []
            for idx, sample_filename in enumerate(raw_samples):
                print('Processing raw sample [', idx, '/', number_samples, ']')
                with open(self.raw_data_path + sample_filename, 'rb') as raw_sample_file:
                    raw_sample = pickle.load(raw_sample_file)

                x_address = (raw_sample['x'] * output_size / input_size).astype('int64')
                y_address = (raw_sample['y'] * output_size / input_size).astype('int64')
                polarity = raw_sample['p']

                time_stamps = raw_sample['t']
                if codification == 'time':
                    time_bins = np.linspace(0, time_stamps.max() + 1, num=self.time_steps + 1)
                    times = np.digitize(time_stamps, time_bins) - 1
                elif codification == 'event':
                    elements_per_bin = int((time_stamps.shape[0]+self.time_steps-1)/self.time_steps)
                    times = (np.arange(time_stamps.shape[0])/elements_per_bin).astype(int)
                else:
                    times = time_stamps

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

                with open(self.processed_data_path + sample_filename, 'wb') as file:
                    pickle.dump(data_event, file)

                processed_samples_list.append(sample_filename)

            with open(processed_samples_filename, 'wb') as file:
                pickle.dump(processed_samples_list, file)

            with open(processed_labels_filename, 'wb') as file:
                pickle.dump(raw_labels, file)

        with open(processed_samples_filename, 'rb') as file:
            self.all_samples = pickle.load(file)

        with open(processed_labels_filename, 'rb') as file:
            self.all_labels = pickle.load(file)

        print("loaded", len(self.all_samples), " samples")

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):

        data_filename = self.all_samples[idx]
        label = self.all_labels[idx]

        with open(self.processed_data_path + data_filename, 'rb') as file:
            event = pickle.load(file)

        if self.transform is not None:
            event = self.transform(event)

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
