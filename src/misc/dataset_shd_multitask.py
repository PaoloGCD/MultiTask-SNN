import os
import h5py
import urllib.request
import gzip
import shutil
import hashlib
import numpy as np

from six.moves.urllib.error import HTTPError
from six.moves.urllib.error import URLError
from six.moves.urllib.request import urlretrieve

from torch.utils.data import Dataset
import torch


# The functions used in this file to download the dataset are based on
# code from the keras library. Specifically, from the following file:
# https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/utils/data_utils.py


def get_shd_dataset(cache_dir, cache_subdir):
    # The remote directory with the data files
    base_url = "https://zenkelab.org/datasets"

    # Retrieve MD5 hashes from remote
    response = urllib.request.urlopen("%s/md5sums.txt" % base_url)
    data = response.read()
    lines = data.decode('utf-8').split("\n")
    file_hashes = {line.split()[1]: line.split()[0] for line in lines if len(line.split()) == 2}

    # Download the Spiking Heidelberg Digits (SHD) dataset
    files = ["shd_train.h5.gz",
             "shd_test.h5.gz",
             ]
    for fn in files:
        origin = "%s/%s" % (base_url, fn)
        hdf5_file_path = get_and_gunzip(origin, fn, md5hash=file_hashes[fn], cache_dir=cache_dir, cache_subdir=cache_subdir)
        # print("File %s decompressed to:"%(fn))
        print("Available at: %s" % hdf5_file_path)


def get_and_gunzip(origin, filename, md5hash=None, cache_dir=None, cache_subdir=None):
    gz_file_path = get_file(filename, origin, md5_hash=md5hash, cache_dir=cache_dir, cache_subdir=cache_subdir)
    hdf5_file_path = gz_file_path[:-3]
    if not os.path.isfile(hdf5_file_path) or os.path.getctime(gz_file_path) > os.path.getctime(hdf5_file_path):
        print("Decompressing %s" % gz_file_path)
        with gzip.open(gz_file_path, 'r') as f_in, open(hdf5_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return hdf5_file_path


def validate_file(fpath, file_hash, algorithm='auto', chunk_size=65535):
    if (algorithm == 'sha256') or (algorithm == 'auto' and len(file_hash) == 64):
        hasher = 'sha256'
    else:
        hasher = 'md5'

    if str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash):
        return True
    else:
        return False


def _hash_file(fpath, algorithm='sha256', chunk_size=65535):
    if (algorithm == 'sha256') or (algorithm == 'auto' and len(hash) == 64):
        hasher = hashlib.sha256()
    else:
        hasher = hashlib.md5()

    with open(fpath, 'rb') as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b''):
            hasher.update(chunk)

    return hasher.hexdigest()


def get_file(fname,
             origin,
             md5_hash=None,
             file_hash=None,
             cache_subdir='datasets',
             hash_algorithm='auto',
             extract=False,
             archive_format='auto',
             cache_dir=None):
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.data-cache')
    if md5_hash is not None and file_hash is None:
        file_hash = md5_hash
        hash_algorithm = 'md5'
    datadir_base = os.path.expanduser(cache_dir)
    if not os.access(datadir_base, os.W_OK):
        datadir_base = os.path.join('/tmp', '.data-cache')
    datadir = os.path.join(datadir_base, cache_subdir)

    # Create directories if they don't exist
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(datadir, exist_ok=True)

    fpath = os.path.join(datadir, fname)

    download = False
    if os.path.exists(fpath):
        # File found; verify integrity if a hash was provided.
        if file_hash is not None:
            if not validate_file(fpath, file_hash, algorithm=hash_algorithm):
                print('A local file was found, but it seems to be '
                      'incomplete or outdated because the ' + hash_algorithm +
                      ' file hash does not match the original value of ' + file_hash +
                      ' so we will re-download the data.')
                download = True
    else:
        download = True

    if download:
        print('Downloading data from', origin)

        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(origin, fpath)
            except HTTPError as e:
                raise Exception(error_msg.format(origin, e.code, e.msg))
            except URLError as e:
                raise Exception(error_msg.format(origin, e.errno, e.reason))
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(fpath):
                os.remove(fpath)

    return fpath


class SHDDataset(Dataset):
    """NMNIST dataset method
    Parameters
    ----------
    path : str, optional
        path of dataset root, by default 'data'
    train : bool, optional
        train/test flag, by default True
    sample_length : int, optional
        length of sample data, by default 300
    transform : None or lambda or fx-ptr, optional
        transformation method. None means no transform. By default Noney.
    download : bool, optional
        enable/disable automatic download, by default True
    """

    def __init__(
            self, path='data',
            train=True,
            max_time=1, sample_length=300, units=700,
            transform=None, download=True
    ):
        super(SHDDataset, self).__init__()

        self.sample_length = sample_length
        self.max_time = max_time

        self.cache_dir = path
        self.cache_subdir = 'spikes'

        get_shd_dataset(self.cache_dir, self.cache_subdir)

        if train:
            data_file = h5py.File(os.path.join(self.cache_dir, self.cache_subdir, 'shd_train.h5'), 'r')
        else:
            data_file = h5py.File(os.path.join(self.cache_dir, self.cache_subdir, 'shd_test.h5'), 'r')

        self.x_train = data_file['spikes']
        self.y_train = data_file['labels']

        self.labels_ = np.array(self.y_train, dtype=np.int)

        # compute discrete firing times
        self.firing_times = self.x_train['times']
        self.units_fired = self.x_train['units']

        self.time_bins = np.linspace(0, 1.4, num=sample_length)

        self.transform = transform

    def __getitem__(self, idx):
        times = np.digitize(self.firing_times[idx], self.time_bins)
        units = self.units_fired[idx]

        if self.transform is not None:
            units = self.transform(units)

        coo = [[], []]

        coo[0].extend(units)
        coo[1].extend(times)

        i = torch.LongTensor(coo)
        v = torch.FloatTensor(np.ones(len(coo[0])))

        X_batch = torch.sparse_coo_tensor(i, v, torch.Size([700, self.sample_length])).to_dense()
        y_batch = torch.tensor(self.labels_[idx])

        return X_batch, y_batch, y_batch % 10 + 20

    def __len__(self):
        return len(self.firing_times)
