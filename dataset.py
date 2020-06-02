import torch

import numpy as np
import os
import pickle
import imageio
import gzip
import random
from test_helper import get_metadata, get_details_from_path


# NOTE: don't transfer data to CUDA in dataset (if we need to, use memory pinning)


class SimpleDataset(torch.utils.data.Dataset):
    """
    Simple, all_purpose dataset class
    X: Requires path to text file containing a list of filenames for x inputs
    Y: Requires path to either a text or a pkl containing all y data
    Loads data from disk on-demand if given file lists, else load at init
    """

    def __init__(self, x_file, y_file=None, x_transform=None, y_transform=None, use_cache=False):
        """
        Parameters
        - x_paths: list of file paths to input images
        - y_paths: path to label files
        - y_files: list of file paths to label files
        - x_transform: callable image preprocessor for x
        - y_transform: callable image preprocessor for y

        NOTE: the channel-first 
        """

        self.x_transform = x_transform
        self.y_transform = y_transform
        self.use_cache = use_cache
        self.cache_dict = {}

        # read x file
        if x_file.endswith(".npy"):
            self.x_arr = self._load_numpy(x_file)
        elif x_file.endswith(".pkl"):
            self.x_arr = self._load_pickle(x_file)
        else:
            with open(x_file, 'r') as f:
                self.x_arr = np.array(f.readlines())

        # read y file
        if y_file is not None:
            if y_file.endswith(".npy"):
                self.y_arr = self._load_numpy(y_file)
            elif y_file.endswith(".pkl"):
                self.y_arr = self._load_pickle(y_file)
            else:
                with open(y_file, 'r') as f:
                    self.y_arr = np.array(f.readlines())
        else:
            self.y_arr = None

        self.metadata = get_metadata(self.x_arr)

    def __len__(self):
        return len(self.x_arr)

    def __getitem__(self, index):
        """
        Returns images and optionally labels at index
        """

        if self.use_cache and index in self.cache_dict:
            return self.cache_dict[index]

        # get x sample
        x = self.x_arr[index]
        x = self._get_data_from_path(x)
        if self.x_transform is not None:
            x = self.x_transform(x)

        # get y sample if requested
        if self.y_arr is not None:
            y = self.y_arr[index]
            y = self._get_data_from_path(y)
            if self.y_transform is not None:
                y = self.y_transform(y)
            if self.use_cache:
                self.cache_dict[index] = (x, y, self._is_ed(self.x_arr[index]))
            return x, y, self._is_ed(self.x_arr[index])
        else:
            if self.use_cache:
                self.cache_dict[index] = (x, self._is_ed(self.x_arr[index]))
            return x, self.x_arr[index], self._is_ed(self.x_arr[index])  # Return X and its path

    def _get_data_from_path(self, path):
        if isinstance(path, str):  # load from file
            x = path.strip()
            if ".npy" in x:
                x = self._load_numpy(x)
            elif ".pkl" in x:  # pickle file, assume ndarray
                x = self._load_pickle(x)
            else:  # assume file in image format
                x = np.array(imageio.imread(x))
        return x

    def _shuffle(self):
        shuffled_idx = np.random.permutation(self.__len__())
        self.x_paths = self.x_paths[shuffled_idx]
        if self.y_arr is not None:
            self.y_arr = self.y_arr[shuffled_idx]

    def _is_ed(self, path):
        p, f, s = get_details_from_path(path)
        return self.metadata[p][f]['type'] == 'ED'

    @staticmethod
    def _load_numpy(path):
        path = path.strip()
        f = gzip.open(path) if path.endswith(".gz") else open(path, 'rb')
        return np.load(f)

    @staticmethod
    def _load_pickle(path):
        path = path.strip()
        f = gzip.open(path) if path.endswith(".gz") else open(path, 'rb')
        ret = pickle.load(f)
        assert isinstance(ret, np.ndarray), "Expected numpy.ndarray"
        return ret
