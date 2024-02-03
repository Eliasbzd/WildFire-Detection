from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from shutil import unpack_archive
import shutil
import os
import os.path


from torch.utils.data import Dataset, DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split as train_test_split_sk
import pandas as pd
import urllib.request
import torch
import torchaudio
import torchaudio.transforms as T

from mult_to_bin import convert_mult_to_bin
from utils import download_file

from datasets.db_utils import SplitableDataset, DownloadableDataset


class ESCDataset(DownloadableDataset, SplitableDataset):
    def __init__(
        self,
        path: str = "data/esc50",
        download: bool = False,
        transformBinary: bool = True,
        train_percentage: float = 0.7,
        test_percentage: float = 0.15,
    ):
        """
        Args:
            path: the path to where the dataset is or should be stored
            download: whether to download the data
            transformBinary: whether to transform to a binary dataset. 1 if the target is fire, 0 if not
            categories: whether to use ESC-10 or ESC-50
        """
        DownloadableDataset.__init__(self=self, path=path, download=download)
        SplitableDataset.__init__(
            self=self,
            train_percentage=train_percentage,
            test_percentage=test_percentage,
        )

        #if transformBinary: OLD VERSION
        #    convert_mult_to_bin(os.path.join(path, "meta/esc50.csv"), 12, "target")
        self.name = "ESC50"
        self.csv = pd.read_csv(os.path.join(path, "meta/esc50.csv"))

        if transformBinary:
            fires = (self.csv["target"] == 12)
            self.csv.loc[~ fires, "target"] = 0
            self.csv.loc[fires, "target"] = 1

    def download(self):
        """Automatically downloads and extracts the dataset in the desired data directory"""
        self._make_dirs()

        url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
        ZIP_FILE_NAME = "temp-esc50.zip"

        urllib.request.urlretrieve(url, ZIP_FILE_NAME)
        unpack_archive(ZIP_FILE_NAME, os.path.join(self.path, ".."))

        os.rename(os.path.join(self.path, "..", "ESC-50-master"), self.path)
        os.remove(ZIP_FILE_NAME)

    def __len__(self) -> int:
        """Computes the size of the dataset.

        Returns
        -------
        int
            the size of the dataset
        """
        return len(self.csv)

    def _get_wav_file_path(self, index: int) -> str:
        """Returns the path to the wav file corresponding to sample at given index in the csv.

        Parameters
        ----------
        index: int
            the index of the item in the csv annotations filemkdir

        Returns
        -------
        string
            the path to the wav file
        """
        return os.path.join(self.path, "audio", self.csv.iloc[index, 0])

    def _get_sample_label(self, index: int) -> str:
        # Default label
        return self.csv.iloc[index, 2]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gets the dataset item at given index

        Parameters
        ----------
        index: int
            the index number where to look for the item

        Returns
        -------
        int
            a tuple that contains the waveform and the corrsponding label at given index
        """
        wav_path = self._get_wav_file_path(index)
        label = self._get_sample_label(index)
        sample, sample_rate = torchaudio.load(wav_path)
        assert sample_rate == 44100

        # Convert to mono, and squeeze the shape, which should be torch.Size([220500]) instead of torch.Size([1, 220500])
        sample = torch.mean(sample, axis=0)

        return sample, label

    def label_target_to_category(self, label: int) -> str:
        """Returns the category name associated to the given label number.

        Parameters
        ----------
        label: int
            the label target number

        Returns
        -------
        string
            the category
        """
        return self.csv.iloc[label, 2]

    def get_all_labels(self) -> list[torch.Tensor]:
        """Returns all possible labels in this dataset

        Returns
        -------
        list[torch.Tensor]
            a list of all possible labels
        """
        return [x for x in self.csv["target"].unique()]

