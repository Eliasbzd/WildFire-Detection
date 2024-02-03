from abc import ABC, abstractmethod
from dataclasses import dataclass
import shutil
import os
import os.path

import torch
import csv
import pandas as pd
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split as train_test_split_sk

@dataclass
class TrainValidTestDataLoader:
    train: DataLoader
    valid: DataLoader
    test: DataLoader


@dataclass
class TrainValidTestDataset:
    train: Dataset
    valid: Dataset
    test: Dataset

    def into_loaders(self, batch_size: int = 32) -> TrainValidTestDataLoader:
        """Turn the datasets into DataLoaders.

        Parameters  
        ----------
        batch_size: int
            the size of the batches in the dataset
        """
        train_loader = DataLoader(self.train, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(self.valid, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(self.test, batch_size=batch_size, shuffle=True)

        return TrainValidTestDataLoader(
            train=train_loader, valid=valid_loader, test=test_loader
        )


class SplitableDataset(ABC, Dataset):
    def __init__(
        self, train_percentage: float = 0.7, test_percentage: float = 0.15, stratify=None
    ) -> None:
        super().__init__()
        self.train_percentage = train_percentage
        self.test_percentage = test_percentage
        self.stratify = stratify

    def train_test_split(self) -> TrainValidTestDataset:
        """Split the dataset into train and test datasets

        Returns
        -------
        TrainValidTestDataset
            an object holding the train dataset and the test (validation) dataset
        """
        train_size = int(self.train_percentage * len(self))
        test_size = int(self.test_percentage * len(self))
        valid_size = len(self) - train_size - test_size

        if self.stratify == None:
            train_dataset, valid_dataset, test_dataset = random_split(
                self, [train_size, test_size, valid_size]
            )
        else:
            train_idx, test_idx, train_targets, test_targets = train_test_split_sk(list(range(len(self))), self.stratify, test_size=test_size, shuffle=True, stratify=self.stratify)
            train_idx, val_idx, train_targets, val_targets = train_test_split_sk(train_idx, train_targets, train_size=train_size, shuffle=True, stratify=train_targets)
            train_dataset = Subset(self, train_idx)
            valid_dataset = Subset(self, val_idx)
            test_dataset = Subset(self, test_idx)

        return TrainValidTestDataset(
            train=train_dataset, valid=valid_dataset, test=test_dataset
        )


class DownloadableDataset(ABC, Dataset):
    def __init__(self, path: str, download: bool = False):
        self.path = path

        if download and os.path.exists(path):
            print(f"[ DB ] Deleting old dataset {path}...")
            shutil.rmtree(path)

        if download or not os.path.exists(path):
            print(f"[ DB ] Downloading dataset in {path}...")
            self.download()

    @abstractmethod
    def download(self):
        raise NotImplementedError

    def _make_dirs(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

class temporary_test_dataset():
    def __init__(self,
        path: str = os.path.join("Data_small")):
        self.SAMPLE_RATE = 44100
        self.name = path
        self.SOUND_DURATION = 5
        self.path = path

        with open(os.path.join(path, "desc.csv"),'w') as f:
            to_f = [["filename"]]
            for element in os.listdir(path+'/audio/'):
                to_f.append([element])
            writer = csv.writer(f)
            writer.writerows(to_f)
        print('[ DB ] Temporary test database initalised')

        self.csv = pd.read_csv(os.path.join(path, "desc.csv"))
    
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
        return os.path.join(self.path, "audio", self.csv.iloc[index, self.csv.columns.get_loc("filename")])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Gets the dataset item at given index

        Parameters
        ----------
        index: int
            the index number where to look for the item

        Returns
        -------
            the waveform at a given index
        """
        wav_path = self._get_wav_file_path(index)
        sample, sample_rate = torchaudio.load(wav_path,)
        sample = sample[:,0:sample_rate*self.SOUND_DURATION]
        assert sample.shape[1]/sample_rate == self.SOUND_DURATION
        if sample_rate != self.SAMPLE_RATE:
            sample = torchaudio.functional.resample(sample, orig_freq=sample_rate, new_freq=self.SAMPLE_RATE) 

        return (torch.mean(sample, axis=0),0)
