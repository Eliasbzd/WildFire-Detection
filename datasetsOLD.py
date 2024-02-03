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
            print(f"Deleting old dataset {path}...")
            shutil.rmtree(path)

        if download or not os.path.exists(path):
            print(f"Downloading dataset in {path}...")
            self.download()

    @abstractmethod
    def download(self):
        raise NotImplementedError

    def _make_dirs(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)


class ESCDataset(DownloadableDataset, SplitableDataset):
    def __init__(
        self,
        path: str = "data/esc50",
        download: bool = False,
        transformBinary: bool = True,
        train_percentage: float = 0.7,
        test_percentage: float = 0.15,
    ) -> None:
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



class CustomDataset1(DownloadableDataset, SplitableDataset):

    """
    CustomDataset1, which takes sounds from ESC50 and other databases.
    Binary dataset : 1 if fire, 0 if not.
    """

    def __init__(
        self,
        path: str = os.path.join("data", "CustomDataset1"),
        download: bool = False,
        train_percentage: float = 0.7,
        test_percentage: float = 0.15,
    ) -> None:
        """
        Args:
            path: the path to where the dataset is or should be stored
            download: whether to download the data
        """
        DownloadableDataset.__init__(self=self, path=path, download=download)

        self.csv = pd.read_csv(os.path.join(path, "desc.csv"))

        SplitableDataset.__init__(
            self=self,
            train_percentage=train_percentage,
            test_percentage=test_percentage,
            stratify=self.csv["database_target"].tolist()
        )
    
    def downloadESC50(self):

        print("Downloading ESC50...")

        # First, download Esc50 in data/CustomDataset1/esc50 :
        url_esc50 = "https://github.com/karoldvl/ESC-50/archive/master.zip"
        ZIP_FILE_NAME = "temp-esc50.zip"

        urllib.request.urlretrieve(url_esc50, ZIP_FILE_NAME)
        unpack_archive(ZIP_FILE_NAME, self.path)

        os.rename(os.path.join(self.path, "ESC-50-master"), os.path.join(self.path, "esc50"))
        os.remove(ZIP_FILE_NAME)

        df_esc50 = pd.read_csv(os.path.join(self.path, "esc50", "meta", "esc50.csv"))

        newAudioID = 0

        for index, row in df_esc50.iterrows():
            filename = row["filename"]
            description = row["category"]
            target = 0
            if description == "crackling_fire":
                target = 1
            
            extension = os.path.splitext(filename)[1]
            if extension != ".wav":
                raise Exception(f"{filename} from ESC50 isn't a wav file")

            old_path = os.path.join(self.path, "esc50", "audio", filename)

            new_filename = f"ESC50_{newAudioID}{extension}"
            new_path = os.path.join(self.path, "audio", new_filename)
            shutil.copyfile(old_path, new_path)

            new_row = [new_filename, target, 'ESC50', f"ESC50_{target}", description]
            self.csv.loc[len(self.csv)] = new_row

            newAudioID += 1
    
    def downloadKaggleFire(self):
        """
        The dataset was found here https://www.kaggle.com/datasets/forestprotection/forest-wild-fire-sound-dataset
        Preprocessing was made in firesoundnormalisation.py
        All files last 5 sec, 44 100 Hz (220 500 values)
        """

        print("Downloading KaggleFire...")

        # First, download Esc50 in data/CustomDataset1/esc50 :
        url = "http://dengerwin.free.fr/files/PoleIA/KaggleFire.zip"
        ZIP_FILE_NAME = "temp-kagglefire.zip"

        urllib.request.urlretrieve(url, ZIP_FILE_NAME)
        unpack_archive(ZIP_FILE_NAME, self.path)

        os.rename(os.path.join(self.path, "final"), os.path.join(self.path, "kaggleFire"))
        os.remove(ZIP_FILE_NAME)

        newAudioID = 0

        dir_path = os.path.join(self.path, "kaggleFire")

        for filename in os.listdir(dir_path):
            
            extension = os.path.splitext(filename)[1]
            if extension != ".wav":
                raise Exception(f"{filename} from kaggleFire isn't a wav file")

            old_path = os.path.join(dir_path, filename)

            new_filename = f"KaggleFire_{newAudioID}{extension}"
            new_path = os.path.join(self.path, "audio", new_filename)
            shutil.copyfile(old_path, new_path)

            new_row = [new_filename, 1, 'KaggleFire', f"KaggleFire_1", ""]
            self.csv.loc[len(self.csv)] = new_row

            newAudioID += 1

    def download(self):
        self._make_dirs()

        # We create the audio folder (in data/CustomDataset1/audio), that will contain all the audio files
        os.mkdir(os.path.join(self.path, "audio"))

        # We create a new csv that will store all the informations
        self.csv = pd.DataFrame(columns=["filename", "target", "database", "database_target", "description"])

        self.downloadESC50()
        self.downloadKaggleFire()

        self.csv.to_csv(os.path.join(self.path, "desc.csv"), index=False)
    

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


    def _get_sample_label(self, index: int) -> str:
        # Default label
        return self.csv.iloc[index, self.csv.columns.get_loc("target")]



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
        assert sample.shape[1] == 220500

        # Convert to mono, and squeeze the shape, which should be torch.Size([220500]) instead of torch.Size([1, 220500])
        sample = torch.mean(sample, axis=0)

        return sample, label
    

class CustomDataset2(DownloadableDataset, SplitableDataset):

    """
    CustomDataset2, which takes sounds from FSC22.
    Binary dataset : 1 if fire, 0 if not.
    """
    

    def __init__(
        self,
        path: str = os.path.join("data", "CustomDataset2"),
        download: bool = False,
        train_percentage: float = 0.7,
        test_percentage: float = 0.15,
    ) -> None:
        """
        Args:
            path: the path to where the dataset is or should be stored
            download:   If set to False, it won't download the dataset unless the folder doesn't exist. 
                        If the to True, it will delete and download the dataset
        """
        DownloadableDataset.__init__(self=self, path=path, download=download)

        self.csv = pd.read_csv(os.path.join(path, "desc.csv"))

        SplitableDataset.__init__(
            self=self,
            train_percentage=train_percentage,
            test_percentage=test_percentage,
            stratify=self.csv["database_target"].tolist()
        )
        
        self.SAMPLE_RATE = 44100
        self.SOUND_DURATION = 5
    
    def downloadFSC22(self):

        print("Downloading FSC22...")

        # First, download FSC22 in data/CustomDataset2/fsc22 :
        url_fsc22 = "https://centralesupelec-my.sharepoint.com/:u:/g/personal/erwin_deng_student-cs_fr/EbwDb3ys7M1Pntiz2xX9WMwBqwgdBurE5fyIobdMcomc0Q?e=30dQ5o"
        ZIP_FILE_NAME = "fsc22.zip"

        download_file(file_url=url_fsc22, save_path=ZIP_FILE_NAME)
        unpack_archive(ZIP_FILE_NAME, os.path.join(self.path, "fsc22"))
        os.remove(ZIP_FILE_NAME)

        os.rename(os.path.join(self.path, "fsc22", "Audio Wise V1.0-20220916T202003Z-001", "Audio Wise V1.0"), os.path.join(self.path, "fsc22", "audio"))
        os.rename(os.path.join(self.path, "fsc22", "Metadata-20220916T202011Z-001", "Metadata"), os.path.join(self.path, "fsc22", "meta"))
        
        os.rmdir(os.path.join(self.path, "fsc22", "Audio Wise V1.0-20220916T202003Z-001"))
        os.rmdir(os.path.join(self.path, "fsc22", "Metadata-20220916T202011Z-001"))

        df_fsc22 = pd.read_csv(os.path.join(self.path, "fsc22", "meta", "Metadata V1.0 FSC22.csv"), sep=",")

        newAudioID = 0

        for index, row in df_fsc22.iterrows():
            filename = row["Dataset File Name"]
            description = row["Class Name"]
            target = 0
            if description == "Fire":
                target = 1
            
            extension = os.path.splitext(filename)[1]
            if extension != ".wav":
                raise Exception(f"{filename} from FSC22 isn't a wav file")

            old_path = os.path.join(self.path, "fsc22", "audio", filename)

            new_filename = f"FSC22_{newAudioID}{extension}"
            new_path = os.path.join(self.path, "audio", new_filename)
            shutil.copyfile(old_path, new_path)

            new_row = [new_filename, target, 'FSC22', f"FSC22_{target}", description]
            self.csv.loc[len(self.csv)] = new_row

            newAudioID += 1

    def download(self):
        self._make_dirs()

        # We create the audio folder (in data/CustomDataset1/audio), that will contain all the audio files
        os.mkdir(os.path.join(self.path, "audio"))

        # We create a new csv that will store all the informations
        self.csv = pd.DataFrame(columns=["filename", "target", "database", "database_target", "description"])

        self.downloadFSC22()

        self.csv.to_csv(os.path.join(self.path, "desc.csv"), index=False)
    

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


    def _get_sample_label(self, index: int) -> str:
        # Default label
        return self.csv.iloc[index, self.csv.columns.get_loc("target")]



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
        assert sample.shape[1]/sample_rate == self.SOUND_DURATION
        if sample_rate != self.SAMPLE_RATE:
            sample = torchaudio.functional.resample(sample, orig_freq=sample_rate, new_freq=self.SAMPLE_RATE) 
        
        # Convert to mono, and squeeze the shape, which should be torch.Size([220500]) instead of torch.Size([1, 220500])
        sample = torch.mean(sample, axis=0)

        return sample, label
    
