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


class Youtube_test(DownloadableDataset, SplitableDataset):

    """
    Youtube_test
    Binary dataset : 1 if fire, 0 if not.
    """
    

    def __init__(
        self,
        path: str = os.path.join("data", "Youtube_test"),
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
        self.name = "Youtube_test"
        SplitableDataset.__init__(
            self=self,
            train_percentage=train_percentage,
            test_percentage=test_percentage,
            stratify=self.csv["database_target"].tolist()
        )
        
        self.SAMPLE_RATE = 44100
        self.SOUND_DURATION = 5
    
    def downloadYoutube_test(self):

        print("[ DB ] Downloading Youtube_test...")

        # First, download FSC22 in data/fsc22/fsc22 :
        url_youtube_test = "https://centralesupelec-my.sharepoint.com/:u:/g/personal/erwin_deng_student-cs_fr/EU8pwnKNv_xGjxPEOplYGNMBxErS1TYfZb0aZXEXXFTZTA?e=pGK2x1"
        ZIP_FILE_NAME = "youtube_test.zip"

        download_file(file_url=url_youtube_test, save_path=ZIP_FILE_NAME)
        unpack_archive(ZIP_FILE_NAME, os.path.join(self.path, "youtube_test"))
        os.remove(ZIP_FILE_NAME)
        

        dir_path = os.path.join(self.path, "youtube_test", "Test_dataset")

        for className in ["fire", "notfire"]:
            class_path = os.path.join(dir_path, className)
            for dbName in os.listdir(class_path):
                newAudioID = 0
                dbPath = os.path.join(class_path, dbName)
                if os.path.isfile(dbPath):
                    continue
                for filename in os.listdir(dbPath):
                    filepath = os.path.join(dbPath, filename)

                    if not os.path.isfile(filepath):
                        continue
                    
                    description = className
                    target = 0
                    if className == "fire":
                        target = 1
                    
                    extension = os.path.splitext(filename)[1]
                    if extension != ".wav":
                        raise Exception(f"{filename} from Youtube_test isn't a wav file")

                    new_filename = f"YoutubeTest{dbName}_{newAudioID}{extension}"
                    new_path = os.path.join(self.path, "audio", new_filename)
                    shutil.copyfile(filepath, new_path)

                    new_row = [new_filename, target, f'YoutubeTest{dbName}', f"YoutubeTest{dbName}_{target}", description]
                    self.csv.loc[len(self.csv)] = new_row

                    newAudioID += 1

    def download(self):
        self._make_dirs()

        # We create the audio folder (in data/fsc22/audio), that will contain all the audio files
        os.mkdir(os.path.join(self.path, "audio"))

        # We create a new csv that will store all the informations
        self.csv = pd.DataFrame(columns=["filename", "target", "database", "database_target", "description"])

        self.downloadYoutube_test()

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
    
