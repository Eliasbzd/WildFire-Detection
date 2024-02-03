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
        self.name = "CustomDB2"
        SplitableDataset.__init__(
            self=self,
            train_percentage=train_percentage,
            test_percentage=test_percentage,
            stratify=self.csv["database_target"].tolist()
        )
        
        self.SAMPLE_RATE = 44100
        self.SOUND_DURATION = 5
    
    def downloadFSC22(self):

        print("[ DB ] Downloading FSC22...")

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

    def downloadKaggleFire(self):
        """
        The dataset was found here https://www.kaggle.com/datasets/forestprotection/forest-wild-fire-sound-dataset
        It is probably from a youtube video like this one: https://www.youtube.com/watch?v=2ya2drfb4rA
        The author might be Michael Ghelfi : https://www.patreon.com/michaelghelfi/about
        Preprocessing was made in firesoundnormalisation.py
        All files last 5 sec, 44 100 Hz (220 500 values)
        """

        print("[ DB ] Downloading KaggleFire...")

        # First, download Esc50 in data/CustomDataset1/esc50 :
        url = "https://centralesupelec-my.sharepoint.com/:u:/g/personal/erwin_deng_student-cs_fr/EdBm8fljIsVFnao77EnSGMkBKEmil5dZjkiGCzeuIEqyPg?e=BEdAE4"
        ZIP_FILE_NAME = "temp-kagglefire.zip"

        download_file(url, ZIP_FILE_NAME)
        unpack_archive(ZIP_FILE_NAME, self.path)
        os.remove(ZIP_FILE_NAME)

        newAudioID = 0

        dir_path = os.path.join(self.path, "KaggleFire")

        for filename in os.listdir(dir_path):
            
            extension = os.path.splitext(filename)[1]
            if extension != ".wav":
                raise Exception(f"{filename} from KaggleFire isn't a wav file")

            old_path = os.path.join(dir_path, filename)

            new_filename = f"KaggleFire_{newAudioID}{extension}"
            new_path = os.path.join(self.path, "audio", new_filename)
            shutil.copyfile(old_path, new_path)

            new_row = [new_filename, 1, 'KaggleFire', f"KaggleFire_1", ""]
            self.csv.loc[len(self.csv)] = new_row

            newAudioID += 1

    def downloadYellowstone(self):
        """
        The dataset was extracted from fires at the Yellowstone National park.
        Long audio recording were found here: https://www.nps.gov/yell/learn/photosmultimedia/sounds-fire.htm and https://acousticatlas.org
        
        Preprocessing was made in firesoundnormalisation.py
        All files last 5 sec, 44 100 Hz (220 500 values)
        """

        print("[ DB ] Downloading Yellowstone...")

        # First, download Esc50 in data/CustomDataset1/esc50 :
        url = "https://centralesupelec-my.sharepoint.com/:u:/g/personal/erwin_deng_student-cs_fr/EQ8X3N3YsFpItGxAiQroTvQBQsJ8zEeeiPJ1G49wFP1vpg?e=ZOnD6E"
        ZIP_FILE_NAME = "temp-yellowstone.zip"

        download_file(url, ZIP_FILE_NAME)
        unpack_archive(ZIP_FILE_NAME, self.path)
        os.remove(ZIP_FILE_NAME)

        newAudioID = 0

        dir_path = os.path.join(self.path, "Yellowstone")

        for filename in os.listdir(dir_path):
            
            extension = os.path.splitext(filename)[1]
            if extension != ".wav":
                raise Exception(f"{filename} from Yellowstone isn't a wav file")

            old_path = os.path.join(dir_path, filename)

            new_filename = f"Yellowstone_{newAudioID}{extension}"
            new_path = os.path.join(self.path, "audio", new_filename)
            shutil.copyfile(old_path, new_path)

            new_row = [new_filename, 1, 'Yellowstone', f"Yellowstone_1", ""]
            self.csv.loc[len(self.csv)] = new_row

            newAudioID += 1

    def download(self):
        self._make_dirs()

        # We create the audio folder (in data/CustomDataset1/audio), that will contain all the audio files
        os.mkdir(os.path.join(self.path, "audio"))

        # We create a new csv that will store all the informations
        self.csv = pd.DataFrame(columns=["filename", "target", "database", "database_target", "description"])

        self.downloadFSC22()
        self.downloadKaggleFire()
        self.downloadYellowstone()

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
    
