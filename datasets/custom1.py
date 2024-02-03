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
        self.name = "CustomDB1"
        SplitableDataset.__init__(
            self=self,
            train_percentage=train_percentage,
            test_percentage=test_percentage,
            stratify=self.csv["database_target"].tolist()
        )
    
    def downloadESC50(self):

        print("[ DB ] Downloading ESC50...")

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
    