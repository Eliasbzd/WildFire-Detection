import librosa as lb
import torch

import numpy as np

class MFCC_LIB(torch.nn.Module):

    def __init__(
        self,
        n_fft: int = 128,
        window_length = None,
        hop_length: int = 512,
        n_mfcc: int = 16,
        n_mels: int = 128,
        sample_rate: int = 44100,
    ) -> None:
        """Mel-Frequency Cepstral-Coefficient feature-extractor.

        Parameters
        ----------
        n_mels: int, optional
            the number of mel filterbanks
        num_frequencies: int, optional
            the number of frequencies to sample on in the Spectrogram. Defaults to 201.
        window_length: int, optional
            the length of the spectrogram window in the Spectrogram, when using forward, in frames. Defaults to 10.
        hop_size: int, optional
            the distance between two spectrogram windows, when using forward, in ms. Defaults to win_length/2.
        """
        
        super(MFCC_LIB, self).__init__()

        self.n_fft = n_fft
        self.window_length = window_length
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.mfcc_transform = None
        
    

    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        data = np.array(waveform.float()) # convert torch Tensor to np array for librosa    
        return torch.Tensor(lb.feature.mfcc(y=data,sr = self.sample_rate,n_mels=self.n_mels,n_mfcc=self.n_mfcc))
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        
        return self.extract(waveform=waveform)