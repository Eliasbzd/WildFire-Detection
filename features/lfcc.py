import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

import numpy as np

class LFCC(torch.nn.Module):

    def __init__(
        self,
        n_fft: int = 128,
        window_length = None,
        hop_length: int = 512,
        n_lfcc: int = 16,
        sample_rate: int = 44100,
    ) -> None:
        """Linear Frequency Cepstral-Coefficient feature-extractor.

        Parameters
        ----------
        n_lfcc: int, optional
            the number of  filterbanks
        window_length: int, optional
            the length of the spectrogram window in the Spectrogram, when using forward, in frames. Defaults to 10.
        hop_size: int, optional
            the distance between two spectrogram windows, when using forward, in ms. Defaults to win_length/2.
        """
        
        super(LFCC, self).__init__()

        self.n_fft = n_fft
        self.window_length = window_length
        self.hop_length = hop_length
        self.n_lfcc = n_lfcc
        self.sample_rate = sample_rate
        self.lfcc_transform = T.LFCC(
            sample_rate=sample_rate,
            n_filter=n_fft,
            n_lfcc=n_lfcc,
            speckwargs={
                "n_fft": n_fft,
                "hop_length": hop_length
            },
        )
    

    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.lfcc_transform(waveform)
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.extract(waveform=waveform)