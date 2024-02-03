import torch
import torchaudio.transforms as T

import numpy as np

class MFCC(torch.nn.Module):

    def __init__(
        self,
        n_mfcc: int = 128,
        n_fft: int = 2048,
        hop_length: int = 1024,
        n_mels: int = 128,
        sample_rate: int = 44100,
    ) -> None:
        """Mel-Frequency Cepstral-Coefficient feature-extractor.

        Parameters
        ----------
        n_mels: int, optional
            the number of mel filterbanks
        window_length: int, optional
            the length of the spectrogram window in the Spectrogram, when using forward, in frames. Defaults to 10.
        hop_size: int, optional
            the distance between two spectrogram windows, when using forward, in ms. Defaults to win_length/2.
        """
        
        super(MFCC, self).__init__()

        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "n_mels": n_mels,
                "hop_length": hop_length,
                "mel_scale": "htk",
            },
        )
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.mfcc_transform(waveform)