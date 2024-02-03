from typing import Optional

import torch
import torchaudio.transforms as transforms

class Spectrogram(torch.nn.Module):
    def __init__(
        self,
        num_frequencies: int = 256,
        window_size: int = 200,
        hop_size: Optional[int]= None,
    ):
        """Spectrogram feature-extractor

        Parameters
        ----------
        num_frequencies: int, optional
            the number of frequencies to sample on. Defaults to 201.
        window_length: int, optional
            the length of the spectrogram window, in frames. Defaults to 10.
        hop_size: int, optional
            the distance between two spectrogram windows, in frames. Defaults to win_length/2.
        """
        super(Spectrogram, self).__init__()
        n_fft = 2 * (num_frequencies - 1)



        self.spec = transforms.Spectrogram(
            n_fft=n_fft, win_length=window_size, hop_length=hop_size, power=2
        )

    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute the spectrogram form the waveform

        Parameters
        ----------
        waveform: torch.Tensor
            the resampled audio signal

        Returns
        -------
            the spectrogram, as a torch tensor
        """

        # Convert to power spectrogram
        spec = self.spec(waveform)

        return spec

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Computes a spectrogram from a waveform, for direct inside a model instead of having to call extract.

        Parameters
        ----------
        waveform: torch.Tensor
            the resampled audio signal

        Returns
        -------
            the spectrogram, as a torch tensor
        """
        return self.extract(waveform=waveform)
