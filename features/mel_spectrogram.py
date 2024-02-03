from typing import Optional

from torchaudio import transforms
import torch

from features.spectrogram import Spectrogram


class MelSpectrogram(torch.nn.Module):
    def __init__(
        self,
        n_mels: int = 128,
        num_frequencies: int = 256,
        window_size: int = 200,
        hop_size: Optional[int] = None,
    ) -> None:
        """Computes a mel-scale spectrogram.

        Parameters
        ----------
        n_mels: int, optional
            the number of mel filterbanks
        num_frequencies: int, optional
            the number of frequencies to sample on in the Spectrogram. Defaults to 201.
        window_length: int, optional
            the length of the spectrogram window in the Spectrogram, when using forward, in frames. Defaults to 10.
        hop_size: int, optional
            the distance between two spectrogram windows, when using forward, in frames. Defaults to win_length/2.
        """
        super(MelSpectrogram, self).__init__()

        self.spectrogram = Spectrogram(
            num_frequencies=num_frequencies,
            window_size=window_size,
            hop_size=hop_size,
        )

        self.mel_scale = transforms.MelScale(n_mels=n_mels, n_stft=num_frequencies)

    def extract(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Computes a Mel-spectrogram from a spectrogram.

        Parameters
        ----------
        spectrogram: torch.Tensor
            the previously-computed spectrogram

        Returns
        -------
        torch.Tensor
            the mel-spectrogram of the waveform associated to the given spectrogram
        """
        return self.mel_scale(spectrogram)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Computes a Mel-spectrogram from a waveform.

        Parameters
        ----------
        waveform: torch.Tensor
            the waveform as a torch tensor

        Returns
        -------
        torch.Tensor
            the mel-spectrogram of the waveform associated to the given waveform
        """
        spec = self.spectrogram(waveform)
        return self.extract(spectrogram=spec)
