import torch
import numpy as np
import librosa


class Zero_crossing(torch.nn.Module):

    """
    Compute the zero-crossing rate of an audio time series using librosa


    y : np.ndarray [shape=(…, n)]
        Audio time series. Multi-channel is supported.
    frame_length : int > 0
        Length of the frame over which to compute zero crossing rates
    hop_length : int > 0
        Number of samples to advance for each frame
    center : bool
        If True, frames are centered by padding the edges of y. This is similar to the padding in librosa.stft, but uses edge-value copies instead of zero-padding.
    """

    def __init__(
        self,
        frame_length=2048, hop_length=512, center=True
    ) -> None:

        super(Zero_crossing, self).__init__()
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.center = center

    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        data = waveform.numpy()
        data = data.ravel()
        coefs = librosa.feature.zero_crossing_rate(
            data, frame_length=self.frame_length, hop_length=self.hop_length, center=self.center)
        return torch.from_numpy(coefs)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.extract(waveform=waveform)
