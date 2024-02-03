import torch

class STFT(torch.nn.Module):

    def __init__(
        self,
        n_fft: int = 128,
        window_length = None,
        hop_length: int = 512,
        sample_rate: int = 44100,
        absolute: bool = True,
    ) -> None:
        """Short Time Fourier Transform.

        Parameters
        ----------
        n_ftt: int, optional
            size of the Fourier Transform. Defaults to 128.
        window_length: int, optional
            the length of the spectrogram window in the Spectrogram, when using forward, in frames. Defaults to 10.
        hop_size: int, optional
            the distance between two spectrogram windows, when using forward, in ms. Defaults to win_length/2.
        absolute: bool, optional
            chooses whether to return the absolute or complex STFT. Defaults to True.
        """
        
        super(STFT, self).__init__()

        self.n_fft = n_fft
        self.window_length = window_length
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.absolute = absolute
        self.stft_transform = torch.stft 
    

    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        # Method will be deprecated
        out_c = self.stft_transform(input = waveform, n_fft=self.n_fft, return_complex=False)
        if self.absolute and out_c.shape == torch.Size([65, 6891, 2]):
            return torch.sqrt(torch.square(out_c[:, :,0])+torch.square(out_c[:, :, 1]))
        if self.absolute and out_c.shape == torch.Size([1, 65, 6891, 2]):
            return torch.sqrt(torch.square(out_c[0, :,:,0])+torch.square(out_c[0, :,:, 1]))

        return out_c
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.extract(waveform=waveform)