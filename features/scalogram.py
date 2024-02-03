import torch
import numpy as np
import pywt


class Scalogram(torch.nn.Module):

    """
    lien de la doc: https://pywavelets.readthedocs.io/en/latest/ref/index.html
    scales: array_like
        The wavelet scales to use. One can use f = scale2frequency(wavelet, scale)/sampling_period to determine what physical frequency, f. Here, f is in hertz when the sampling_period is given in seconds.

    wavelet: Wavelet object or name
        Wavelet to use. 

            ['Gaussian', 'Mexican hat wavelet', 'Morlet wavelet', 'Complex Gaussian wavelets', 'Shannon wavelets', 'Frequency B-Spline wavelets', 'Complex Morlet wavelets']
            ['cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5', 'cgau6', 'cgau7', 'cgau8', 'cmor', 'fbsp', 'gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8', 'mexh', 'morl', 'shan']

                    gaus family: gaus1, gaus2, gaus3, gaus4, gaus5, gaus6, gaus7, gaus8
                    mexh family: mexh
                    morl family: morl
                    cgau family: cgau1, cgau2, cgau3, cgau4, cgau5, cgau6, cgau7, cgau8
                    shan family: shan
                    fbsp family: fbsp
                    cmor family: cmor

    sampling_period: float
        Sampling period for the frequencies output (optional). The values computed for coefs are independent of the choice of sampling_period (i.e. scales is not scaled by the sampling period).

    method: {‘conv’, ‘fft’}, optional
        The method used to compute the CWT. Can be any of:
            conv uses numpy.convolve.
            fft uses frequency domain convolution.
            auto uses automatic selection based on an estimate of the computational complexity at each scale.
        The conv method complexity is O(len(scale) * len(data)). The fft method is O(N * log2(N)) with N = len(scale) + len(data) - 1. It is well suited for large size signals but slightly slower than conv on small ones.

    """

    def __init__(
        self,
        fs=44100,
        # frequencies=np.array(
        #    [40000, 1000, 150, 100, 50, 40, 35, 25, 20, 15, 10, 5]) / 44100,  # normalize
        # scale=pywt.frequency2scale('mexh', np.array(
        #    [40000, 1000, 150, 100, 50, 40, 35, 25, 20, 15, 10, 5]) / 44100),
        frequencies=np.array(
            [1000, 400, 200, 100, 50]) / 44100,  # normalize
        scale=pywt.frequency2scale('mexh', np.array(
            [1000, 400, 200, 100, 50]) / 44100),
        wavelet='gaus7'
    ) -> None:

        super(Scalogram, self).__init__()

        self.scale = scale
        self.frequencies = frequencies
        self.sample_rate = fs
        self.wavelet = wavelet

    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        data = waveform.numpy()
        # on prend un quart du son parce que sinon trop long et trop de mémoire
        data = data.ravel()[0:140000]
        sc = pywt.frequency2scale('gaus7', np.arange(100, 7000, 100) / 44100)
        coefs, freqs = pywt.cwt(data,
                                scales=sc,
                                wavelet=self.wavelet,
                                method='fft'
                                )

        return torch.from_numpy(coefs)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.extract(waveform=waveform)
