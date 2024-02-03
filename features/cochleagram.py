import torch
import torchaudio
import numpy as np
import chcochleagram
from skimage.feature import graycomatrix, graycoprops
from skimage import data
import matplotlib.pyplot as plt
import cv2


class Cochleagram(torch.nn.Module):
    def __init__(self, signal_size=5*44100, s_rate=44100, pad_factor=1.0, use_rfft=True, n_filters=256, env_sr=245, dist1=20, angle1=0, dist2=10, angle2=np.pi/2):
        """Cochleagram feature-extractor.

        Parameters
        ----------
        signal_size : int
            -- The number of samples of the signal
        sr : int
            -- The sampling frequency of the signal
        pad_factor : float
            --  Zero padding applied to the waveform, so the end signal is length pad_factor*signal_length
        use_rfft : booleen
            -- Whether to use rfft operations when appropriate (recommended)
        n_filters : int
            -- Number of filters to evenly tile the space
        env_sr : int
            -- Sampling rate after downsampling
        dist_1 : int
            --  distance parameter for generating the GLCM matrixes
        angle_1 : float
            --  angle parameter for generating the GLCM matrixes




        """
        super(Cochleagram, self).__init__()
        self.signal_size = signal_size
        self.sr = s_rate
        self.dist1 = dist1
        self.dist2 = dist2
        self.angle1 = angle1
        self.angle2 = angle2

        half_cos_filter_kwargs = {
            'n': n_filters,
            # Lowest center frequency for full filter (if lowpass filters are used they can be centered lower)
            'low_lim': 50,
            'high_lim': 8000,  # Highest center frequency
            # Positive integer that determines how densely ERB function will be sampled
            'sample_factor': 4,
            # Whether to use the full-filter. Must be False if rFFT is true.
            'full_filter': False,
        }
        # These arguments are for the CochFilters class (generic to any filters).
        coch_filter_kwargs = {'use_rfft': use_rfft,
                              'pad_factor': pad_factor,
                              'filter_kwargs': half_cos_filter_kwargs}
        # This (and most) cochleagrams use ERBCosFilters, however other types of filterbanks can be
        # constructed for linear spaced filters or different shapes. Make a new CochlearFilter class for
        # these.
        self.filters = chcochleagram.cochlear_filters.ERBCosFilters(
            signal_size, s_rate, **coch_filter_kwargs)
        # Define an envelope extraction operation
        # Use the analytic amplitude of the hilbert transform here. Other types of envelope extraction
        # are also implemented in envelope_extraction.py. Can use Identity if want the raw subbands.
        envelope_extraction = chcochleagram.envelope_extraction.HilbertEnvelopeExtraction(
            signal_size, s_rate, use_rfft, pad_factor)

        # Define a downsampling operation
        # Downsample the extracted envelopes. Can use Identity if want the raw subbands.
        # Parameters for the downsampling filter (see downsampling.py)
        downsampling_kwargs = {'window_size': 1001}
        downsampling_op = chcochleagram.downsampling.SincWithKaiserWindow(
            s_rate, env_sr, **downsampling_kwargs)
        # Define a compression operation.
        compression_kwargs = {'power': 0.3,  # Power compression of 0.3
                              'offset': 1e-8,  # Offset for numerical stability in backwards pass
                              'scale': 1,  # Optional multiplicative value applied to the envelopes before compression
                              'clip_value': 100}  # Clip the gradients for this compression for stability
        compression = chcochleagram.compression.ClippedGradPowerCompression(
            **compression_kwargs)

        self.coch = chcochleagram.cochleagram.Cochleagram(
            self.filters, envelope_extraction,  downsampling_op, compression=compression)

    def extract(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute the Cochleagram form the waveform


        Parameters
        ----------
        waveform : torch.Tensor
            -- The resampled audio signal.



        Returns
        -------
        coch : torch.Tensor
            -- The Cochleagram, as a torch tensor
        """

        # Convert to power spectrogram
        coch = self.coch(waveform)

        return coch

    def plotCochlea(self, waveform, title='Coachleagram'):
        """Plot the cochleagram of the signal


        Parameters
        ----------
        waveform : torch.Tensor
            -- The resampled audio signal.
        title : string
            -- The title of the plot
        """
        cochleagram_computed = self.coch(waveform)
        plt.imshow(np.squeeze(cochleagram_computed.detach().numpy()), origin='lower', extent=(
            0, cochleagram_computed.shape[2], 0, cochleagram_computed.shape[1]))
        plt.title(title)

        # Depending on the temporal padding the cochleagram length may not be exactly equal env_sr*signal_size/sr
        # Because of this, set the x-axis tick labels based on the original audio.
        num_ticks = 9
        x_tick_numbers = [t_num*cochleagram_computed.shape[-1] /
                          (num_ticks-1) for t_num in range(num_ticks)]
        x_tick_labels = [t_num*self.signal_size/self.sr /
                         (num_ticks-1) for t_num in range(num_ticks)]
        plt.xticks(x_tick_numbers, x_tick_labels)
        plt.xlabel('Time (s)')

        # Label the frequency axis based on the center frequencies for the ERB filters.
        self.filters.filter_extras['cf']
        # Use ticks starting at the lowest non-lowpass filter center frequency.
        y_ticks = [y_t+3 for y_t in plt.yticks()[0] if y_t <=
                   cochleagram_computed.shape[1]]
        plt.yticks(y_ticks, [
                   int(round(self.filters.filter_extras['cf'][int(f_num)])) for f_num in y_ticks])
        plt.ylabel('Frequency (Hz)')
        plt.show()

    def extract_glcm(self, waveform):
        """Extracts the Gray Level Co-Occurence Matrix from the cochleagram.


        Parameters
        ----------
        waveform : torch.Tensor
            -- The resampled audio signal.

        Returns
        ----------
        glcm : numpy.ndarray 
            -- The GLCM matrix in 4 dimensions. 
            For each combination (distance,angle) given, a 256x256 matrix is given 
            on the 2 first dimensions. The 2 last dimensions are used to map the 
            (distance,angle) combinations to the 2 dimensions matrixes.        

        """
        coch_int = self.extract(waveform)
        coch_int = cv2.normalize(
            coch_int.numpy(), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)[0, :, :]
        glcm = graycomatrix(coch_int, distances=[self.dist1, self.dist2], angles=[
                            self.angle1, self.angle2], levels=256, symmetric=True, normed=True)
        return glcm
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
            return self.extract(waveform=waveform)


# TEST OF COCHLEAGRAM
if __name__ == '__main__':
    cochlea = Cochleagram(env_sr=245)

    wave1, sample_rate = torchaudio.load('Data_small/airplane05.wav')

    cochlea.plotCochlea(wave1[:, :220500])
    c = cochlea.extract(wave1[:, :220500])
    print('Shape of cochleagram : ', c[0, :, :].shape)

    g = cochlea.extract_glcm(wave1[:, :220500])
    print(f'Shape GLCM : {g.shape}')
