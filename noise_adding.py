from sound_extraction import extract_randomly
import torch
import torchaudio
import torchaudio.functional as F
import os
import random
import librosa
from IPython.display import Audio


def add_forest_noise(waveform_path, save_path, noise_type="rain1", db=torch.tensor([3])):
    """This function adds a background noise randomly extracted from the chosen noise file ("birds" or "rain"). 

    Args:
        waveform_path (str): path to the initial waveform
        save_path (str): path to save the noisy waveform
        noise_type (str, optional): type of noise. Defaults to "birds2".
        db (torch.tensor, optional): strength of noise. Defaults to torch.tensor([3]).
    """
    path = os.path.join('data/Noise', noise_type+'.wav')
    extract_randomly(path, 'data/Noise/temp.wav')
    waveform, sr = torchaudio.load(waveform_path)
    segment_noise, sr = torchaudio.load('data/Noise/temp.wav')
    noisy_waveform = F.add_noise(waveform, segment_noise, db)
    torchaudio.save(save_path, noisy_waveform, sample_rate=sr)


if __name__ == "__main__":
    add_forest_noise(
        '/Users/karinamusina/pole-ia-s705-detection-feu-de-foret-public/data/esc50/audio/1-1791-A-26.wav', '/Users/karinamusina/pole-ia-s705-detection-feu-de-foret-public/data/Noise/temp.wav')
