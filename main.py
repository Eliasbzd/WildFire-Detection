from datasets import ESCDataset, CustomDataset1,CustomDataset2
from models import train, transform_input
import noise_adding

FEATURES = ['spectrogram''mel_spectrogram','MFCC','LFCC','cochleagram','scalogram','MFCC_lib', 'STFT']
MODELS = ['svm','LDA','xgboost','randomForest','logisticregression']
FEATURE_TRANSFORMATIONS = ["original", "mean", "mean_var", "flatten", 'flatten_pca', 'flatten_pca_torch']


## TRAINING A MODELA
# CD2 = CustomDataset2(download=False)
# train(CD2, "MFCC", "mean")

## USING A MODEL
from models import transform_input
transform_input('./computed_models/17-01-2024-14-45-07_mel_spectrogram_logisticregression_.pickle','./Data_small/',method='mean')



# ## Exemple of data augmentation
# add_forest_noise(waveform_path, save_path, noise_type="rain1", db=torch.tensor([3])):
# noise_adding.add_forest_noise('./data/CustomDataset1/audio/ESC50_34.wav','./Data_small/audio/noise_audio_bird.wav',"birds1")







## COMPLETE
print("[ MAIN ] Execution complete.")