<h1 style="text-align: center;">Wildfire Classification</h1>


<p align="center">
<img src="./1200px-Logo_CentraleSupélec.svg.png" style="width:120px;height:60px; taxt-align:center;">
</p>
<h5 style="text-align: center;">AI-S7-05</h5>

<h2 style="text-align: center;">Quick Links</h2>
<p style = "text-align: center;">
<table style="height:100%;width:100%; text-align: center"><tr>
<td width="16%"> <a href="#l1" >Overview</a></td>
<td width="17%"> <a href="#l2" >Installation</a> </td>
<td width="17%"> <a href="#l3" >Repository Structure</a> </td>
<td width="17%"> <a href="#l4" >Modules </a> </td>
<td width="17%"> <a href="#l5" >Databases </a> </td>
<td width="16%"> <a href="#l6" >Tests</a> </td>
</tr></table></p>


<h2 id="l1" style="text-align: center;">Overview</h2>


On this page, you will find everything necessary to train, test and use a wildfire detection AI based on audio inputs. This project has been made as part of a course at _CentraleSupélec_, one of France's premier graduate institutions.


This project has been made possible thanks to the team: Erwin, Elias, Jad, Karina and Noé and the supervision of Prof Wassila Ouerdane, Prof Jean-Philippe Poli and Prof Frédéric Magoulès.


Do contact us for any more information or suggestions.


<h2 id="l2" style="text-align: center;">Installation</h2>


You can choose to run the project in a virtual environment by installing the [virtualenv](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) package and creating a virtual environment by running `python3 -m venv .env`. Then, activate it by running `source .env/bin/activate` on Unix/macOS, or `.\.env\Scripts\activate` on Windows.


The repository can then be cloned using your preferred technique.
You will need to install the dependencies using: `pip install -r requirements.txt`.


Everything will then happen through the `main.py` script.


<h2 id="l3" style="text-align: center;">Repository Structure</h2>


The repository has been built in the following structure. Different folders exist :


<table style="height:100%;width:100%; ">
<tr><th width="30%">Folder name</th><th width="70%">Purpose</th></tr>
<tr><td>calculated_features</td><td >This folder stocks precomputed features in binary format using pickle. They are named in the <i>database_feature_extraction_reducer</i> convention, thus fully describing the file. The <i>reducer</i> is the technique used to reduce input size. </td></tr>
<tr><td>classifiers</td><td>This folder is a package holding all the python scripts for the different implemented classifiers. See below a detailed list of their capabilities.</td></tr>
<tr><td>computed_models</td><td>This folder holds all the trained pickled computed ML models. They can be reused for efficient fire detection and are saved in a <i>date_feature_classifier</i> format.</td></tr>
<tr><td>data</td><td>This folder holds the downloaded datasets and their respective CSV files.</td></tr>
<tr><td>datasets</td><td>A package containing the source code for downloading and handling the datasets.</td></tr>
<tr><td>plots</td><td>This folder is used to save data regarding our computations: a <i>stats.csv</i> file holds all the important measures regarding every tested method for an easy choice. It also contains plotted information on the tested methods. Upon training of a new model, it will automatically update both these elements.</td></tr>
</table>

<br/>

Some files are also included in the main folder :



<table style="height:100%;width:100%; ">
<tr><th width="30%">File</th><th width="70%">Purpose</th></tr>
<tr><td>.gitignore</td><td>Standard gitignore</td></tr>
<tr><td>csv_util.py</td><td>Custom utility file for efficient CSV handling. </td></tr>
<tr><td>main.py</td><td>The main file: this is the control center from which the project can be easily executed and reused.</td></tr>
<tr><td>models.py</td><td>The second in command of the project: in this file, the code to train and use the models is written.</td></tr>
<tr><td>mult_to_bin.py</td><td>A custom script for simplifying a database CSV to a binary labeled CSV.</td></tr>
<tr><td>noise_adding.py</td><td>A script to transform an audio file by adding a noise effect</td></tr>
<tr><td>README.md</td><td> Yours truly</td></tr>
<tr><td>requirements.txt</td><td>Standard <I>requirements.txt</i> file, containing all required modules. </td></tr>
<tr><td>sound_extraction.py</td><td>Permits the random extraction of five seconds (standard input in our models) from longer files.</td></tr>
<tr><td>utils.py</td><td>Utility file for the creation of plots and the calculation of our various scores.</td></tr>
</table>




<h2 id="l4" style="text-align: center;">Modules</h2>


Many different techniques are implemented here. Below is a table looking at all the implemented feature extraction techniques:

| File   | Summary |
---|---|
| mel_spectrogram.py | The `MelSpectrogram` class is responsible for computing a mel-scale spectrogram from a waveform, using `spectrogram.py` and the `MelScale` class from `torchaudio`.|
| zero_crossing.py|The `Zero_crossing` class computes the zero-crossing rate of an audio waveform using the `librosa` library.|
| STFT.py | The `STFT.py` module implements the Short Time Fourier Transform (STFT) using `torchaudio`. The output can be either the absolute or complex form of the STFT Sectrogram but defaults to the absolute value.|
| lfcc.py | The `LFCC` class calculates Linear Frequency Cepstral-Coefficient features from an audio waveform using the `torchaudio`. It is comparable to an MFCC method with a linear scale.|
| scalogram.py | Generates a Scalogram, using `PyWavelets` to perform Continuous Wavelet Transform (CWT) on a waveform.|
| PCA.py| The `PCA` module manually reimplements a Principal Component Analysis.It is used to reduce the input size of the methods.
| cochleagram.py| The `Cochleagram` class extracts cochleagrams from audio waveforms, using a special module. *It is however very slow to compute and not recommended.* |
| mfcc_lib.py|  The `MFCC_LIB` class extracts the Mel-Frequency Cepstral Coefficients (MFCC) features using `librosa`.|
| spectrogram.py| Defines the `Spectrogram` class, using `torchaudio`.|
| mfcc.py| Alternative `MFCC` class to compute Mel-Frequency Cepstral Coefficients (MFCC) using `torchaudio`.

</br>

<p style="text-align: center;">Feature Extraction Techniques</p>


Finally, another table gives a look at the various models used for classification. All classifiers print their scores automatically upon completion of testing and training.

|File|Summary|
| ---| ---|
| randomforest.py             | Implements a RandomForest classifier. The class uses the `RandomForestClassifier` from the `scikit-learn` library to perform the classification task.|
| lda.py| Implements Linear Discriminant Analysis (LDA). It uses the `LinearDiscriminantAnalysis` class from the `sklearn.discriminant_analysis` module. |
| logisticregression.py | Implements logistic regression. It uses the `LogisticRegressionClassifier` from `sklearn` |
| ACDNet.py| Represents a CNN model designed to classify environmental sounds into one of 50 classes. The code also includes functions for creating the model's layers and determining the pool sizes for the temporal feature extraction stage. |
| xgboost.py| Implements an XGBoost model. It uses the `xgboost` library|
| cnn_bardou.py                 | Implements the `CNNBardou` class.|
| svm.py                        | Implements an `SVM` (Support Vector Machine) classifier. |
| nn_utils.py                     | Contains several utility modules and classes for neural network operations. It includes functions for flattening tensors, local response normalization, creating CNN layers, and saving models. |
| crnn_zhang.py| Implements a Convolutional Reccurrent Neural Networks (CRNN) model based on Zhang's paper. It consists of several convolutional layers followed by a GRU layer and linear layers for classification.|
| cnn_simple.py|Implements a simple CNN model, using `torch`.|
| mobilenet.py| Implements a pre-trained CNN model using previous work by F. Schmid. It also uses the script `mobilenet_utils.py`.|
</br>

<p style="text-align: center;">Classifiers</p>

Most of the NN presented were done by a previous group working on this project. We have chosen to focus on the other, more efficient and more effective, classifiers.

Do have a look at the <a link="l6">tests</a> section for an in depth comparison of these techniques.


<h2 id="l4" style="text-align: center;">Getting Started</h2>
Different functionalities are offered by this project.


### Train and test a model


To train and test a model, you will need to use the main.py file. In it, simply edit the line using as follows:


Once you have chosen a database, feature extractor, and classifier from the options you will need to:
1. Import the database


Import the database by starting an instance as follows `CD2 = CustomDataset2(download=False)`. Only set download to true if you wish to overwrite any existing database files. By default, it will attempt to find the database and otherwise download it.


2. Identify the necessary arguments


Identify the corresponding key word with each method in the `FEATURES` and `CLASSIFIERS` lists. For instance, Mel Filter Cepstral Coefficients corresponds to ```MFCC``` and Linear Discriminant Analysis to `LDA`. You can then call the required function as follows:


```
train(CD2, "MFCC", "LDA")
```
For information, the arguments of ```train``` are:


```
def train(
    dataset: SplitableDataset,
    feature_name: str,
    classifier_name: str,
    feature_transf: str = "mean_var",
    epochs: int = 30,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    recompute=False):
```
The output in the terminal will then contain all necessary information to judge the model's pertinence. It will equally be scored in the `./plots/` folder under two time-stamped plots (train and test) and as a line in the `./plots/stats.csv` file


### Using a precomputed model


One might wish to use a previously trained model to try and predict the presence - or absence - of wildfires. To do so, once you have identified the file name of the model in the ``./computed_models/`` folder, in the main.py file, you can call:


```
models.transform_input(computed_model_path,input_path)
# Optional keywords: feature_model (should be included in the path), method (defaults to "mean")
```
Where the `audio_files` argument corresponds to the audio samples you wish to feed to the model.


> ⚠️ **The audio files have to respect the same format as the database used for training: they will be fed to the same model.**


<h2 id="l5" style="text-align: center;">Database</h2>

</br>

Various Databases are implemented, recapped in the following table:


<table style="height:100%;width:100%; ">
<thead>
    <tr>
        <th width="10%">Name</th>
        <th width="40%">Source</th>
        <th width="40%">Number of sounds</th>
        <th width=10%>URL(*)</th>
    </tr>
</thead>
<tbody>
<tr>
    <td><br>ESC50</td>
    <td><br>Freesound</td>
    <td><br>2000 (50 classes)</td>
    <td><br>https://github.com/karoldvl/ESC-50/archive/master.zip</td>
</tr>
<tr>
    <td><br>FSC22</td>
    <td><br>Freesound</td>
    <td><br>2025 (27 classes)</td>
    <td><br>https://doi.org/10.3390/s23042032 or https://www.kaggle.com/datasets/irmiot22/fsc22-dataset</td>
</tr>
<tr>
    <td><br>KaggleFire (custom name)</td>
    <td><br>Kaggle, and probably from Youtube.</td>
    <td><br>280, 50s</td>
    <td><br>https://www.kaggle.com/datasets/forestprotection/forest-wild-fire-sound-dataset</td>
</tr>
<tr>
    <td><br>Yellowstone (custom name)</td>
    <td><br>From the national park Yellowstone</td>
    <td><br>5 sounds, from 30s to 1min</td>
    <td><br>https://www.nps.gov/yell/learn/photosmultimedia/sounds-fire.htm and https://acousticatlas.org/search.php?q=fire</td>
</tr>
<tr>
    <td><br>CustomDataset1</td>
    <td><br>ESC50 + KaggleFire, transformed into a binary classification (1 if fire), and 5s</td>
    <td><br>2282, 5s</td>
    <td><br>https://centralesupelec-my.sharepoint.com/:u:/g/personal/erwin_deng_student-cs_fr/Ee9ZCIDKwFhPkgCep43zb-IBjm4dWDm147M2Mgb032hMaQ?e=Tr9EoQ</td>
</tr>
<tr>
    <td><br>CustomDataset2</td>
    <td><br>FSC22 + KaggleFire + Yellowstone, transformed into a binary classification (1 if fire), and 5s</td>
    <td><br>2388, 5s</td>
    <td><br>https://centralesupelec-my.sharepoint.com/:u:/g/personal/erwin_deng_student-cs_fr/EbXM4vYtsLZMsNL49Q7ZwEkB7z2Bt_od-V4CTc1m4KGEPQ?e=2tQTRU</td>
</tr>
</tbody></table>

</br>
These databases are downloaded from the Internet using the scripts in datasets, and then processed to normalize and create standardized CSV indexes. They are only downloaded once when required by the user in the main.py function.
(*) There is no need to download the datasets manually, our script automatically create them. However, you can still have a look at them manually with these urls. 


<h2 id="l6" style="text-align: center;">Tests</h2>


We have conducted an extensive array of tests on the different methods using various scores, namely:
- Accuracy
- F1-Score
- Kappa (Cohen)
- AUC
- Confusion Matrix


For our purposes (binary classification) the most interesting score is of course the Confusion Matrix, F1-Score and Kappa.


Below are examples of the best methods according to our scores (this is an extract from a fully stocked `stats.csv` file). These results were obtained from a test datase, obtained by splitting our complete dataset into train/test datasets.


<table>
<thead>
  <tr>
    <th>Classifier</th>
    <th>Feature Extraction</th>
    <th>Transformer</th>
    <th>Accuracy</th>
    <th>F1-Score</th>
    <th>Kappa (Cohen)</th>
    <th>AUC</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>LDA</td>
    <td>MFCC</td>
    <td>mean_var</td>
    <td>0,975</td>
    <td>0,927</td>
    <td>0,912</td>
    <td>0,99</td>
  </tr>
  <tr>
    <td>logisticregression</td>
    <td>MFCC_lib</td>
    <td>mean_var</td>
    <td>0,919</td>
    <td>0,808</td>
    <td>0,758</td>
    <td>0,966</td>
  </tr>
  <tr>
    <td>randomForest</td>
    <td>spectrogram</td>
    <td>mean_var</td>
    <td>0,986</td>
    <td>0,961</td>
    <td>0,953</td>
    <td>0,998</td>
  </tr>
  <tr>
    <td>svm</td>
    <td>STFT</td>
    <td>mean_var</td>
    <td>0,86</td>
    <td>0,653</td>
    <td>0,566</td>
    <td> </td>
  </tr>
  <tr>
    <td>xgboost</td>
    <td>MFCC</td>
    <td>mean_var</td>
    <td>0,986</td>
    <td>0,961</td>
    <td>0,952</td>
    <td>0,999</td>
  </tr>
</tbody>
</table>


In a general sense, the most succesful methods usually involve MFCC techniques and XGBoost or RandomForest algorithms.

