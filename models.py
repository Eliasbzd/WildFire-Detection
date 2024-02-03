import logging
import json

import torch
from urllib.request import urlopen
from urllib.parse import urlencode
from classifiers.svm import SVM
from classifiers.lda import LDA
from classifiers.xgboost import XGBoost
from classifiers.randomforest import RandomForest
from classifiers.logisticregression import LogisticRegression
from classifiers.cnn_bardou import CNNBardou
from classifiers.cnn_simple import CNNSimple
from classifiers import MobileNet
from classifiers.nn_utils import SequentialSaveableModel, SaveableModel
from datasets import TrainValidTestDataLoader, SplitableDataset
from features import Spectrogram, MelSpectrogram, Cochleagram, MFCC, Scalogram, LFCC, MFCC_LIB, PCA, STFT, Zero_crossing

import numpy as np
from itertools import chain
import pickle
import csv_util
from datetime import date
from datetime import datetime
import os
from datasets import db_utils

from sklearn.decomposition import PCA as PCA_sklearn


FEATURES = {'spectrogram': Spectrogram,
            'mel_spectrogram': MelSpectrogram,
            'MFCC': MFCC,
            'LFCC': LFCC,
            'cochleagram': Cochleagram,
            'scalogram': Scalogram,
            'MFCC_lib': MFCC_LIB,
            'STFT': STFT,
            'zero_crossing': Zero_crossing}
MODELS = {'svm': SVM,
          'LDA': LDA,
          'xgboost': XGBoost,
          'randomForest': RandomForest,
          'logisticregression': LogisticRegression}
# size of a spectrogram/mel-spectrogram
MODEL_KWARGS = {"input_size": (256, 2206)}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_pytorch(
    model: SaveableModel,
    loaders: TrainValidTestDataLoader,
    epochs: int = 30,
    learning_rate: float = 0.001,
    logging_file: str = "classifiers.log",
):
    """Train a neural network on the ESC-50 dataset.

    Parameters
    ----------
    model: torch.nn.Module
        the model to train on the ESC-50 dataset
    train_percentage: float
        the percentage data to use for training
    test_percentage: float
        the percentage data to use for testing
    learning_rate: float
        the learning rate to use for training
    """

    logging.basicConfig(filename=logging_file)

    # We use cross-entropy as it is well-known for performing well in classification problems
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        print(
            f"----------------------- EPOCH {epoch} -----------------------")

        running_loss = 0.0
        model.train(True)

        for batch_num, data in enumerate(loaders.train):
            waveforms, labels = data
            waveforms, labels = waveforms.to(device), labels.to(device)
            # set optimizer params to zero
            optimizer.zero_grad()

            predictions = model(waveforms)

            loss = loss_func(predictions, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Batch {batch_num}, loss: {running_loss}")

        confusion = torch.zeros((50, 50), dtype=torch.int16)

        model.eval()
        with torch.no_grad():  # set_grad_enabled(False):
            for batch_num, data in enumerate(loaders.valid):
                waveforms, labels = data
                waveforms, labels = waveforms.to(device), labels.to(device)

                outputs = model(waveforms).tolist()
                labels = labels.tolist()

                for pred, truth in zip(outputs, labels):
                    print(">Pred", pred)
                    pred = np.argmax(pred)
                    print(">Pred", pred)
                    print("truth", truth)
                    confusion[pred, truth] += 1

        true_preds = torch.sum(torch.diagonal(confusion))

        accuracy = true_preds/torch.sum(confusion)

        print(f"Validation Accuracy: {accuracy*100:.2f}%")

        torch.set_printoptions(profile="full")
        # print(confusion)
        torch.set_printoptions(profile="default")

        model.save(epoch=epoch)

    print("----------------------FINISHED TRAINING----------------------")


def generate_models(
    feature_extractions: dict[str, torch.nn.Module],
    classifier: torch.nn.Module,
    classifier_name: str,
    classifier_kwargs: dict
) -> SequentialSaveableModel:
    """Generates models for all given feature extractions given a classifier

    Parameters
    ----------
    feature_extractions: list[str, torch.nn.Module]
        a list of feature extractions with their names
    classifier: torch.nn.Module
        the classifier to use
    classifier_name: the name of the classifier

    Returns
    list[SaveableModel]
        a list of saveable models, one for each feature extraction
    """

    models = []
    for name, feature_extraction in feature_extractions.items():

        model = SequentialSaveableModel(
            (feature_extraction(), name), (classifier(
                **classifier_kwargs), classifier_name)
        )

        model.to(device)
        models.append(model)

    return models


FEATURES_TRANSF = ["original", "mean", "mean_var",
                   "flatten", 'flatten_pca', 'flatten_pca_torch']


def getFeatures(
    feature_model: torch.nn.Module,
    subset,
    method="mean"
):
    """
    Extract all the features for a given feature_model and a subset
    This is usefull when trying to train a machine learning method, where all the training data have to be loaded.
    However, be carefull with the memory usage. Choose "mean" or "mean_var" in that case. 

    Args:
    - feature_model (torch.nn.Module): the feature model to use (in FEATURES) 
    - subset: the subset to use (train, valid or test, or any subset)
    - method: "original", for no transformation | "mean", to convert a 2D shape to 1D shape feature, by computing the  mean for each column


    Return:
    - (X, Y), with X an array that contains all the data (which are np arrays), and Y an array with all the labels
    """

    if method not in FEATURES_TRANSF:
        print(f"getFeatures error: {method} unknown")
        return

    X = []
    Y = []
    for i in range(len(subset)):
        sample, label = subset[i]
        pred = feature_model(sample)
        if method == "mean":
            pred = torch.mean(pred, dim=1)
        elif method == "mean_var":
            pred = torch.cat([torch.mean(pred, dim=1), torch.var(pred, dim=1)])
        elif method == "flatten" or method == 'flatten_pca' or method == 'flatten_pca_torch':
            pred = torch.flatten(pred)
        X.append(pred.numpy())
        Y.append(label)
    return (X, Y)


def train(
    dataset: SplitableDataset,
    feature_name: str,
    classifier_name: str,
    feature_transf: str = "mean_var",
    epochs: int = 30,
    batch_size: int = 32,
    recompute=False
):
    """Construct and train a model, with a specified feature_extraction and classifier (which can be a pytorch model or not)

    Inputs:
    - dataset: dataset used for training. located in ./datasets
    - feature_name: str in FEATURES
    - feature_transf: str in FEATURES_TRANSF
    - epochs: deprecated, used in NN training
    - learning_rate: deprecated, used in NN training
    - batch_size: used to split dataset
    - recompute: whether to recompute the feature or use precomputed
    Outputs:
      None
    """
    """
    Prep to csv
    """
    curr = csv_util.csv_handler('./plots/stats.csv')
    curr.add_row([classifier_name, feature_name, feature_transf, dataset.name,
                 None, None, None, None, date.today().strftime("%d/%m/%Y %H:%M:%S")])

    if feature_name not in FEATURES:
        print("Feature", feature_name, "unknown. Abort training.")
        return

    feature_model = FEATURES[feature_name]()

    if classifier_name == "mobilenet":
        print("Deep Learning Method : MobileNet")
        loaders = dataset.train_test_split().into_loaders(batch_size=batch_size)
        model = MobileNet(cuda=True, n_epochs=epochs)
        model.train(loaders.train, loaders.valid)

        print("==== MobileNet Test scores ====")
        model.test(loaders.test)

        return model

    """ NOT IMPLEMENTED 
    loaders = dataset_train_test.into_loaders(batch_size=batch_size)
    if classifier_name == "cnn_bardou":
        print("CNN Bardou voulu")
        classifier_model = CNNBardou(**MODEL_KWARGS)
        print("CNN Bardou créé")
        model = SequentialSaveableModel(
            (feature_model, feature_name), (classifier_model, classifier_name)
        )
        model.to(device)
        print("CNN Bardou chargé")
        train_pytorch(model, loaders, epochs, learning_rate)
        return
    if classifier_name == "cnn_simple":
        print("CNN_SIMPLE voulu")
        classifier_model = CNNSimple(**MODEL_KWARGS)
        print("CNN_SIMPLE créé")
        model = SequentialSaveableModel(
            (feature_model, feature_name), (classifier_model, classifier_name)
        )
        model.to(device)
        print("CNN_SIMPLE chargé")
        train_pytorch(model, loaders, epochs, learning_rate)
        return
    """

    """
    ======= This part : Machine Learning methods ======
    """
    print("[ INFO ] Method", classifier_name, " is a Machine Learning method.")

    def compute(file_name):
        '''Callable function to calculate features if user wishes to. Stores the computed values.
        Returns:
        - (train_data, train_valid, train_test) triplet
        '''
        dataset_train_test = dataset.train_test_split()
        T = [getFeatures(feature_model, dataset_train_test.train, feature_transf),
             getFeatures(feature_model, dataset_train_test.valid,
                         feature_transf),
             getFeatures(feature_model, dataset_train_test.test, feature_transf)]
        with open(file_name, 'wb') as f:
            pickle.dump(T, f)
        print("[ INFO ] New Computed features stored in {} ".format(file_name))
        return T

    file_name = os.path.join(
        './calculated_features', f"{dataset.name}_{feature_name}_{feature_transf}_calculated_features.pickle")

    if recompute:
        """
        User asked to recompute the features: this implies a shuffling of the test/train/valid sets.
        """
        print("\n[ FE ] Computing features...")
        train_data, valid_data, test_data = compute(file_name)
        """
      EXPORT DATASETS
      """

    else:
        if not os.path.isfile(file_name):
            '''Inexistent file ...'''
            print("\n[ FE ] No existing features found. Computing features...")
            # Recomputes all the data
            train_data, valid_data, test_data = compute(file_name)
        else:
            with open(file_name, 'rb') as curr_file:
                curr_db = pickle.load(curr_file)
            print(
                "\n[ INFO ] Loaded features from {}. Continuing...".format(file_name))
            train_data, valid_data, test_data = curr_db

    X = train_data[0] + valid_data[0]
    Y = train_data[1] + valid_data[1]

    X_test = test_data[0]
    Y_test = test_data[1]

    print("[ FE ] Features computed:", len(X),
          "train values, and", len(X_test), "test values")
    print("[ FE ] Shape of elements of X :", X[0].shape)

    """
    Computing for PCA 
    """
    if feature_transf == "flatten_pca":
        print("[ FE ] Dimension reduction on X with pca")

        pca = PCA_sklearn(n_components=min(10,len(X)))
        pca.fit(X)
        X = pca.transform(X)
        X_test = pca.transform(X_test)

    if feature_transf == "flatten_pca_torch":
        pca = PCA(128)
        X = torch.tensor(X)
        pca.SVD(X)
        X = np.array(pca.extract(X))
        X_test = np.array(pca.extract(torch.tensor(X_test)))

    """
    ML models
    """
    print("\n[ ML ] Initializing {}".format(classifier_name))
    machine = MODELS[classifier_name]()
    machine.train(X, Y)
    machine.test(X_test, Y_test)

    """
    EXPORT CURRENT MODEL
    """
    exp_name = "./computed_models/" + datetime.now().strftime("%d-%m-%Y-%H-%M-%S") + \
        "_" + dataset.name + "_" + feature_name + \
        '_' + classifier_name + '_' + feature_transf +'.pickle'
    with open(exp_name, 'wb') as f:
        pickle.dump(machine, f)
    print("===[ Model exported as {} ]===".format(exp_name))


def load_model(filename: str):
    """
    Load a pre-computed model.

    Args:
    - filename (String): path to the pickel file containing the model

    Return:
    - model (): returns the pre-computed model
    """
    try:
        with open(filename, 'rb') as curr_file:
            model = pickle.load(curr_file)
        print("[ SUCCESS ] Model loaded from {}".format(filename))
    except:
        print("[ ERROR ] File could not be opened.")
        model = None
    return model

def test_model(model_file, model_name, dataset, feature_name, feature_transf):
    """
    Test a model on a given dataset, feature_name, and feature_transf
    
    Args:
    - model_file (String): path the pickel file containing the model
    - model_name (String): give a model name
    - dataset: the dataset to test
    - feature_name (String): the feature to use
    - feature_transf (String): the feature transform to use
    """
    statsTest = './plots/statsTest.csv'
    curr = csv_util.csv_handler(statsTest)
    curr.add_row([model_name, feature_name, feature_transf, dataset.name,
                 None, None, None, None, date.today().strftime("%d/%m/%Y %H:%M:%S")])

    model = load_model(model_file)

    if feature_name not in FEATURES:
        print("Feature", feature_name, "unknown. Abort training.")
        return

    feature_model = FEATURES[feature_name]()

    features = getFeatures(feature_model, dataset, feature_transf)
    X = features[0]
    Y = features[1]
    
    model.test(X, Y, statsTest)

def transform_input(
        computed_model_path:str,
        input_path:str,
        feature_model:str = "MFCC",
        method:str = "mean",
        binary:bool = True,
    ):
    """
    Easy prediction to use a model
    Args:
    - computed_model: path, should be  "./computer_models/<model_name>.pickle"
    - input_path: path to the directory containing a <input_path>/audio/ with all the files to test
    - feature_model: the feature model to use (from FEATURES) 
    - method: "original", for no transformation | "mean", to convert a 2D shape to 1D shape feature, by computing the  mean for each column
    - binary: binary classifying system. True by default.

    Return:
    - Y the prediction label according to the model

    /!\ Requires implementation of predict function in the models. This is not implemented on all codes yet
    """
    # Check the correctness of inputs
    try:
        l = computed_model_path.split('_')
        if l[2] in FEATURES:
            feature_model = l[2]
        else:
            print("[ INFO ] Could not check the feature model. You might get an error.")    
        if l[4] in FEATURES_TRANSF:
            method = l[4]
        else:
            print("[ INFO ] Could not check the feature_transf. You might get an error.")    
    except:
        print("[ INFO ] Could not check the feature_model. You might get an error.")

    curr_d = db_utils.temporary_test_dataset(input_path)
    computed_model = load_model(computed_model_path)
    
    # Transform input
    if method not in FEATURES_TRANSF:
        print(f"[ ERROR ] Method {method} unknown")
        return
    
    if feature_model not in FEATURES:
        print(f"[ ERRORS ] Feature {feature_model} unknown")
        return
    
    feature_model = FEATURES[feature_model]()

    X,_ = getFeatures(feature_model,curr_d,method)
    print("[ FE ] Features computed for:", len(X),)
    print("[ FE ] Shape of elements of X :", X[0].shape)
    # predict
    try:
        Y = 1*(np.array(computed_model.evaluate(X))>0.5)
    except Exception as exc:
       print("[ ERROR ] Could not predict from {}. Is evaluate implemented in the source code?".format(computed_model))
       print(exc)
       print("Try changing the method to reduce the size of the input: it is currently ", method)
       return
    print("[ PREDICTION ] Prediction:")
    predictions = []
    for i in range(curr_d.__len__()):
        print('\t',curr_d.csv['filename'][i], '\t',Y[i] )
        predictions.append('\t'+str(curr_d.csv['filename'][i])+ '\t \t'+str(Y[i])+"\n")

    return predictions