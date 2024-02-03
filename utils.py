import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, cohen_kappa_score, classification_report, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import requests, mimetypes
import urllib

import os
import csv_util
import re
from pytube import YouTube
import time
import random
import numpy as np
from pydub import AudioSegment



def show_scores_pred(Y, Y_pred,title=None,csvfile='./plots/stats.csv',showPlots=True):
    """
    Affiche les métriques d'évaluation à partir des prédictions
    """
    if title==None:
        title=int(time.time())
    else:
        title+=str(int(time.time()))
    if showPlots:
        fig, ax = plt.subplots(figsize=(4,4))
        cm = confusion_matrix(Y,Y_pred)
        cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [False, True]) 
        cm_display.plot(cmap=plt.cm.Blues, ax=ax)

    if csvfile != None:
        curr = csv_util.csv_handler(csvfile)
        rows = curr.fetch_rows()
        title=rows[-1][0]+'-'+rows[-1][1]+'-'+rows[-1][2]+'-'+rows[-1][3]
        plt.title(title)
        plt.savefig('./plots/{}.png'.format(title), format='png')
        """
        edit csv before ending
        """
        curr.edit_last_row([None,None,None,None]+list(np.round(scores,3))+[None,None])
        
        curr.update()
        del curr
        
    if showPlots:
        plt.show() 
    scores = [accuracy_score(Y, Y_pred),f1_score(Y, Y_pred),cohen_kappa_score(Y, Y_pred)]
    print("Accuracy:", scores[0])
    print("F1-score:", scores[1])
    print("Cohen's Kappa:", scores[2])
    if showPlots:
        print("Classification Report:\n", classification_report(Y, Y_pred))
    
    
def show_scores(Y, Y_pred_probs, title=None, csvfile='./plots/stats.csv', showPlots = True):
    if title==None:
        title=int(time.time())
    else:
        title+=str(int(time.time()))
    """
    Affiche les métriques d'évaluation à partir des probabilités 
    Classification binaire !
    """
    if title==None:
        title=int(time.time())
    Y_pred = []
    for prob in Y_pred_probs:
        if prob > 0.5:
            Y_pred.append(1)
        else:
            Y_pred.append(0)
    
    scores = [accuracy_score(Y, Y_pred),f1_score(Y, Y_pred),cohen_kappa_score(Y, Y_pred),roc_auc_score(Y, Y_pred_probs)]
    print("Accuracy:", scores[0])
    print("F1-score:", scores[1])
    print("Cohen's Kappa:", scores[2])
    print("AUC:",scores[3])

    if showPlots:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
        cm = confusion_matrix(Y,Y_pred)
        cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [False, True]) 
        cm_display.plot(cmap=plt.cm.Blues, ax=ax1)

        fpr, tpr, _ = roc_curve(Y, Y_pred_probs)
        ax2.plot(fpr, tpr, color='darkorange', lw=2)
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        fig.tight_layout()
    
    if csvfile != None:
        curr = csv_util.csv_handler(csvfile)
        rows = curr.fetch_rows()
        title=rows[-1][0]+'-'+rows[-1][1]+'-'+rows[-1][2]+'-'+rows[-1][3]
        plt.title(title)
        plt.savefig('./plots/{}.png'.format(title), format='png')
        curr.edit_last_row([None,None,None,None]+list(np.round(scores,3))+[None])
        del curr
    
    if showPlots:
        plt.show()
        print("Classification Report:\n", classification_report(Y, Y_pred))
    
    
    

def download_file(file_url, save_path):
    """
    Downloading a file from a url
    Can download from a sharepoint url
    """

    print(f"Downloading file from url {file_url}")

    if "sharepoint.com" in file_url:
        print("Sharepoint url detected...")
        # Make GET request with allow_redirect
        res = requests.get(file_url, allow_redirects=True)

        if res.status_code == 200:
            # Get redirect url & cookies for using in next request
            new_url = res.url
            cookies = res.cookies.get_dict()
            for r in res.history:
                cookies.update(r.cookies.get_dict())
            
            # Do some magic on redirect url
            new_url = new_url.replace("onedrive.aspx","download.aspx").replace("?id=","?SourceUrl=")

            # Make new redirect request
            response = requests.get(new_url, cookies=cookies)

            if response.status_code == 200:
                content_type = response.headers.get('Content-Type')
                #print(content_type)
                file_extension = mimetypes.guess_extension(content_type)
                #print(response.content)
                if file_extension:
                    destination_with_extension = f"{save_path}{file_extension}"
                else:
                    destination_with_extension = save_path

                with open(destination_with_extension, 'wb') as file:
                    for chunk in response.iter_content(1024):
                        file.write(chunk)
                print("File downloaded successfully!")
            else:
                print("Failed to download the file.")
                print(response.status_code)
    else:
        print("Default file downloading...")
        urllib.request.urlretrieve(file_url, save_path)
        print("File downloaded successfully!")

def download_youtube_audio(youtube_link, output_path="."):
    """
    Downloads the audio from a YouTube video and saves it to a specified location.

    Parameters:
    - youtube_link (str): The YouTube video URL.
    - output_path (str): The directory where the downloaded audio file will be saved.
                         Defaults to the current working directory if not provided.

    Returns:
    None
    """

    def clean_filename(s):
        # Replace special characters with dashes and remove spaces
        return re.sub(r'[^\w\s.-]', '', s).strip()

    try:
        # Download YouTube video (audio only)
        yt = YouTube(youtube_link)
        video_stream = yt.streams.filter(only_audio=True).first()

        # Adapt the output file name based on the YouTube video title
        cleaned_title = clean_filename(yt.title)
        output_file = f"{cleaned_title}_audio.wav"

        # Create the full path of the output file
        output_full_path = os.path.join(output_path, output_file)

        # Download and save the audio file
        video_stream.download(output_path, filename=output_file)

        print(f"Audio downloaded and saved as {output_full_path}")

    except Exception as e:
        print(f"Error: {e}")        

def create_random_extracts(source_folder, num_extracts):
    """
    Creates random audio extracts from sound files in a source folder and saves them to individual subfolders.

    Parameters:
    - source_folder (str): The path to the folder containing sound files.
    - num_extracts (int): The number of random audio extracts to generate for each sound file.

    Returns:
    None
    """
    # Check if the source folder exists
    if not os.path.exists(source_folder):
        print(f"The source folder '{source_folder}' does not exist.")
        return

    # Iterate through all files in the source folder
    for filename in os.listdir(source_folder):
        if filename.lower().endswith(('.mp3', '.wav', '.flac', '.ogg')):  # Exclude unwanted file types
            file_path = os.path.join(source_folder, filename)

            # Check if the path is a file
            if os.path.isfile(file_path):
                # Create a folder with the same name as the file (without extension)
                folder_name = os.path.splitext(filename)[0]
                folder_path = os.path.join(source_folder, folder_name)

                # Ensure the folder does not already exist
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                # Load the audio file using pydub
                audio = AudioSegment.from_file(file_path)

                # Generate 200 random extracts of 5 seconds each
                for i in range(num_extracts):
                    start_time = random.randint(0, len(audio) - 5000)  # 5000 ms = 5 seconds
                    end_time = start_time + 5000
                    extract = audio[start_time:end_time]

                    # Save the extract in the corresponding folder
                    extract_filename = f"{folder_name}_extract_{i+1}.wav"
                    extract_path = os.path.join(folder_path, extract_filename)
                    extract.export(extract_path, format="wav")

    print(f"Extraction completed for files in '{source_folder}'.")