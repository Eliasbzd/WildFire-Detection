# Script to extract random 5-second segments from the Kaggle Firesound dataset and save them as individual fire sound files. "https://www.kaggle.com/datasets/forestprotection/forest-wild-fire-sound-dataset?resource=download-directory"


# Import necessary libraries
import os
import random
import librosa
import soundfile as sf

# Function to randomly extract a 5-second segment from an audio file and save it
def extract_randomly(source_path, destination_path, extraction_duration=5):
    # Load the audio file using librosa
    audio, sr = librosa.load(source_path, sr=None)

    # Get the total duration of the audio file
    total_duration = librosa.get_duration(y=audio, sr=sr)

    # Calculate the number of 5-second segments in the audio file
    num_segments = int(total_duration / extraction_duration)

    # Check if there are valid segments to extract ( not all files have a duration >= 5scd)
    if num_segments > 0:
        # Randomly select a segment
        start_segment = random.randint(0, num_segments - 1)
        start_time = start_segment * extraction_duration
        end_time = start_time + extraction_duration

        # Extract the selected segment from the audio
        segment_audio = audio[int(start_time * sr):int(end_time * sr)]

        # Save the extracted segment to the destination path
        sf.write(destination_path, segment_audio, sr)
    else:
        # Print a message if no valid segments are found
        print(f"No valid segments for {source_path}")

# Function to process all WAV files in a source folder and save extracted segments to a destination folder
def process_folder(source_folder, destination_folder):
    # Ensure the destination folder exists; create if it doesn't
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Iterate through all files in the source folder
    for filename in os.listdir(source_folder):
        print(filename)
        # Check if the file is a WAV file
        if filename.endswith(".wav"):
            # Create full paths for the source and destination files
            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(destination_folder, filename)
            # Apply the extraction function to each WAV file
            extract_randomly(source_path, destination_path)

            
# Main execution block
if __name__ == "__main__":
    # Replace "path/source_folder" with the path to your source folder containing WAV files
    source_folder = "TO_COMPLETE"

    # Replace "path/destination_folder" with the path to your destination folder for extracted segments
    destination_folder = "TO_COMPLETE"

    # Process the source folder and save extracted segments to the destination folder
    process_folder(source_folder, destination_folder, True)
