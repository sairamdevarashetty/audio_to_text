import pandas as pd
import requests
import os

# Function to download audio file
def download_audio(url, output_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_path, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded: {output_path}")
    else:
        print(f"Failed to download: {url}")

# Function to process the CSV file and download audio files
def process_csv(csv_path, output_dir):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Iterate over rows and download audio files
    for index, row in df.iterrows():
        recording_url = row['Recording Url']
        file_name = f"call_{index}.wav"  # Generating a file name
        output_path = os.path.join(output_dir, file_name)
        download_audio(recording_url, output_path)

# Main function
def main():
    csv_path = 'User_Recording_2024-08-13_15-23-28.csv'  # Replace with your CSV file path
    output_dir = './audio_files'  # Replace with your desired output directory
    process_csv(csv_path, output_dir)

if __name__ == "__main__":
    main()
