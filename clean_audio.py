import os
import librosa
import noisereduce as nr
import soundfile as sf

def reduce_noise_and_save(input_file, output_file):
    # Load the audio file
    waveform, sr = librosa.load(input_file, sr=None)
    
    # Apply noise reduction
    reduced_noise = nr.reduce_noise(y=waveform, sr=sr)
    
    # Save the cleaned audio to a new .wav file
    sf.write(output_file, reduced_noise, sr)
    print(f"Noise-reduced audio saved to {output_file}")

# Directory paths
input_dir = "audio_files"
output_dir = "cleaned_audio_files"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each .wav file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".wav"):
        input_file_path = os.path.join(input_dir, filename)
        output_file_path = os.path.join(output_dir, filename)
        print(f"Processing file: {filename}")
        reduce_noise_and_save(input_file_path, output_file_path)

print("Noise reduction completed for all files.")
