import torch
import torchaudio
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from scipy.signal import butter, lfilter
import librosa

# Load Whisper model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

def preprocess_audio(waveform, sampling_rate=16000):
    # Normalize audio
    waveform = waveform / waveform.abs().max()
    
    # Apply a low-pass filter to reduce noise
    def butter_lowpass(cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y

    # Apply the filter
    cutoff_frequency = 3000  # Adjust as needed
    filtered_waveform = butter_lowpass_filter(waveform.numpy(), cutoff_frequency, sampling_rate)
    return torch.tensor(filtered_waveform)

def load_and_filter_audio(file_path, sr=16000, threshold=0.01):
    # Load audio file
    waveform, original_sr = librosa.load(file_path, sr=sr)
    
    # Compute energy of the signal
    energy = np.square(waveform)
    
    # Compute mean energy
    mean_energy = np.mean(energy)
    
    # Create a mask where energy is above the threshold
    mask = energy > (mean_energy * threshold)
    
    # Filter waveform based on the mask
    filtered_waveform = waveform[mask]
    
    # Ensure waveform is not empty
    if filtered_waveform.size == 0:
        raise ValueError("No speech detected in the audio.")
    
    return torch.tensor(filtered_waveform).unsqueeze(0)  # Add batch dimension

def chunk_waveform(waveform, chunk_length=16000*30, overlap_length=16000*5):
    waveform_length = waveform.shape[-1]
    chunks = []
    start = 0
    while start < waveform_length:
        end = min(start + chunk_length, waveform_length)
        chunks.append(waveform[:, start:end])
        start += chunk_length - overlap_length
    return chunks

def transcribe_chunk(model, processor, waveform_chunk):
    # Convert the waveform_chunk to numpy array
    waveform_np = waveform_chunk.squeeze().numpy()
    
    # Prepare inputs
    try:
        inputs = processor(waveform_np, sampling_rate=16000, return_tensors="pt", padding=True, return_attention_mask=True)
    except Exception as e:
        print(f"Error processing with WhisperProcessor: {e}")
        return "Error during processing"
    
    input_features = inputs["input_features"]
    attention_mask = inputs["attention_mask"]
    
    # Ensure input_features is a tensor
    if isinstance(input_features, list):
        input_features = torch.tensor(input_features)
    
    input_features = input_features.squeeze(0)  # Remove batch dimension if needed
    
    # Check if padding is required
    expected_length = 3000
    if input_features.size(1) < expected_length:
        padding = torch.zeros((input_features.size(0), expected_length - input_features.size(1)))
        input_features = torch.cat((input_features, padding), dim=1)
    
    # Generate transcription
    try:
        with torch.no_grad():
            if input_features.dim() == 2:  # if unbatched
                input_features = input_features.unsqueeze(0)  # Add batch dimension
            generated_ids = model.generate(input_features, attention_mask=attention_mask, max_length=448)
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    except Exception as e:
        print(f"Error during model generation: {e}")
        transcription = "Error during transcription"
    
    return transcription

def transcribe_audio(model, processor, waveform):
    chunks = chunk_waveform(waveform)
    full_transcription = []
    for chunk in chunks:
        transcription = transcribe_chunk(model, processor, chunk)
        full_transcription.append(transcription)
    return " ".join(full_transcription)

# Path to your audio file
audio_file_path = "audio_files/call_0.wav"

# Load and filter the audio file using librosa for VAD
filtered_waveform = load_and_filter_audio(audio_file_path)

# Debug: Print filtered_waveform shape
print(f"Filtered waveform shape: {filtered_waveform.shape}")

# Transcribe the filtered audio
transcription = transcribe_audio(model, processor, filtered_waveform)
print(f"Final Transcription: {transcription}")
