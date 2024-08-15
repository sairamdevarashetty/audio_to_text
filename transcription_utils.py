# transcription_utils.py

from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torch
import numpy as np

def transcribe_audio(audio: np.ndarray) -> dict:
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    inputs = tokenizer(audio, return_tensors="pt", padding="longest")
    with torch.no_grad():
        logits = model(input_values=inputs.input_values).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)

    return {
        "segments": [{
            "start": 0,
            "end": len(audio) / 16000,
            "text": transcription[0]
        }]
    }