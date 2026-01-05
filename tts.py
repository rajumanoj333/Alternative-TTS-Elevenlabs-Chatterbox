import torch
import soundfile as sf
import numpy as np
from chatterbox.tts_turbo import ChatterboxTurboTTS

tts = ChatterboxTurboTTS.from_pretrained(device="cpu")

audio = tts.generate(
    "Hello Manoj. This audio is correctly generated in GitHub Codespaces."
)

# 🔥 FIX: handle Tensor output
if isinstance(audio, torch.Tensor):
    audio = audio.detach().cpu().numpy()

# Ensure correct shape
audio = np.squeeze(audio)

# Write WAV
sf.write("output.wav", audio, samplerate=24000)

print("✅ output.wav generated successfully")
