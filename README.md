# Alternative-TTS-Elevenlabs-Chatterbox
Chatterbox TTS Demo 

# Chatterbox TTS on GitHub Codespaces (CPU, Python 3.11)

This repository documents a reliable way to run **Chatterbox Text-to-Speech** in **GitHub Codespaces**, even though the default system Python version is **3.12**.

The setup uses **Python 3.11**, **CPU-only PyTorch**, and **Hugging Face authentication** to generate WAV audio files successfully.

---

## Requirements

* GitHub Codespaces (Ubuntu)
* Minimum 4 vCPUs and 8 GB RAM recommended or Above
* Internet access for model downloads

---

## Overview

GitHub Codespaces currently ships with Python 3.12, which is not supported by Chatterbox.
Instead of removing system Python, this setup installs Python 3.11 using `pyenv` and isolates it inside a virtual environment.

CUDA is not available in Codespaces, so PyTorch must be installed using **CPU-only wheels**.

---

## Step 1: Install system build dependencies

These packages are required to compile Python using `pyenv`.

```bash
sudo apt update
sudo apt install -y \
  make build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev \
  wget curl llvm libncursesw5-dev xz-utils \
  tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

---

## Step 2: Install pyenv

```bash
curl https://pyenv.run | bash
```

Add pyenv to the shell configuration:

```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc
```

Verify installation:

```bash
pyenv --version
```

---

## Step 3: Install Python 3.11

```bash
pyenv install 3.11.8
pyenv local 3.11.8
```

Verify:

```bash
python --version
```

Expected output:

```
Python 3.11.8
```

---

## Step 4: Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

Verify:

```bash
which python
```

---

## Step 5: Upgrade pip tooling

```bash
pip install --upgrade pip setuptools wheel
```

---

## Step 6: Install CPU-only PyTorch (critical)

Do not install default or CUDA-enabled PyTorch builds.
CUDA wheels will fail in GitHub Codespaces.

```bash
pip uninstall -y torch torchvision torchaudio triton

pip install \
  torch==2.2.2 \
  torchvision==0.17.2 \
  torchaudio==2.2.2 \
  --index-url https://download.pytorch.org/whl/cpu
```

Verify:

```bash
python - <<EOF
import torch, torchvision, torchaudio
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("torchaudio:", torchaudio.__version__)
print("cuda available:", torch.cuda.is_available())
EOF
```

Expected:

```
torch: 2.2.2
torchvision: 0.17.2
torchaudio: 2.2.2
cuda available: False
```

---

## Step 7: Install Chatterbox

```bash
pip install chatterbox-tts
```

Note:
You may see a warning that Chatterbox requires torch 2.6.0.
This warning can be ignored for CPU usage.

---

## Step 8: Authenticate with Hugging Face

Chatterbox models are hosted on Hugging Face and require authentication.

1. Create a token at:
   [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   Use a **Read** token.

2. Log in from the terminal:

```bash
huggingface-cli login
```

Paste the token when prompted.

---

## Step 9: Generate speech

Create a file named `test_tts.py`:

```python
import torch
import numpy as np
import soundfile as sf
from chatterbox.tts_turbo import ChatterboxTurboTTS

tts = ChatterboxTurboTTS.from_pretrained(device="cpu")

audio = tts.generate(
    "This audio was generated using Chatterbox in GitHub Codespaces."
)

if isinstance(audio, torch.Tensor):
    audio = audio.detach().cpu().numpy()

audio = np.squeeze(audio)

sf.write("output.wav", audio, samplerate=24000)
```

Run the script:

```bash
python test_tts.py
```

The first run will download the model and may take several minutes.

---

## Step 10: Download the audio file

* Open the Explorer panel in Codespaces
* Right-click `output.wav`
* Select **Download**
* Play the file locally

---

## Common Issues

### `operator torchvision::nms does not exist`

Cause: CUDA-enabled PyTorch wheels installed
Fix: Reinstall PyTorch using the CPU-only index URL

---

### `TypeError: a bytes-like object is required`

Cause: Chatterbox returned a Tensor instead of raw bytes
Fix: Convert the Tensor to NumPy before writing to WAV (already handled above)

---

### Hugging Face token error

Cause: Not logged in or token missing
Fix: Run `huggingface-cli login` again

---

## Result

This setup provides:

* Python 3.11 isolated from system Python 3.12
* Stable CPU-only PyTorch environment
* Fully working Chatterbox TTS
* WAV audio generation inside GitHub Codespaces

---

## Optional Next Steps

* Voice cloning
* FastAPI TTS service
* Multilingual TTS
* Docker deployment
* Integration with messaging or voice bots


