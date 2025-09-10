# Whisper Installation Guide for Windows (Cantonese Speech-to-Text)

This guide provides step-by-step instructions for installing OpenAI's Whisper on a Windows system, with a focus on transcribing Cantonese audio.

## 1. Prerequisites

### Python Environment
- **Supported Versions:** Python 3.8-3.11
- **Recommended:** Python 3.9 or 3.10

#### Check your Python version
Open Command Prompt (cmd) or PowerShell and run:
```bash
python --version
```
If Python is not installed, download it from the [official Python website](https://www.python.org/downloads/). **Important:** During installation, make sure to check the box that says **"Add Python to PATH"**.

## 2. Install FFmpeg (Required)

Whisper requires the FFmpeg library to process audio files. Choose one of the following methods for installation.

### Option A: Using a Package Manager (Recommended)

Using a package manager like [Chocolatey](https://chocolatey.org/) is the easiest way to install and manage FFmpeg.

1.  **Install Chocolatey** (if you don't have it):
    Open PowerShell **as Administrator** and run:
    ```powershell
    Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
    ```
2.  **Install FFmpeg using Chocolatey**:
    In a new Administrator Command Prompt or PowerShell, run:
    ```bash
    choco install ffmpeg
    ```

### Option B: Manual Installation

1.  Download a pre-built FFmpeg package from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) or the [official site](https://ffmpeg.org/download.html).
2.  Extract the downloaded `.zip` file to a permanent location, for example, `C:\ffmpeg`.
3.  Add the `bin` directory from the extracted folder (e.g., `C:\ffmpeg\bin`) to your system's **PATH environment variable**.
4.  Restart your Command Prompt or PowerShell and verify the installation:
    ```bash
    ffmpeg -version
    ```

## 3. Install Whisper

Open Command Prompt or PowerShell to install Whisper using `pip`.

#### Standard Installation
For the latest stable version from PyPI:
```bash
pip install -U openai-whisper
```

#### Development Version
To get the latest updates directly from the GitHub repository:
```bash
pip install git+https://github.com/openai/whisper.git
```

## 4. Optional: Install GPU Support (for NVIDIA GPUs)

If you have an NVIDIA graphics card, you can significantly speed up transcription by installing PyTorch with CUDA support.

1.  **Uninstall existing PyTorch** (if any):
    ```bash
    pip uninstall torch torchvision torchaudio
    ```
2.  **Install PyTorch with CUDA**:
    The command below is for CUDA 11.8. Check the [PyTorch website](https://pytorch.org/get-started/locally/) for the command matching your specific CUDA version.
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

## 5. Verify Installation

Run these commands to ensure Whisper is installed correctly.

#### Test the command-line tool
You should see a list of available options and commands.
```bash
whisper --help
```

#### Test the Python module
If this command runs without errors and prints the message, the installation was successful.
```bash
python -c "import whisper; print('Whisper installed successfully!')"
```

## 6. Usage for Cantonese Transcription

Here’s how to use Whisper to transcribe a Cantonese audio file (e.g., `audio.mp3`).

#### Command-Line Usage

Whisper can detect the language, but specifying it can improve accuracy. For Cantonese, use `Chinese`.

```bash
# Basic transcription using the 'base' model
whisper audio.mp3 --model base --language Chinese

# For higher accuracy, use a larger model (e.g., 'medium')
whisper audio.mp3 --model medium --language Chinese

# To generate subtitles (e.g., SRT format)
whisper audio.mp3 --model medium --language Chinese --output_format srt
```

#### Python Script Usage

```python
import whisper

# Load a model (e.g., "base", "small", "medium", "large")
model = whisper.load_model("medium")

# Transcribe the audio file, specifying the language as "zh" for Chinese
result = model.transcribe("your_cantonese_audio.mp3", language="zh")

# Print the transcribed text
print(result["text"])

# Save the transcription to a text file
with open("transcription.txt", "w", encoding="utf-8") as f:
    f.write(result["text"])

print("Transcription saved to transcription.txt")
```

### Available Models

Choose a model based on your needs for speed vs. accuracy.

| Model  | Parameters | Multilingual | Required VRAM | Relative Speed |
|:-------|:-----------|:-------------|:--------------|:---------------|
| `tiny`   | 39 M       | Yes          | ~1 GB         | ~32x           |
| `base`   | 74 M       | Yes          | ~1 GB         | ~16x           |
| `small`  | 244 M      | Yes          | ~2 GB         | ~6x            |
| `medium` | 769 M      | Yes          | ~5 GB         | ~2x            |
| `large`  | 1550 M     | Yes          | ~10 GB        | 1x             |

## 7. Troubleshooting Common Issues

### Error: `No module named 'setuptools_rust'`
This error means a required build dependency is missing. Install it with pip:
```bash
pip install setuptools-rust
```

### Rust Environment Required
Some dependencies may need the Rust compiler.
1.  Visit [rustup.rs](https://rustup.rs/) and follow the instructions to install Rust.
2.  Download and run the installer.
3.  Restart your Command Prompt or PowerShell.
4.  Ensure `%USERPROFILE%\.cargo\bin` is in your PATH.

### Slow `pip` Installation
If your download is slow, use a regional mirror.
```bash
# Example using a mirror in China
pip install -U openai-whisper -i https://pypi.tuna.tsinghua.edu.cn/simple/
```
