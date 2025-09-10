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

Of course. Here is that section translated into English, formatted perfectly for your GitHub README file.

This new section provides an advanced option for users who need higher accuracy and more authentic results for Cantonese.

---

### ⭐ Advanced Option: For More Authentic Cantonese Transcription

While the standard Whisper model is powerful, it tends to transcribe spoken Cantonese into **standard written Chinese**. To achieve better results that recognize and output **authentic colloquial Cantonese words** (e.g., `喺`, `嘅`, `唔該`, `食晏`), using a model fine-tuned specifically on Cantonese datasets is highly recommended.

A high-performing community model for this is **`alvanlii/whisper-small-cantonese`**.

#### Feature Comparison

| Feature | Standard Whisper (`openai/whisper`) | Fine-tuned Model (`alvanlii/whisper-small-cantonese`) |
| :--- | :--- | :--- |
| **Output Style** | Standard Written Chinese | **Authentic Spoken Cantonese** |
| **Accuracy** | Good | **Higher** (lower CER on Cantonese test sets) |
| **Installation** | `pip install openai-whisper` | `pip install transformers accelerate` |
| **Best For** | General, multi-language transcription | Users needing high accuracy and authentic Cantonese text |

---

### How to Use the `alvanlii/whisper-small-cantonese` Model

#### Step 1: Install Required Libraries

You'll need the Hugging Face `transformers` library to load and run this model.

```bash
pip install torch transformers accelerate
```
*   `torch`: The core PyTorch framework.
*   `transformers`: The main Hugging Face library.
*   `accelerate`: Helps optimize model loading and execution on different hardware (CPU/GPU).

#### Step 2: Transcribe Using a Python Script

The easiest way to use the model is with the `pipeline` function, which handles all the complex pre-processing and post-processing steps for you.

```python
import torch
from transformers import pipeline

# Automatically detect if a GPU is available, otherwise use CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Define the model ID from Hugging Face
MODEL_NAME = "alvanlii/whisper-small-cantonese"

# Create the automatic speech recognition pipeline
# chunk_length_s=30 is useful for long audio files, splitting them into 30-second chunks
pipe = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_NAME,
    chunk_length_s=30,
    device=device,
)

# Forcing the decoder can improve accuracy by prompting it for the correct language and task
pipe.model.config.forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(language="zh", task="transcribe")

# Specify the path to your Cantonese audio file
audio_file = "path/to/your/cantonese_audio.mp3"

# Perform the transcription
result = pipe(audio_file)

# Print the transcribed text
print(result["text"])
```

#### Pro Tip: Accelerate Inference with Flash Attention 2

If you have a compatible NVIDIA GPU (e.g., RTX 30/40 series), you can dramatically speed up processing by installing and enabling Flash Attention 2.

1.  **Install Flash Attention**:
    ```bash
    pip install flash-attn --no-build-isolation
    ```

2.  **Enable it when loading the model**:
    Modify the model loading part of your script to include the `attn_implementation` argument.

    ```python
    # This block replaces the initial pipeline creation in the script above
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    # Use float16 for better performance on GPU
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load the model with Flash Attention enabled
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        attn_implementation="flash_attention_2" # <-- Enable Flash Attention
    )
    model.to(device)
    
    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    # Re-create the pipeline with the optimized model
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    # ... your transcription code follows here ...
    ```
    According to the model's author, enabling Flash Attention reduces the processing time per sample from **0.308s** to just **0.055s**—a massive speed improvement.
