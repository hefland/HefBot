# Hef-Bot

Hef-Bot is a voice-activated AI assistant that lets you interact using both voice and text. It uses advanced speech recognition, text-to-speech, and a state-of-the-art language model to provide a natural interaction experience.

## Features

- **Voice & Text Interaction:**  
  Speak or type your queries to interact with the assistant.
  
- **Accurate Speech Recognition:**  
  Uses the Whisper model for high-quality transcription.

- **Natural Text-to-Speech:**  
  Uses Kokoro for converting LLM responses into natural-sounding audio.

- **Custom Commands:**  
  - **Wake Word ("computer"):** Activates voice recording.  
  - **Interrupt Command ("wait a minute"):** Interrupts TTS and starts recording immediately.  
  - **Cancel Command ("cancel that"):** Cancels the current recording.  
  - **Interrupt Command ("wait a minute" or your chosen phrase):** (Optional, see below for changing this phrase.)

- **Ollama Integration:**  
  Hef-Bot uses the Ollama framework to power its LLM. You must download and install Ollama and the appropriate model (e.g., `deepseek-r1:32b`).  
  - Download Ollama from [ollama.com](https://ollama.com).  
  - Follow the instructions on the Ollama website to install it and download the required model.

- **Extensible with RAG:**  
  Future plans include integrating Retrieval-Augmented Generation (RAG) so that the LLM can access real-time data and reference user-uploaded documents.

- **GPU Acceleration:**  
  Uses CUDA 12.8 and cuDNN 9.7.1 for improved performance.

## Installation Instructions

### Prerequisites

- **Python 3.12.1** – (Use `py -3.12` on Windows if you have multiple versions.)
- **CUDA 12.8** – Required for GPU acceleration. Download from [CUDA Toolkit 12.8 Download Archive](https://developer.nvidia.com/cuda-12-8-download-archive).
- **cuDNN 9.7.1** – Required for GPU acceleration. Download from [cuDNN 9.7.1 Download Archive](https://developer.nvidia.com/cudnn-9-7-1-download-archive).
- **Ollama** – Download and install from [ollama.com](https://ollama.com). Make sure to download the required model (e.g., `deepseek-r1:32b`) as per the instructions on the Ollama website.

### Steps

1. **Clone the Repository**

   Open Command Prompt (or Terminal) and type:

   ```bash
   git clone https://github.com/hefland/HefBot.git
   ```
   then:

   ```bash
   cd HefBot
   ```
2. **Install Python 3.12.1**

Download Python 3.12.1 from python.org/downloads and run the installer. Make sure to check “Add Python 3.12 to PATH” during installation.

3. **Create a Virtual Environment**

In the repository folder, run:

```bash
py -3.12 -m venv venv
```
This creates an isolated Python environment in a folder named venv.

4. **Activate the Virtual Environment**

Windows:

```bash
venv\Scripts\activate
```

macOS/Linux:

```bash
source venv/bin/activate
```

Your prompt should now display (venv).

5. **Install Python Dependencies**

With the virtual environment active, run:

```bash
pip install -r requirements.txt
```
Your requirements.txt should contain:
```
plaintext
PyQt5
numpy
sounddevice
soundfile
faster-whisper
kokoro
langchain
langchain_community
pocketsphinx
webrtcvad
pyenchant
```

6. **Install and Configure CUDA 12.8 and cuDNN 9.7.1**

a. Install CUDA 12.8: Visit the [CUDA Toolkit 12.8 Download Archive](https://developer.nvidia.com/cuda-12-8-download-archive) and download the installer for Windows. Run the installer and install CUDA (default path is usually C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8).

b. Install cuDNN 9.7.1: Go to the [cuDNN 9.7.1 Download Archive](https://developer.nvidia.com/cudnn-9-7-1-download-archive) (you will need a free NVIDIA Developer account) and download cuDNN 9.7.1 for CUDA 12.8. Unzip the cuDNN package. You will see folders such as bin, include, and lib.

From the cuDNN/bin folder, copy all files to:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin
```

From the cuDNN/include folder, copy all files to:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include
```

From the cuDNN/lib folder (typically within a subfolder like x64), copy all files to:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64
```

c. Set Environment Variables:
Open Control Panel → System and Security → System → Advanced system settings → Environment Variables. Under System Variables, select the Path variable, then click Edit. Click New and add:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin
```
```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\libnvvp
```

Click OK to save.

7. **Install and Run Ollama**

Download and install Ollama from [ollama.com](https://ollama.com). Follow the instructions on the Ollama website to install it and download the required model (e.g., deepseek-r1:32b). Ensure Ollama is running in the background.

7. **Run the Application**

With your virtual environment still activated, type:

```bash
python assistant.py
```

Hef-Bot should now launch and you can start interacting with it!
