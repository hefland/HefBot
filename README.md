# HefBot
Offline LLM Voice Chat using Ollama

Hef-Bot is a voice-activated AI assistant designed to let you interact using both voice and text. It leverages state-of-the-art speech recognition, text-to-speech, and natural language processing to provide a conversational experience. Whether you want to ask about current events, check the weather, or simply have a conversation, Hef-Bot is here to help.

## Features

- **Voice & Text Interaction:**  
  Speak to or type your queries and get responses both visually and through speech.

- **Accurate Speech Recognition:**  
  Uses the Whisper model to transcribe your spoken words accurately.

- **Natural Text-to-Speech:**  
  Converts LLM responses to natural-sounding audio using Kokoro.

- **Custom Commands:**  
  - **Wake Word ("computer"):** The app listens for this word to start recording.
  - **Interrupt Command ("wait a minute"):** Interrupt TTS playback and start recording immediately.
  - **Cancel Command ("cancel that"):** Cancel the current recording and stop all ongoing processes.

- **Conversation Management:**  
  The app manages conversation state visually (with color cues and status messages) and plays sound effects when conversations start and end.

- **Extensibility:**  
  Future plans include integrating Retrieval Augmented Generation (RAG) to allow real-time data access and file uploads for document referencing.

- **GPU Acceleration:**  
  With CUDA 12.8 and cuDNN 9.7.1, the app can take advantage of NVIDIA GPU acceleration for improved performance.

## Installation Instructions

### Prerequisites

- **Python 3.12.1** – This version has been tested with the app.
- **NVIDIA GPU with CUDA 12.8 and cuDNN 9.7.1** – Required for GPU acceleration.
- Please note: you’ll need to sign up for a free NVIDIA Developer account to be able to download cuDNN.

### Steps

1. **Clone the Repository**
   Open Command Prompt (or Terminal) and type the following commands:
   ```bash
   git clone https://github.com/hefland/HefBot.git
   ```
2. **then enter that folder**:
   ```bash
   cd HefBot
   ```

3. **Install Python 3.12.1**
   - Go to [python.org/downloads](https://www.python.org/downloads) and download Python 3.12.1 for Windows.
   - Run the installer and check the box “Add Python 3.12 to PATH” before installing.

4. **Create a Virtual Environment**
   In the Command Prompt (while inside your repository folder), run:
   ```bash
   py -3.12 -m venv venv
   ```
   This creates a new folder called `venv` that contains an isolated Python environment.

5. **Activate the Virtual Environment**
   For Windows, type:
   ```bash
   venv\Scripts\activate
   ```
   Once activated, your prompt should change to include `(venv)`.

6. **Install Required Python Packages**
   With your virtual environment active, run:
   ```bash
   pip install -r requirements.txt
   ```
   This installs all the dependencies listed in the `requirements.txt` file.

7. **Install CUDA 12.8 and cuDNN 9.7.1**
   a. **Install CUDA 12.8:**
      - Visit the [CUDA Toolkit 12.8 Download Archive](https://developer.nvidia.com/cuda-toolkit-archive) and download the installer for Windows.
      - Run the installer and install CUDA (the default installation path is usually: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8`).

   b. **Install cuDNN 9.7.1:**
      - Go to the [cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive) (you’ll need to sign up for a free NVIDIA Developer account) and download cuDNN 9.7.1 for CUDA 12.8.
      - Unzip the downloaded cuDNN package. You’ll see folders such as `bin`, `include`, and `lib`.
      - Copy the files as follows:
        - From the cuDNN `bin` folder: Copy all files to: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin`
        - From the cuDNN `include` folder: Copy all files to: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include`
        - From the cuDNN `lib` folder: Copy all files (usually inside a subfolder like `x64`) to: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64`

   c. **Set Environment Variables:**
      - Open the Control Panel, go to System and Security > System > Advanced system settings > Environment Variables.
      - Under System Variables, find and select the `Path` variable, then click `Edit`.
      - Add the following paths (click `New` for each):
        - `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin`
        - (Optionally) `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\libnvvp`
      - Click `OK` to save the changes.

8. **Run the Application**
   In your Command Prompt (with the virtual environment still activated), type:
   ```bash
   python assistant.py
   ```
   The app should now launch.
