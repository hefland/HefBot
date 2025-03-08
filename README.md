# HefBot
Offline LLM Voice Chat using Ollama

Installation Instructions

Prerequisites

Python 3.12.1 – This version has been tested with the app.
NVIDIA GPU with CUDA 12.8 and cuDNN 9.7.1 – Required for GPU acceleration.

Steps
Clone the Repository
Open Command Prompt (or Terminal) and type the following commands:

bash
Copy
git clone https://github.com/yourusername/yourrepository.git
cd yourrepository
Note: The cd yourrepository command (short for “change directory”) moves you into the folder that contains your project files.

Install Python 3.12.1

Go to python.org/downloads and download Python 3.12.1 for Windows.
Run the installer and check the box “Add Python 3.12 to PATH” before installing.
Create a Virtual Environment
In the Command Prompt (while inside your repository folder), run:

bash
Copy
py -3.12 -m venv venv
This creates a new folder called venv that contains an isolated Python environment.

Activate the Virtual Environment
For Windows, type:

bash
Copy
venv\Scripts\activate
Once activated, your prompt should change to include (venv).

Install Required Python Packages
With your virtual environment active, run:

bash
Copy
pip install -r requirements.txt
This installs all the dependencies listed in the requirements.txt file.

Install CUDA 12.8 and cuDNN 9.7.1

a. Install CUDA 12.8:

Visit the CUDA Toolkit 12.8 Download Archive and download the installer for Windows.
Run the installer and install CUDA (the default installation path is usually:
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8).
b. Install cuDNN 9.7.1:

Go to the cuDNN Archive (you’ll need to sign up for a free NVIDIA Developer account) and download cuDNN 9.7.1 for CUDA 12.8.
Unzip the downloaded cuDNN package. You’ll see folders such as bin, include, and lib.
Copy the files as follows:
From the cuDNN bin folder:
Copy all files to:
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin
From the cuDNN include folder:
Copy all files to:
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include
From the cuDNN lib folder:
Copy all files (usually inside a subfolder like x64) to:
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\lib\x64
c. Set Environment Variables:

Open the Control Panel, go to System and Security > System > Advanced system settings > Environment Variables.
Under System Variables, find and select the Path variable, then click Edit.
Add the following paths (click New for each):
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin
(Optionally) C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\libnvvp
Click OK to save the changes.
Run the Application
In your Command Prompt (with the virtual environment still activated), type:

bash
Copy
python assistant.py
The app should now launch.
