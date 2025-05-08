# ECE386-FINALPROJECT
# BLUF
was not able to use keyword, KWS was not working to begin with. was not able to use GPIO for some weird reason that i couldnt figure out. I decided to take in input from the keyboard instead, hopefully "button" enough... Additionally the code does have a fallback that uses the CPU when there is no GPU located. Finally... it works. Note that i did figure out how to run sudo on a virtual enviorment but the GPIO for some reason did not want to work on me even after running it in sudo.

## Overview
This application allows you to:

-Record audio from a microphone
-Transcribe speech to text using Whisper
-Process the text with an LLM (Gemma 3 27B)
-Extract weather queries
-Fetch and display weather information

## HOW TO USE
1. Clone Repo
   
2. Set up virtual enviorment
  #Create a virtual environment
  python3 -m venv venv
  #Activate the virtual environment
  source venv/bin/activate

3.Install dependencies
#Install required system packages
sudo apt-get update
sudo apt-get install -y python3-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg

#install python dependencies
pip install -r requirements.txt
  
4.Run the application
#For normal execution
python3 weatherNoGPIO.py

#For execution with root privileges (required for keyboard monitoring)
sudo /path/to/venv/bin/python3 weatherNoGPIO.py

# API Endpoints
The application exposes several API endpoints:

GET /: Web interface
POST /record_and_transcribe: Record and transcribe audio
POST /process_with_llm: Process text with LLM
POST /get_weather: Complete workflow (record, transcribe, process, get weather)

# Configuration
You can modify the following variables in the script:

OLLAMA_URL: URL of your Ollama server
OLLAMA_MODEL: LLM model to use
SAMPLE_RATE: Audio sample rate
CHANNELS: Audio channels
TEMP_AUDIO_FILE: Path to temporary audio file
