# ECE386-FINALPROJECT
# BLUF
was not able to demo due to server issues.

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
4.Run the application
  python weatherNoGPIO.py
  OR
  python weather_assistant.py

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
