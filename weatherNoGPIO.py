import os
import sys
import requests
import subprocess
import threading
import time
import importlib.util
import json
import sounddevice as sd
import numpy as np
import wave
import keyboard  

# Import your existing modules
# Assuming speechrecognition.py is in the Whisper folder
whisper_path = os.path.join(os.path.dirname(__file__), "Whisper", "speech_recognition.py")
spec = importlib.util.spec_from_file_location("speech_recognition", whisper_path)
speechrecognition = importlib.util.module_from_spec(spec)
spec.loader.exec_module(speechrecognition)

# Import the llm_parse_for_wttr module
llm_parse_path = os.path.join(os.path.dirname(__file__), "llm_parse_for_wttr.py")
spec = importlib.util.spec_from_file_location("llm_parse_for_wttr", llm_parse_path)
llm_parse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llm_parse)

# Ollama server URL and model
OLLAMA_URL = "http://10.1.69.214:11434"
OLLAMA_MODEL = "gemma3:27b"  # Using gemma3:27b model

# Flag to control recording
is_recording = False

# Audio configuration
SAMPLE_RATE = 16000
CHANNELS = 1
TEMP_AUDIO_FILE = "/tmp/recorded_audio.wav"

def record_audio(duration=5, sample_rate=SAMPLE_RATE, channels=CHANNELS):
    """Record audio from the microphone"""
    print(f"Recording for {duration} seconds...")
    
    # Record audio
    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        dtype='int16'
    )
    sd.wait()  # Wait until recording is finished
    
    # Save to temporary WAV file
    with wave.open(TEMP_AUDIO_FILE, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit audio
        wf.setframerate(sample_rate)
        wf.writeframes(recording.tobytes())
    
    print(f"Audio saved to {TEMP_AUDIO_FILE}")
    return TEMP_AUDIO_FILE

def record_and_process():
    """Record audio and process it"""
    global is_recording
    try:
        is_recording = True
        print("Recording started...")
        
        # Record audio
        audio_file = record_audio(duration=5)
        
        # Use the existing speechrecognition module to transcribe
        transcription = speechrecognition.transcribe_audio(audio_file)
        print(f"Transcription: {transcription}")
        
        # Process with LLM
        ollama_request = {
            "model": OLLAMA_MODEL,
            "prompt": transcription
        }
        
        # Send request to Ollama
        print(f"Sending request to Ollama ({OLLAMA_MODEL})...")
        response = requests.post(f"{OLLAMA_URL}/api/generate", json=ollama_request)
        response.raise_for_status()
        
        # Extract the response
        llm_response = response.json().get("response", "")
        print(f"LLM Response: {llm_response}")
        
        # Use the llm_parse module to extract weather query
        weather_query = llm_parse.parse_for_wttr(llm_response)
        print(f"Parsed weather query: {weather_query}")
        
        # Call wttr.in API
        wttr_url = f"https://wttr.in/{weather_query}?format=j1"
        print(f"Fetching weather data for: {weather_query}")
        weather_response = requests.get(wttr_url)
        weather_response.raise_for_status()
        weather_data = weather_response.json()
        
        # Format and display the result in terminal
        formatted_result = format_weather_output(weather_data)
        print("\n" + "="*50)
        print("WEATHER INFORMATION:")
        print(formatted_result)
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"Error in recording and processing: {str(e)}")
    finally:
        is_recording = False
        print("Ready for next recording.")

def format_weather_output(weather_data):
    """Format weather data for terminal display"""
    try:
        location = weather_data.get("nearest_area", [{}])[0].get("areaName", [{}])[0].get("value", "Unknown location")
        current = weather_data.get("current_condition", [{}])[0]
        temp_c = current.get("temp_C", "N/A")
        temp_f = current.get("temp_F", "N/A")
        desc = current.get("weatherDesc", [{}])[0].get("value", "N/A")
        humidity = current.get("humidity", "N/A")
        
        forecast = weather_data.get("weather", [])
        
        output = f"Weather for {location}:\n"
        output += f"Current conditions: {desc}\n"
        output += f"Temperature: {temp_c}°C / {temp_f}°F\n"
        output += f"Humidity: {humidity}%\n\n"
        
        if forecast:
            output += "Forecast:\n"
            for day in forecast[:3]:  # Next 3 days
                date = day.get("date", "N/A")
                max_temp = day.get("maxtempC", "N/A")
                min_temp = day.get("mintempC", "N/A")
                desc = day.get("hourly", [{}])[0].get("weatherDesc", [{}])[0].get("value", "N/A")
                
                output += f"{date}: {desc}, {min_temp}°C to {max_temp}°C\n"
        
        return output
    except Exception as e:
        return f"Error formatting weather data: {str(e)}"

def on_space_press(e):
    """Callback for space key press"""
    global is_recording
    if not is_recording:
        print("Space pressed! Starting recording...")
        threading.Thread(target=record_and_process).start()

def main():
    """Main function to run the keyboard-triggered voice assistant"""
    print("Starting Weather Voice Assistant with keyboard trigger...")
    print(f"Using Ollama model: {OLLAMA_MODEL} at {OLLAMA_URL}")
    print("Press SPACE to start recording")
    print("Press Ctrl+C to exit")
    
    # Register the space key callback
    keyboard.on_press_key('space', on_space_press)
    
    try:
        # Keep the program running
        keyboard.wait('esc')  # Wait until ESC is pressed
    except KeyboardInterrupt:
        print("\nExiting Weather Voice Assistant. Goodbye!")

if __name__ == "__main__":
    main()