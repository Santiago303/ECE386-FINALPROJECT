import os
import sys
import requests
import subprocess
from fastapi import FastAPI, BackgroundTasks
import uvicorn
from pydantic import BaseModel
import importlib.util
import json
import RPi.GPIO as GPIO
import time
import threading

# Set up GPIO using BOARD numbering
GPIO.setmode(GPIO.BOARD)
BUTTON_PIN = 10  # GPIO pin 10 in BOARD numbering
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)  # Set up with pull-down resistor

# Import your existing modules
# Assuming speechrecognition.py is in the Whisper folder
whisper_path = os.path.join(os.path.dirname(__file__), "Whisper", "speechrecognition.py")
spec = importlib.util.spec_from_file_location("speechrecognition", whisper_path)
speechrecognition = importlib.util.module_from_spec(spec)
spec.loader.exec_module(speechrecognition)

# Import the llm_parse_for_wttr module
llm_parse_path = os.path.join(os.path.dirname(__file__), "llm_parse_for_wttr.py")
spec = importlib.util.spec_from_file_location("llm_parse_for_wttr", llm_parse_path)
llm_parse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(llm_parse)

# Initialize FastAPI app
app = FastAPI()

# Ollama server URL and model
OLLAMA_URL = "http://10.1.69.214:11434"
OLLAMA_MODEL = "gemma3:27b"  # Updated to use gemma3:27b model

# Flag to control recording
is_recording = False
recording_thread = None

class TranscriptionRequest(BaseModel):
    audio_path: str = None
    live_audio: bool = True
    duration: int = 5  # Default recording duration in seconds

class OllamaRequest(BaseModel):
    model: str
    prompt: str

def button_callback(channel):
    """Callback function for button press (rising edge)"""
    global is_recording
    if not is_recording:
        is_recording = True
        print("Button pressed! Starting recording...")
        # Start recording in a separate thread
        threading.Thread(target=record_and_process).start()

def record_and_process():
    """Record audio and process it when button is pressed"""
    global is_recording
    try:
        print("Recording started...")
        # Use the existing speechrecognition module to capture and transcribe
        transcription = speechrecognition.capture_and_transcribe(duration=5)
        print(f"Transcription: {transcription}")
        
        # Process with LLM
        ollama_request = {
            "model": OLLAMA_MODEL,  # Using gemma3:27b model
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

@app.post("/transcribe")
async def transcribe_audio(request: TranscriptionRequest):
    """Transcribe audio using Distil-Whisper"""
    try:
        if request.live_audio:
            # Use the existing speechrecognition module to capture and transcribe
            transcription = speechrecognition.capture_and_transcribe(duration=request.duration)
        else:
            # Use a pre-recorded audio file
            transcription = speechrecognition.transcribe_audio(request.audio_path)
            
        return {"status": "success", "transcription": transcription}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/process_with_llm")
async def process_with_llm(text: str):
    """Send text to Ollama LLM for processing"""
    try:
        # Prepare the request for Ollama
        ollama_request = {
            "model": OLLAMA_MODEL,  # Using gemma3:27b model
            "prompt": text
        }
        
        # Send request to Ollama
        response = requests.post(f"{OLLAMA_URL}/api/generate", json=ollama_request)
        response.raise_for_status()
        
        # Extract the response
        llm_response = response.json().get("response", "")
        
        return {"status": "success", "llm_response": llm_response}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/get_weather")
async def get_weather(text: str, background_tasks: BackgroundTasks):
    """Process text and get weather information"""
    try:
        # First transcribe the audio
        transcription_result = await transcribe_audio(TranscriptionRequest(live_audio=True))
        
        if transcription_result["status"] != "success":
            return transcription_result
            
        transcribed_text = transcription_result["transcription"]
        
        # Process with LLM if needed
        llm_result = await process_with_llm(transcribed_text)
        
        if llm_result["status"] != "success":
            return llm_result
            
        processed_text = llm_result["llm_response"]
        
        # Use the llm_parse module to extract weather query
        weather_query = llm_parse.parse_for_wttr(processed_text)
        
        # Call wttr.in API
        wttr_url = f"https://wttr.in/{weather_query}?format=j1"
        weather_response = requests.get(wttr_url)
        weather_response.raise_for_status()
        weather_data = weather_response.json()
        
        # Format and display the result in terminal
        formatted_result = format_weather_output(weather_data)
        print("\n" + "="*50)
        print("WEATHER INFORMATION:")
        print(formatted_result)
        print("="*50 + "\n")
        
        return {
            "status": "success", 
            "transcription": transcribed_text,
            "processed_query": weather_query,
            "weather_data": weather_data
        }
    except Exception as e:
        error_msg = f"Error getting weather: {str(e)}"
        print(error_msg)
        return {"status": "error", "message": error_msg}

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

@app.get("/")
async def root():
    return {"message": "Weather Voice Assistant API is running"}

def main():
    """Main function to run the GPIO-triggered voice assistant"""
    print("Starting Weather Voice Assistant with GPIO trigger...")
    print(f"Using Ollama model: {OLLAMA_MODEL} at {OLLAMA_URL}")
    print("Press the button connected to GPIO pin 10 to start recording")
    print("Press Ctrl+C to exit")
    
    # Add event detection for rising edge on button pin
    GPIO.add_event_detect(BUTTON_PIN, GPIO.RISING, callback=button_callback, bouncetime=300)
    
    try:
        # Keep the program running
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nExiting Weather Voice Assistant. Goodbye!")
    finally:
        # Clean up GPIO on exit
        GPIO.cleanup()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        # Run in CLI mode with GPIO trigger
        main()
    else:
        # Run as FastAPI server
        # Start GPIO monitoring in a separate thread
        threading.Thread(target=main, daemon=True).start()
        uvicorn.run(app, host="0.0.0.0", port=8000)