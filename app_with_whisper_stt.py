import os
import tempfile
import wave
import threading
import time

import gradio as gr
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastrtc import (
    AdditionalOutputs,
    ReplyOnPause,
    Stream,
    AlgoOptions,
)
from openai import OpenAI
from numpy.typing import NDArray
from kokoro import KPipeline

"""
Voice Chat Application
---------------------
This application provides a voice interface for chatting with a language model.
It uses:
- SambaNova's LLM API for chat responses
- OpenAI's Whisper for speech-to-text
- Kokoro for text-to-speech
- WebRTC for real-time audio streaming
"""

# Global client objects
sambanova_client = None
openai_client = None
tts_pipeline = None

# Flag to track if TTS is currently speaking
is_tts_speaking = False
tts_lock = threading.Lock()

# Global stream object
stream_instance = None


def initialize_models():
    """
    Initialize all required models and clients.
    """
    global sambanova_client, openai_client, tts_pipeline
    
    print("Initializing models...")
    
    # Load environment variables
    load_dotenv()
    
    # Initialize SambaNova client
    sambanova_client = OpenAI(
        api_key=os.environ.get("SAMBANOVA_API_KEY"),
        base_url="https://api.sambanova.ai/v1"
    )
    
    # Initialize OpenAI client for Whisper
    print("Setting up OpenAI client for Whisper STT...")
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    print("OpenAI client ready")
    
    # Initialize Kokoro TTS pipeline
    print("Loading Kokoro TTS model...")
    tts_pipeline = KPipeline(lang_code='a')
    print("Kokoro TTS model loaded")
    
    print("All models initialized")


def whisper_stt(audio_data: NDArray, sample_rate: int = 16000) -> str:
    """
    Use OpenAI's Whisper model for speech-to-text transcription.
    """
    # Create a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Ensure audio_data is the right shape and type
        if len(audio_data.shape) > 1:
            # If stereo or multi-channel, convert to mono by averaging channels
            audio_data = np.mean(audio_data, axis=0)
        
        # Normalize audio data to be in the range [-1, 1]
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Apply a simple noise gate to filter out background noise
        # Calculate the RMS (root mean square) of the audio signal
        rms = np.sqrt(np.mean(audio_data**2))
        print(f"Audio RMS: {rms}")
        
        # Set a threshold for the noise gate (adjust as needed)
        noise_threshold = 0.095  # This is a relative value
        
        # Apply the noise gate
        if rms < noise_threshold:
            print(f"Audio level too low (RMS: {rms:.5f}), likely just background noise")
            return ""
        
        # Optional: Apply a noise gate at the sample level
        # This will zero out samples below the threshold
        sample_threshold = 0.01  
        audio_data = np.where(np.abs(audio_data) < sample_threshold, 0, audio_data)
        
        # Save as WAV file with proper parameters
        with wave.open(temp_path, 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            
            # Convert float32 [-1.0, 1.0] to int16 [-32768, 32767]
            audio_int16 = (audio_data * 32767).astype(np.int16)
            wf.writeframes(audio_int16.tobytes())
        
        # Verify the file exists and has content
        if os.path.getsize(temp_path) == 0:
            print("Error: Generated WAV file is empty")
            return ""
            
        # Open the file and send to Whisper API
        with open(temp_path, "rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en"
            )
        
        return transcript.text
    except Exception as e:
        print(f"Whisper STT Error: {e}")
        return ""
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)


def response(
    audio: tuple[int, NDArray[np.int16 | np.float32]],
    chatbot: list[dict] | None = None,
):
    """
    Process audio input, transcribe it, get LLM response, and generate TTS output.
    """
    global is_tts_speaking
    
    # Check if TTS is currently speaking
    with tts_lock:
        if is_tts_speaking:
            print("TTS is currently speaking, ignoring audio input")
            return
    
    # Initialize models if they haven't been initialized yet
    if sambanova_client is None or openai_client is None or tts_pipeline is None:
        initialize_models()
        
    chatbot = chatbot or []
    
    # Process audio input
    sample_rate, audio_data = audio
    
    # Convert to float32 if not already
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    
    # Use Whisper for transcription with audio
    start = time.time()
    text = whisper_stt(audio_data, sample_rate)
    print(f"Transcription time: {time.time() - start:.2f}s")
    
    # Skip if no text was transcribed
    if not text:
        print("No speech detected")
        return
        
    print("\n----- USER INPUT -----")
    print(text)
    print("----------------------\n")
    
    # Create a new message for this user input
    chatbot.append({"role": "user", "content": text})
    # Update the UI immediately to show the new message
    yield AdditionalOutputs(chatbot)
    
    # Create a new message list for the LLM
    messages = [{"role": d["role"], "content": d["content"]} for d in chatbot]
    
    # Add system message to instruct the model to keep responses short and simple
    system_message = {
        "role": "system", 
        "content": "You are a friendly and kind assistant. You greet people and use simple, short words."
    }
    
    # Add the temporary "thinking" message
    chatbot.append({"role": "assistant", "content": "Thinking..."})
    yield AdditionalOutputs(chatbot)
    
    # Get response from LLM
    try:
        response_text = (
            sambanova_client.chat.completions.create(
                model="Meta-Llama-3.2-3B-Instruct",
                max_tokens=100,  
                messages=[system_message] + messages,
            )
            .choices[0]
            .message.content
        )
    except Exception as e:
        response_text = f"Error occurred."  
        print(f"LLM Error: {e}")
    
    # Remove the temporary "thinking" message
    chatbot.pop()
    
    print("\n----- LLM RESPONSE -----")
    print(response_text)
    print("------------------------\n")
    
    # Add the assistant's response as a new message
    chatbot.append({"role": "assistant", "content": response_text})
    # Update the UI to show the assistant's response
    yield AdditionalOutputs(chatbot)
    
    try:
        # Set the TTS speaking flag to true before starting TTS
        with tts_lock:
            is_tts_speaking = True
        print("TTS started speaking - STT disabled")
        
        # Track TTS start time
        tts_start_time = time.time()
        
        # Keep track of total audio duration
        total_audio_duration = 0
        
        # Use Kokoro for text-to-speech
        generator = tts_pipeline(response_text, voice='af_heart', speed=1)
        
        # Process the generated audio
        for _, _, audio_data in generator:
            # Kokoro outputs at 24000 Hz
            audio_array = np.asarray(audio_data, dtype=np.float32).reshape(1, -1)
            # Calculate duration of this audio chunk in seconds
            chunk_duration = len(audio_data) / 24000
            total_audio_duration += chunk_duration
            yield (24000, audio_array)
        
        # Calculate an optimized delay based on response length
        if total_audio_duration < 15:
            # Short response - wait full duration
            safe_delay = total_audio_duration
        elif total_audio_duration < 25:
            # Medium response - wait 80% of duration
            safe_delay = total_audio_duration * 0.8
        else:
            # Long response - wait 60% of duration + 1 second buffer
            safe_delay = (total_audio_duration * 0.6) + 1.0
            
        print(f"TTS audio duration: {total_audio_duration:.2f}s, waiting {safe_delay:.2f}s before re-enabling STT")
        
        # Wait for the calculated time
        time.sleep(safe_delay)
        
        # Total TTS processing time
        tts_elapsed = time.time() - tts_start_time
        print(f"Total TTS handling time: {tts_elapsed:.2f}s")
            
        # Set the TTS speaking flag to false after TTS is done
        with tts_lock:
            is_tts_speaking = False
        print("TTS finished speaking - STT enabled")
    except Exception as e:
        # Make sure to reset the flag even if there's an error
        with tts_lock:
            is_tts_speaking = False
        print(f"TTS Error: {e}")
        
    yield AdditionalOutputs(chatbot)


def on_stream_startup(stream_args):
    """Function called when the audio stream starts up"""
    print("\n\n***** AUDIO STREAM STARTED *****\n\n")
    print("Voice chat is ready! You can start speaking now.")
    print(f"Stream arguments: {stream_args}")
    
    # You can perform any initialization here that should happen when the stream starts
    # For example, reset the TTS speaking flag to ensure it starts in the correct state
    with tts_lock:
        global is_tts_speaking
        is_tts_speaking = False
    
    # Return a generator that yields the data
    # This is required by ReplyOnPause
    startup_data = {"stream_started_at": time.time()}
    
    # Create a generator that yields nothing but stores the data
    def startup_generator():
        # Store the startup data for later use
        on_stream_startup.data = startup_data
        # Yield nothing - this is just to make it a generator
        if False:
            yield
            
    # Return the generator
    return startup_generator()


def create_app():
    """
    Create and configure the Gradio UI and FastAPI app.
    """
    global stream_instance
    
    chatbot = gr.Chatbot(type="messages")
    stream = Stream(
        modality="audio",
        mode="send-receive",
        handler=ReplyOnPause(
            response,
            can_interrupt=False,
            algo_options=AlgoOptions(
                audio_chunk_duration=1.0,      
                started_talking_threshold=0.9, 
                speech_threshold=0.85,         
            ),
            startup_fn=on_stream_startup,
        ),
        additional_outputs_handler=lambda a, b: b,
        additional_inputs=[chatbot],
        additional_outputs=[chatbot],
        concurrency_limit=5,
        ui_args={"title": "Ai chat 🤖"},
    )
    
    # Store the stream instance globally
    stream_instance = stream
    
    # Mount the STREAM UI to the FastAPI app
    app = FastAPI()
    app = gr.mount_gradio_app(app, stream.ui, path="/")
    
    return app, stream


if __name__ == "__main__":
    os.environ["GRADIO_SSR_MODE"] = "false"
    
    # Initialize models upfront before launching the app
    initialize_models()
    
    app, stream = create_app()
    
    stream.ui.launch(server_port=7860)