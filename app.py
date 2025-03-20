import os
import re
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

"""
Voice Chat Application
---------------------
This application provides a voice interface for chatting with a language model.
It uses:
- SambaNova's LLM API for chat responses
- OpenAI's Whisper for speech-to-text
- Kokoro for text-to-speech
- WebRTC for real-time audio streaming

The application handles audio input, transcribes it, sends it to the LLM,
and plays back the LLM's response through text-to-speech.
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
    
    This function loads environment variables, initializes the SambaNova client
    for LLM interactions, OpenAI client for Whisper STT, and Kokoro for TTS.
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
    from kokoro import KPipeline
    tts_pipeline = KPipeline(lang_code='a')
    print("Kokoro TTS model loaded")
    
    print("All models initialized")


def whisper_stt(audio_data: NDArray, sample_rate: int = 16000) -> str:
    """
    Use OpenAI's Whisper model for speech-to-text transcription.
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate of the audio (default: 16000 Hz)
        
    Returns:
        Transcribed text as a string, or empty string if transcription fails
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


def is_valid_speech(text: str) -> bool:
    """
    Validate if the transcribed text is likely to be actual speech rather than noise.
    
    This function applies several heuristics to determine if the transcribed
    text represents valid speech or just background noise.
    
    Args:
        text: The transcribed text to validate
        
    Returns:
        True if the text appears to be valid speech, False otherwise
    """
    if not text or len(text.strip()) < 4:
        return False
    
    # Check if text contains at least one word with 4+ characters
    # This helps filter out random noises that might be transcribed as short sounds
    has_real_word = any(len(word) >= 4 for word in text.split())
    if not has_real_word:
        return False
    
    # Check for high ratio of non-alphanumeric characters (often noise)
    total_chars = len(text)
    alpha_chars = sum(c.isalnum() or c.isspace() for c in text)
    if total_chars > 0 and alpha_chars / total_chars < 0.8:
        return False
    
    # Remove common filler words/sounds that might be detected from background noise
    filler_pattern = r'\b(um|uh|hmm|ah|eh|oh|mm|hm|er|mhm|huh)\b'
    cleaned_text = re.sub(filler_pattern, '', text, flags=re.IGNORECASE)
    
    # If after removing fillers we have very little left, it's probably not valid speech
    if len(cleaned_text.strip()) < 4:
        return False
    
    # Check if the text has a reasonable word count (not just one or two words)
    word_count = len(cleaned_text.split())
    if word_count < 2:
        return False
    
    return True


def response(
    audio: tuple[int, NDArray[np.int16 | np.float32]],
    chatbot: list[dict] | None = None,
):
    """
    Process audio input, transcribe it, get LLM response, and generate TTS output.
    
    This is the main handler function for the voice chat application.
    
    Args:
        audio: Tuple of (sample_rate, audio_data)
        chatbot: List of chat messages for context
        
    Yields:
        Audio data for TTS playback and updated chatbot state
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
    
    # Apply bandpass filter to focus on speech frequencies (300Hz to 3400Hz)
    # This removes low-frequency background noise and high-frequency artifacts
    from scipy import signal
    
    # Design bandpass filter - focuses on human speech frequencies
    # Higher low cutoff (300Hz) helps eliminate distant mumbly voices and room noise
    low_cutoff = 1000  # Hz - increased to filter out low rumbles and distant voices
    high_cutoff = 3400  # Hz - standard telephone upper limit for speech
    
    # Normalize frequencies to Nyquist frequency
    nyquist = sample_rate / 2
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    
    # Create and apply bandpass filter
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_audio = signal.filtfilt(b, a, audio_data)
    
    # Apply exponential moving average for smoothing
    alpha = 0.05  # Smoothing factor
    smoothed_audio = np.zeros_like(filtered_audio)
    smoothed_audio[0] = filtered_audio[0]
    for i in range(1, len(filtered_audio)):
        smoothed_audio[i] = alpha * filtered_audio[i] + (1 - alpha) * smoothed_audio[i-1]
    
    # Apply enhanced noise gate with distance-based rejection
    audio_abs = np.abs(smoothed_audio)
    audio_mean = np.mean(audio_abs)
    audio_std = np.std(audio_abs)
    audio_peak = np.max(audio_abs)
    audio_energy = np.mean(audio_abs**2)  # Energy metric helps identify foreground vs background
    
    print(f"Audio metrics: mean={audio_mean:.6f}, std={audio_std:.6f}, peak={audio_peak:.6f}, energy={audio_energy:.6f}")
    
    # Enhanced threshold that considers both level and consistency
    # This helps distinguish between nearby speech and distant voices
    # Increased thresholds to reject more background/distant sounds
    if (audio_mean < 0.04 or  # Increased minimum level threshold
        (audio_energy < 0.003) or  # Energy threshold for distance detection
        (audio_std < 0.06 and audio_peak < 0.25) or  # Increased variance threshold
        (audio_peak / audio_mean < 4)):  # Dynamic range check - nearby voices have higher peaks
        print("Audio likely contains distant voices or background noise - ignoring")
        return
    
    # Calculate signal-to-noise ratio (simple approximation)
    # Divide the audio into segments and compute variance
    segment_size = int(sample_rate * 0.05)  # 50ms segments
    if len(smoothed_audio) > segment_size * 3:  # Ensure we have enough samples
        segments = np.array_split(smoothed_audio, len(smoothed_audio) // segment_size)
        segment_vars = [np.var(seg) for seg in segments]
        
        # Segments with higher variance likely contain speech
        speech_segments = sum(1 for var in segment_vars if var > np.mean(segment_vars) * 1.5)
        total_segments = len(segment_vars)
        
        # If less than 30% of segments contain speech, it's likely background noise or distant talking
        if speech_segments / total_segments < 0.3:
            print(f"Low speech density ({speech_segments}/{total_segments} segments) - likely distant voices")
            return
    
    # Use Whisper for transcription with filtered audio
    start = time.time()
    text = whisper_stt(smoothed_audio, sample_rate)
    print(f"Transcription time: {time.time() - start:.2f}s")
    
    # Apply stricter validation to the transcribed text
    if not is_valid_speech(text):
        print("Skipping invalid speech input:", text)
        return
        
    print("\n----- USER INPUT -----")
    print(text)
    print("----------------------\n")
    
    # Create a new message for this user input
    chatbot.append({"role": "user", "content": text})
    # Update the UI immediately to show the new message
    yield AdditionalOutputs(chatbot)
    
    # Rest of the function remains unchanged
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
        # For short responses: use full duration
        # For medium responses: use 80% of duration
        # For long responses: use 60% of duration + a small fixed buffer
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


# Create the UI
def create_app():
    """
    Create and configure the Gradio UI and FastAPI app.
    
    Returns:
        FastAPI app configured with the Gradio UI
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
                # Increase chunk duration for better speech detection
                audio_chunk_duration=0.8,
                # Increase these thresholds to make speech detection more strict
                started_talking_threshold=0.8,  # Higher threshold to start detecting speech
                speech_threshold=0.8,           # Higher threshold to consider audio as speech
            ),
            input_sample_rate=16000
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
    import time  
    os.environ["GRADIO_SSR_MODE"] = "false"
    
    # Initialize models upfront before launching the app
    initialize_models()
    
    app, stream = create_app()
    
    stream.ui.launch(server_port=7860)