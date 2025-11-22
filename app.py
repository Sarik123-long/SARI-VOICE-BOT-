import streamlit as st
from openai import OpenAI
from gtts import gTTS
from io import BytesIO
import tempfile
import subprocess
import os
import imageio_ffmpeg as ffmpeg

# Streamlit page settings
st.set_page_config(page_title="AI VoiceBot", page_icon="🎤")
st.title("🎤 AI VoiceBot Demo")
st.write("Upload a voice message and get an AI response with audio.")

# --- Use environment variable or hardcode key for local testing ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# OPENAI_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx"  # Uncomment for local testing only

if not OPENAI_API_KEY:
    st.error("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# Upload voice input
voice_file = st.file_uploader(
    "Upload your voice file (mp3/wav/opus)", 
    type=["mp3", "wav", "opus"]
)

if voice_file:
    try:
        # Save uploaded file temporarily
        file_ext = voice_file.name.split('.')[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp:
            tmp.write(voice_file.read())
            tmp_path = tmp.name

        # Convert .opus to .wav using imageio-ffmpeg
        if file_ext == "opus":
            wav_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            ffmpeg_path = ffmpeg.get_ffmpeg_exe()
            subprocess.run(
                [ffmpeg_path, "-y", "-i", tmp_path, wav_temp.name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            tmp_path = wav_temp.name

        # Play uploaded/converted audio
        st.audio(tmp_path)

        # Transcribe audio using OpenAI Whisper
        with open(tmp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        user_text = transcription.text
        st.text_area("You said:", value=user_text, height=100)

        # Generate AI response using GPT-4
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": user_text}],
            max_tokens=200
        )
        bot_text = response.choices[0].message.content
        st.text_area("AI Response:", value=bot_text, height=100)

        # Convert AI response to audio
        tts = gTTS(bot_text)
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        st.audio(audio_bytes.getvalue(), format="audio/mp3")

    except Exception as e:
        st.error(f"An error occurred: {e}")
