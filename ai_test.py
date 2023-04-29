import os, requests
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import pandas as pd
import pyaudio
import wave
import keyboard
import openai
from pydub import AudioSegment
from pydub.playback import play

# openAI key
os.environ["OPENAI_API_KEY"] = "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"
ELEVEN_LABS_API_KEY = "731539a9bd654037041f83186a358790"
openai.api_key = "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"
# read in to dataframe
df = pd.read_csv('WillBacon_2.csv')

# st.title("somethingsomething.ai")
# prompt = st.text_input("What would you like to know?")
#create our model
model = OpenAI(temperature = 0)

agent = create_pandas_dataframe_agent(model, df, verbose=True)
# prompt1 = "rows?"
# prompt2 = "Whats the cost for JOBNO 201470168W" 
# audio_bytes = audio_recorder()
# if audio_bytes:
#     st.audio(audio_bytes, format="audio/wav")

# set up the PyAudio object
# Note: these are values found online (not played around with yet)
chunk = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5  # adjust this to the desired length of the recording
# Need to make a config file for this stuff
WAVE_OUTPUT_FILENAME = "recording.wav"
ALT_ADVISOR_VOICE_ID = "1c8JE4vBzynMqcLpEH6u"
ADVISOR_VOICE_ID = "tu6IpO2JH3DKEu8ZuJ44"
ADVISOR_CUSTOM_PROMPT = "Answer in the style of a friendly assistant"
RACHEL_PROMPT = "21m00Tcm4TlvDq8ikWAM"

p = pyaudio.PyAudio()

# callback function to record audio when spacebar is pressed
def record_callback():
    frames = []
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=chunk)

    print("* recording")

    # recording device go brrrrrrrr
    while True:
        data = stream.read(chunk)
        frames.append(data)
        if keyboard.is_pressed(' '):
            break

    print("* done recording")

    # shut it down
    stream.stop_stream()
    stream.close()

    # output to wav
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

# register the spacebar as a hotkey to start and stop the recording
keyboard.add_hotkey('space', record_callback)
# print("made it here")
# start listening for hotkeys
print("Press the spacebar to start recording. Press it again to stop.")
keyboard.wait('esc')  # wait for the user to press the 'esc' key to stop the program

# open the WAV file
wf = wave.open("recording.wav", 'rb')
other_f = open("recording.wav", 'rb')
transcript = openai.Audio.transcribe("whisper-1",other_f)

# open the audio stream
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)

# play the audio in chunks
data = wf.readframes(chunk)
while data:
    stream.write(data)
    data = wf.readframes(chunk)

# close the audio stream and PyAudio object
stream.close()
p.terminate()
print(transcript["text"])
question = transcript["text"]
answer = agent.run(question)
print(answer)

url = f"https://api.elevenlabs.io/v1/text-to-speech/{ADVISOR_VOICE_ID}/stream"
data = {
    "text": answer,
    "voice_settings": {
        "stability": 0.1,
        "similarity_boost": 0.8
    }
}
r = requests.post(url, headers={'xi-api-key': ELEVEN_LABS_API_KEY}, json=data)
output_filename = "reply.mp3"
with open(output_filename, "wb") as output:
	output.write(r.content)
audio_file = AudioSegment.from_file("reply.mp3", format="mp3")
play(audio_file)


# play the recorded audio back to the user
# os.system("aplay " + WAVE_OUTPUT_FILENAME)
# os.system(f'aplay {WAVE_OUTPUT_FILENAME}')



