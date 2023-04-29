import os, requests
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI, PromptLayerOpenAI
import promptlayer
import pandas as pd
import pyaudio
import wave
import keyboard
import openai
from pydub import AudioSegment
from pydub.playback import play
import configparser


# global ELEVEN_LABS_API_KEY
# global cvs_file
# global ADVISOR_VOICE_ID
# openai.api_key


def getConfigs(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    global ELEVEN_LABS_API_KEY
    ELEVEN_LABS_API_KEY = config['API']['ELEVEN_LABS_API_KEY']
    global OPENAI_API_KEY
    OPENAI_API_KEY = config['API']['OPENAI_API_KEY']
    openai.api_key = config['API']['OPENAI_API_KEY']
    global cvs_file
    cvs_file = config['DATABASE']['CSV_FILE']
    global ADVISOR_VOICE_ID
    ADVISOR_VOICE_ID = config['ADVISOR_VOICE']['ADVISOR_VOICE_ID']
    global WAV_OUTPUT_FILENAME
    WAV_OUTPUT_FILENAME = config['WAV']['OUTPUT_FILENAME']
    global PLAYBACK_COND
    PLAYBACK_COND = config.getboolean('CONDITIONS', 'PLAYBACK_COND')
    global PROMPT_LAYER_API
    PROMPT_LAYER_API = config['API']['PROMPT_LAYER_API']
    promptlayer.api_key = PROMPT_LAYER_API
    print(OPENAI_API_KEY)




# print("=========")
getConfigs('config.ini')

# set up the PyAudio object
# Note: these are values found online (not played around with yet)
chunk = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 3  # adjust this to the desired length of the recording
# Need to make a config file for this stuff
# WAVE_OUTPUT_FILENAME = "recording.wav"
# ALT_ADVISOR_VOICE_ID = "1c8JE4vBzynMqcLpEH6u"
# ADVISOR_VOICE_ID = "tu6IpO2JH3DKEu8ZuJ44"
ADVISOR_CUSTOM_PROMPT = "Answer in the style of a friendly assistant"
# RACHEL_PROMPT = "21m00Tcm4TlvDq8ikWAM"
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
    wf = wave.open(WAV_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()



# what art thou question?
def getQstring():
    rec = open(WAV_OUTPUT_FILENAME, 'rb')
    transcript = openai.Audio.transcribe("whisper-1",rec)
    return transcript["text"]

# main function, everything else is 
def runner():
    df = pd.read_csv(cvs_file)
    openai=promptlayer.openai
    openai.api_key = OPENAI_API_KEY
    # openai.api_key = os.environ.get("OPENAI_API_KEY")
    openai.Completion.create(
                    engine="text-ada-001", 
                    prompt=ADVISOR_CUSTOM_PROMPT
                    
)

    # model = OpenAI(temperature = 0.7, openai_api_key=openai.api_key)

    agent = create_pandas_dataframe_agent(openai, df, verbose=True)

    print("Press the spacebar to ask a question. Press it again to stop.")
    print("When you're all done, print the escape key to exit")
    while not keyboard.is_pressed('esc'):
        if keyboard.is_pressed('space'):
            record_callback()
            question = getQstring()
            print(question)
            if PLAYBACK_COND:
                wf = wave.open("recording.wav", 'rb')
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
            print(df)
    p.terminate()

        

runner()
# open the WAV file
# wf = wave.open("recording.wav", 'rb')
# other_f = open("recording.wav", 'rb')
# transcript = openai.Audio.transcribe("whisper-1",other_f)
# print(transcript)

# # open the audio stream
# stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
#                 channels=wf.getnchannels(),
#                 rate=wf.getframerate(),
#                 output=True)

# # play the audio in chunks
# data = wf.readframes(chunk)
# while data:
#     stream.write(data)
#     data = wf.readframes(chunk)

# # close the audio stream and PyAudio object
# stream.close()
# p.terminate()
# print(transcript["text"])
# question = transcript["text"]
# answer = agent.run(question)
# print(answer)

# url = f"https://api.elevenlabs.io/v1/text-to-speech/{ADVISOR_VOICE_ID}/stream"
# data = {
#     "text": answer,
#     "voice_settings": {
#         "stability": 0.1,
#         "similarity_boost": 0.8
#     }
# }
# r = requests.post(url, headers={'xi-api-key': ELEVEN_LABS_API_KEY}, json=data)
# output_filename = "reply.mp3"
# with open(output_filename, "wb") as output:
# 	output.write(r.content)
# audio_file = AudioSegment.from_file("reply.mp3", format="mp3")
# play(audio_file)


# play the recorded audio back to the user
# os.system("aplay " + WAVE_OUTPUT_FILENAME)
# os.system(f'aplay {WAVE_OUTPUT_FILENAME}')



