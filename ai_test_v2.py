import os, requests
from langchain.agents import create_pandas_dataframe_agent, load_tools, Tool
from langchain.llms import OpenAI, PromptLayerOpenAI
from langchain.chains import SimpleSequentialChain, LLMChain, ConversationChain
from langchain.prompts import PromptTemplate
from langchain.memory import SimpleMemory
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
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
    # os.environ['TWILIO_ACOUNT_SID'] = config['TWILIO']['TWILIO_ACOUNT_SID']
    # os.environ['TWILIO_AUTH_TOKEN'] = config['TWILIO']['TWILIO_AUTH_TOKEN']

    # print(OPENAI_API_KEY)


# agent_prompt = "You are working with a pandas dataframe in Python. The name of the dataframe is `df`.\nYou should use the tools below to answer the question posed of you:\n\npython_repl_ast: A Python shell. Use this to execute python commands. Input should be a valid python command. When using this tool, sometimes output is abbreviated - make sure it does not look abbreviated before using it in your answer.\n\nHuman: You can ask a human for guidance when you think you got stuck or you are not sure what to do next. The input should be a question for the human.\n\nPlease note:\n\nIf the question requires you to modify the dataframe, please ensure that the format of any modification matches the rest of the dataframe. If you are asked to add a new row, please ensure that you have been provided all the column values to fill that row. As long as you are missing values, do not create the new row and continue to ask the user for the values you are missing. Also, please make sure to check your memory for context\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [python_repl_ast]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\n\nThis is the result of `print(df.head())`:\n{df}\n\nBegin!\n{chat_history}\nQuestion: {input}\n{agent_scratchpad}"
agent_prompt = "You are working with a pandas dataframe in Python. The name of the dataframe is `df`.\nYou should use the tools below to answer the question posed of you:\n\npython_repl_ast: A Python shell. Use this to execute python commands. Input should be a valid python command. When using this tool, sometimes output is abbreviated - make sure it does not look abbreviated before using it in your answer.\n\ntalk_to_me: You can ask a human for guidance when you think you got stuck or you are not sure what to do next. Simply output a question for the Human.\n\nPlease note:\n\nIf the question requires you to modify the dataframe, please ensure that the format of any modification matches the rest of the dataframe. If you are asked to add a new row, please ensure that you have been provided all the column values to fill that row. As long as you are missing values, do not create the new row and continue to ask the user for the values you are missing. Also, please make sure to check your memory for context\n\n Also note that to get the length of dataframe `df` you must use len(df.index) rather than len(df)\n\nUse the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction: the action to take, should be one of [python_repl_ast]\nAction Input: the input to the action\nObservation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\n\nThis is the result of `print(df.head())`:\n{df}\n\nBegin!\n{chat_history}\nQuestion: {input}\n{agent_scratchpad}"


# print("=========")
getConfigs('config.ini')

# set up the PyAudio object
# Note: these are values found online (not played around with yet)
chunk = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 3  # adjust this to the desired length of the recording

with open('agent_prompt.txt', 'r') as file:
    agent_prompt = file.read()
# print(ADVISOR_CUSTOM_PROMPT)
ADVISOR_CUSTOM_PROMPT = "You are a personal assistant, given information please provide it in a simple way\n Information: {output}\n reply: "
# RACHEL_PROMPT = "21m00Tcm4TlvDq8ikWAM"
p = pyaudio.PyAudio()



def voice(input):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ADVISOR_VOICE_ID}/stream"
    data = {
        "text": input,
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
    while True:
        if keyboard.is_pressed('space'):
                record_callback()
                response = getQstring()
                return response

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
  

    model = OpenAI(temperature = 0, openai_api_key=openai.api_key)
    # model = ChatOpenAI(model_name = "gpt-3.5-turbo-0301", temperature = 0.1, openai_api_key=openai.api_key)


    agent = create_pandas_dataframe_agent(model, df, verbose=True)
    # print(agent.agent.llm_chain.prompt.input_variables)
    # print(agent.agent.llm_chain.output_keys)
    agent.agent.llm_chain.prompt.template = agent_prompt
    # print("========")

    # "gpt-3.5-turbo-0301"
    Talk_to_me = Tool(
    name="talk_to_me",
    func=voice,
    description="You can ask a human for guidance when you think you got stuck or you are not sure what to do next. Simply output a question for the Human.")
    # I talk back, lets talk money, I talk that
    
    agent.tools.append(Talk_to_me)
    
    # print("========")
    agent.agent.llm_chain.prompt.input_variables = ['input', 'agent_scratchpad', 'chat_history']

    # print(agent.agent.llm_chain.prompt.input_variables)
    agent.memory = ConversationBufferWindowMemory(memory_key="chat_history")
    # agent
    # print(dir(agent))
    # my_t = agent.memory
    # prompt_template = PromptTemplate(input_variables=["output"], template=ADVISOR_CUSTOM_PROMPT)
    # agent.agent.llm_chain.prompt.template = prompt_template
    # chain_two = LLMChain(llm=model, prompt=prompt_template, memory = ConversationBufferWindowMemory(k=5)) 
    # conv_chain = ConversationChain(llm=model)
#     conversation = ConversationChain(
#     llm=llm, prompt = PromptTemplate,
#     memory=ConversationBufferWindowMemory(k=5)
# )
    # print(agent.agent.llm_chain.output_keys)
    # overall_chain = SimpleSequentialChain(
                # chains=[agent, conv_chain],
                # verbose=True)
    # print(my_t)

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
            # print("before")
            answer = agent.run(question)
            # print("after")
            print(len(agent.memory.buffer))
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
            # print(df)
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



