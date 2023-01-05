import pyaudio
import wave
import time
from pynput import keyboard

paused = False    # global to track if the audio is paused
def on_press(key):
    global paused
    print (key)
    if key == keyboard.Key.space:
        if stream.is_stopped():     # time to play audio
            print ('play pressed')
            stream.start_stream()
            paused = False
            return False
        elif stream.is_active():   # time to pause audio
            print ('pause pressed')
            stream.stop_stream()
            paused = True
            return False
    return False

toggle_sound = 1
# you audio here
wf = wave.open('violence_sound_1.wav', 'rb')

# instantiate PyAudio
p = pyaudio.PyAudio()

# define callback
def callback(in_data, frame_count, time_info, status):
    data = wf.readframes(frame_count)
    return (data, pyaudio.paContinue)

# open stream using callback
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
                stream_callback=callback)

# start the stream
stream.start_stream()

count = 0

while stream.is_active() or paused==True:
    count +=1
    print(count)
    time.sleep(1)
    if(count % 5) == 0:
        stream.stop_stream()
        stream.close()
        wf.close()
        if toggle_sound == 1:
            toggle_sound = 2
        else:
            toggle_sound = 1
        wf = wave.open(f"violence_sound_{toggle_sound}.wav")
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
                stream_callback=callback)
    

# stop stream
# stream.stop_stream()
# stream.close()
# wf.close()

# # close PyAudio
# p.terminate()
