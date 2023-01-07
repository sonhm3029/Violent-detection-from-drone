import pyaudio
import wave
import time
from pynput import keyboard


# you audio here
wf = wave.open('violence_sound_1.wav', 'rb')

# instantiate PyAudio
p = pyaudio.PyAudio()

# define callback
def callback(in_data, frame_count, time_info, status):
    data = wf.readframes(frame_count)
    return (data, pyaudio.paContinue)

# open stream using callback


for i in range(100):
    
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
                stream_callback=callback)
    stream.start_stream()
    time.sleep(0.18)
    stream.close()
    
    
# start the stream
# stream.start_stream()

# count = 0
# toggle_sound = 1
# while stream.is_active():
#     count +=1
#     print(count)
#     time.sleep(0.18)
#     # stream.stop_stream()
#     # stream.close()
#     # wf.close()
#     if toggle_sound == 1:
#         toggle_sound = 2
#         stream.stop_stream()
#         stream.close()
#         wf.close()
#     else:
#         toggle_sound = 1
#     wf = wave.open(f"violence_sound_{toggle_sound}.wav")
#     stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
#             channels=wf.getnchannels(),
#             rate=wf.getframerate(),
#             output=True,
#             stream_callback=callback)
    

# stop stream
# stream.stop_stream()
# stream.close()
# wf.close()

# # close PyAudio
# p.terminate()
