from playsound import playsound
import time
import threading
import multiprocessing as mp


alarm = False
tempThread = None

def sound():
    print("Play")
    global alarm
    playsound("violence_sound_1.wav")
    alarm = False

for i in range(10):
    time.sleep(0.18)
    print(i)
    if not alarm:
        alarm = True
        threading.Thread(target=sound).start()

    