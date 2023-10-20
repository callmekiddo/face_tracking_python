import time
import threading
from playsound import playsound
from datetime import datetime

alertFilePath = "Alert.mp3"

def alert() : 
    if(datetime.now().hour > 5 ):
        if(datetime.now().hour < 7 and datetime.now().minute < 59):
            global isAlert
            playsound(alertFilePath)
            
            isAlert = False

while 1:
    alert()