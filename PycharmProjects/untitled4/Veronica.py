import time
import speech_recognition as sr
from fuzzywuzzy import fuzz
import pyttsx3
import datetime
import pyowm
import webbrowser
import urllib2
import urllib
import json


word = {
    "name": ('вероника','женщина','ника','мадам','вер','мисс'),
    "tbr": ('скажи','расскажи','покажи','произнеси','сколько','открой','активируй'),
    "cmds": {
        "ctime":('который час','сколько вермени','текущее веремя','сейчас вермени',),
        "radio": ('включи музыку','музыка','включи радио','радио'),
        "stup": ('анекдот','рассмеши меня'),
        "wet": ('погода','прогноз погоды'),
        "command": ('новости', 'страницу','google'),
        "command1": ('вк', 'vk','страницу вк')
    }

    }
def speak(what):
    print(what)
    speak_engine.say(what)
    speak_engine.runAndWait()
    speak_engine.stop()


def callback(recognizer, audio):
    try:
        voice = recognizer.recognize_google(audio, language="ru-RU").lower()
        print("[Вероника] Распознано: " + voice)

        if voice.startswith(word["name"]):
            cmd = voice
            for x in word['name']:
                cmd = cmd.replace(x, "").strip()
            for x in word['tbr']:
                cmd = cmd.replace(x, "").strip()
            cmd = recognize_cmd(cmd)
            execute_cmd(cmd['cmd'])

    except sr.UnknownValueError:
        print("[Вероника] Голос не распознан!")
    except sr.RequestError as e:
        print("[Вероника] Неизвестная ошибка, проверьте интернет!")


def recognize_cmd(cmd):
    RC = {'cmd': '', 'percent': 0}
    for c, v in word['cmds'].items():

        for x in v:
            vrt = fuzz.ratio(cmd, x)
            if vrt > RC['percent']:
                RC['cmd'] = c
                RC['percent'] = vrt

    return RC

def execute_cmd(cmd):
    if cmd == "ctime":
        # сказать текущее время
        now = datetime.datetime.now()
        speak("Сейчас " + str(now.hour) + ":" + str(now.minute))


    elif cmd == 'stup':

        speak("— Почему Дед Мороз всегда счастлив? — Потому что он знает, где живут плохие девочки. ")




    elif cmd == 'wet':
        owm = pyowm.OWM('fb5f8127f82131b3a24784cb37da0d57', language='ru-RU')
        city =('Мстиславль')
        obser = owm.weather_at_place(city)
        w = obser.get_weather()
        tempet = w.get_temperature('celsius')['temp']

        speak("Погода на сегодня: "+ str(round(tempet))+ " градусов по Цельсию " + " на улице сейчас " + str(w.get_status()))

    elif cmd == 'command':
        voice1 = recognizer.recognize_google(audio, language="ru-RU").lower()
        print("[Вероника] Распознано: " + voice1)
        webbrowser.open("http://www.google.com", new = 2)

        speak("Секундочку ")

    elif cmd == 'command1':
        webbrowser.open("https://vk.com/magic_han", new = 2)

        speak("Секундочку ")

    else:
        print('Команда не распознана, повторите!')

r = sr.Recognizer()
m = sr.Microphone(device_index=1)

with m as source:
    r.adjust_for_ambient_noise(source)

speak_engine = pyttsx3.init()

voices = speak_engine.getProperty('voices')
speak_engine.setProperty('voice', voices[0].id)

speak("С возвращением Сэр")
speak("Ве роника ожидает ")

stop_list = r.listen_in_background(m,callback)
while True: time.sleep(0.1)

