import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty("voices")

for i, v in enumerate(voices):
    print(f"{i}: {v.name} ({v.id})")

engine.say("Hello Pvp, this is Jarvis reporting, what can I do for you today sir.")
engine.runAndWait()
