import pyttsx3
import speech_recognition as sr
import eel
import time
import traceback  # Import traceback for detailed error output

def speak(text):
    text = str(text)
    engine = pyttsx3.init('sapi5')
    voices = engine.getProperty('voices') 
    engine.setProperty('voice', voices[1].id)
    engine.setProperty('rate', 200)
    eel.DisplayMessage(text)
    engine.say(text)
    eel.receiverText(text)
    engine.runAndWait()

def takecommand():

    r = sr.Recognizer()

    with sr.Microphone() as source:
        print('listening....')
        eel.DisplayMessage('listening....')
        r.pause_threshold = 1
        r.adjust_for_ambient_noise(source)
        
        audio = r.listen(source, 10, 6)

    try:
        print('recognizing')
        eel.DisplayMessage('recognizing....')
        query = r.recognize_google(audio, language='en-in')
        print(f"user said: {query}")
        eel.DisplayMessage(query)
        time.sleep(2)
       
    except Exception as e:
        print(f"Error in speech recognition: {e}")
        eel.DisplayMessage(f"Error in speech recognition: {e}")
        return ""
    
    return query.lower()

@eel.expose
def allCommands(message=1):
    if message == 1:
        query = takecommand()
        print(query)
        eel.senderText(query)
    else:
        query = message
        eel.senderText(query)

    try:
            print(query)
            from engine.features import chatBot
            chatBot(query)
    except Exception as e:
        error_message = traceback.format_exc()  # Get detailed error traceback
        print(f"Error occurred: {e}\n{error_message}")
        eel.DisplayMessage(f"Error occurred: {e}\n{error_message}")
    
    eel.ShowHood()
