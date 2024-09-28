import os
import pygame
from pipes import quote
import re
import sqlite3
import struct
import subprocess
import time
import webbrowser
from playsound import playsound
import eel
import pyaudio
import pyautogui
from engine.command import speak
from engine.config import ASSISTANT_NAME
import pywhatkit as kit
import pvporcupine
import os
import re
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import speech_recognition as sr
import pyttsx3
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
from engine.helper import extract_yt_term, remove_words

# Import LangChain and Google Generative AI modules
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

# Load API keys
load_dotenv()

con = sqlite3.connect("jarvis.db")
cursor = con.cursor()

# Configure Google Generative AI
import google.generativeai as genai
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

@eel.expose
# Playing assistant sound function
def playAssistantSound():
    music_dir = r"D:\Coding\Request Jarvis\www\assets\audio\start_sound.mp3"
    
    if os.path.exists(music_dir):
        pygame.mixer.init()  # Initialize the mixer
        pygame.mixer.music.load(music_dir)  # Load the music file
        pygame.mixer.music.play()  # Play the music
        while pygame.mixer.music.get_busy():  # Wait until the music finishes
            pass
    else:
        print("File does not exist:", music_dir)


def hotword():
    porcupine = None
    paud = None
    audio_stream = None
    try:

        # pre-trained keywords    
        porcupine = pvporcupine.create(keywords=["jarvis", "alexa"])
        paud = pyaudio.PyAudio()
        audio_stream = paud.open(rate=porcupine.sample_rate, channels=1, format=pyaudio.paInt16,
                                 input=True, frames_per_buffer=porcupine.frame_length)

        # loop for streaming
        while True:
            keyword = audio_stream.read(porcupine.frame_length)
            keyword = struct.unpack_from("h" * porcupine.frame_length, keyword)

            # processing keyword comes from mic 
            keyword_index = porcupine.process(keyword)

            # checking if keyword is detected
            if keyword_index >= 0:
                print("Hotword detected")

                # pressing shortcut key win+j
                pyautogui.keyDown("win")
                pyautogui.press("j")
                time.sleep(2)
                pyautogui.keyUp("win")
    except:
        if porcupine is not None:
            porcupine.delete()
        if audio_stream is not None:
            audio_stream.close()
        if paud is not None:
            paud.terminate()


def get_pdf_text(pdf_file_path):
    """
    Extracts text from a PDF file given its file path.
    """
    text = ""
    pdf_reader = PdfReader(pdf_file_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    """
    Splits the extracted text into chunks for processing.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """
    Creates a FAISS vector store from text chunks and embeddings.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local('faiss_index')

def get_conversational_chain():
    """
    Loads a QA chain model with a custom prompt template for conversational context.
    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the 
    details. If the answer is not in the provided context, just say "answer is not related to request games" 
    and don't provide the wrong answer.\n\n
    Context:\n{context}?\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def process_user_input(user_question):
    """
    Handles user input, retrieves relevant documents, and generates a response using the conversational chain.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    new_db = FAISS.load_local('faiss_index', embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain()
    print(user_question)
    response = chain(
        {"input_documents": docs, "question": user_question}, 
        return_only_outputs=True
    )
    
    # Remove asterisks (*) from the response using regex
    cleaned_response = re.sub(r'\*', '', response['output_text'])
    
    return(cleaned_response)

def speak_text(text):
    """
    Converts text to speech using pyttsx3 with a female voice and increased speed.
    """
    engine = pyttsx3.init()
    
    # Set voice to female
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    
    # Increase the speech rate (default rate is around 200 words per minute)
    engine.setProperty('rate', 200)  # Increase speed
    
    engine.say(text)
    engine.runAndWait()

def get_speech_input():
    """
    Captures speech input from the user and returns it as text.
    """
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")
        audio = recognizer.listen(source)

    try:
        print("Recognizing...")
        user_input = recognizer.recognize_google(audio)
        print(f"User said: {user_input}")
        return user_input
    except sr.UnknownValueError:
        print("Sorry, I could not understand your speech.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return None

def preprocess():
    """
    Main function to execute the PDF processing and user interaction.
    """
    # Step 1: Load and process the PDF file
    pdf_file_path = 'Satoskar_Bhandarker_cology_ilovepdf_comp.pdf'
    
    if not os.path.exists(pdf_file_path):
        print(f"File not found: {pdf_file_path}")
        return
    
    print("Extracting text from the PDF...")
    raw_text = get_pdf_text(pdf_file_path)
    
    print("Splitting text into chunks...")
    text_chunks = get_text_chunks(raw_text)
    
    print("Creating vector store from text chunks...")
    get_vector_store(text_chunks)
    
    print("Processing complete. You can now ask questions about the event.")
        
def chatBot(query):
    print("In the chatbot function")
    """
    This function now uses the process_user_input function for handling user queries.
    """
    response = process_user_input(query)
    print(response)
    speak(response)
    return response


