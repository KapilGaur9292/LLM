import base64
import os
import time
import numpy as np
import cv2
import torch
import pyttsx3  # ✅ Text-to-Speech Library
import whisper  # ✅ OpenAI Whisper for STT
import mss  # ✅ Improved screenshot capture
from pyaudio import PyAudio, paInt16
from threading import Lock, Thread
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.llms import Ollama  # ✅ Corrected import
from speech_recognition import Microphone, Recognizer, UnknownValueError


# ✅ Load Whisper model
whisper_model = whisper.load_model("base")


class DesktopScreenshot:
    def __init__(self):
        self.screenshot = None
        self.running = False
        self.lock = Lock()

    def start(self):
        if self.running:
            return self
        self.running = True
        self.thread = Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        with mss.mss() as sct:
            while self.running:
                screenshot = sct.grab(sct.monitors[1])  # ✅ Capture Primary Monitor
                screenshot = np.array(screenshot)[:, :, :3]  # Remove alpha channel
                screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
                with self.lock:
                    self.screenshot = screenshot
                time.sleep(0.1)

    def read(self, encode=False):
        with self.lock:
            screenshot = self.screenshot.copy() if self.screenshot is not None else None
        if encode and screenshot is not None:
            _, buffer = cv2.imencode(".jpeg", screenshot)
            return base64.b64encode(buffer)
        return screenshot

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()


class Assistant:
    def __init__(self, model_name="llama3.1"):
        self.chain = self._create_inference_chain(model_name)
        self.tts_engine = pyttsx3.init()  # ✅ Initialize Text-to-Speech

    def answer(self, prompt, image):
        if not prompt:
            return
        print("Prompt:", prompt)

        response = self.chain.invoke(
            {"prompt": prompt, "image_base64": image.decode()},
            config={"configurable": {"session_id": "unused"}},
        ).strip()

        print("Response:", response)
        self._speak(response)  # ✅ Speak Response

    def _speak(self, text):
        """Converts text to speech and plays it."""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def _create_inference_chain(self, model_name):
        SYSTEM_PROMPT = """
        You are a helpful assistant that can see the user's desktop and respond to their queries.
        Keep your answers concise and informative.
        """

        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", [{"type": "text", "text": "{prompt}"}]),
        ])

        model = Ollama(model=model_name)  # ✅ Using Ollama
        chain = prompt_template | model | StrOutputParser()
        chat_message_history = ChatMessageHistory()

        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )


# ✅ Start desktop screenshot capture
desktop_screenshot = DesktopScreenshot().start()
assistant = Assistant(model_name="llama3.1")  # ✅ Use LLaMA 3.1 Model


def audio_callback(recognizer, audio):
    """Processes live speech and converts it to text."""
    try:
        prompt = recognizer.recognize_whisper(audio, model="base", language="english")
        assistant.answer(prompt, desktop_screenshot.read(encode=True))
    except UnknownValueError:
        print("There was an error processing the audio.")


# ✅ Set up speech recognition
recognizer = Recognizer()
microphone = Microphone()
with microphone as source:
    recognizer.adjust_for_ambient_noise(source)

# ✅ Start background listening
stop_listening = recognizer.listen_in_background(microphone, audio_callback)

# ✅ Display the desktop screen
while True:
    screenshot = desktop_screenshot.read()
    if screenshot is not None:
        cv2.imshow("Desktop", screenshot)
    if cv2.waitKey(1) in [27, ord("q")]:  # Press 'ESC' or 'q' to exit
        break

# ✅ Cleanup
desktop_screenshot.stop()
cv2.destroyAllWindows()
stop_listening(wait_for_stop=False)
