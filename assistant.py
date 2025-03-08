import sys
import os
import tempfile
import logging
import uuid
import html
import re
import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
import queue
import time
import webrtcvad  # For voice activity detection

# PyQt5 imports for the GUI and threading
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QUrl, QTimer
from PyQt5.QtGui import QFont, QGuiApplication, QSyntaxHighlighter, QTextCharFormat, QColor, QTextCursor
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QFileDialog,
    QLabel,
    QInputDialog,
)
from PyQt5.QtMultimedia import QSoundEffect  # For playing sound effects

# Third-party spell checking library
try:
    import enchant
except ImportError:
    print("Please install pyenchant (pip install pyenchant)")
    enchant = None

spell_dict = enchant.Dict("en_US") if enchant is not None else None

# ASR & TTS libraries and LangChain imports
from faster_whisper import WhisperModel
from kokoro import KPipeline
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama

# -------------------------------
# Global Configuration and Models
# -------------------------------
SAMPLE_RATE = 16000         # Whisper expects 16 kHz for ASR
CHANNELS = 1                # Mono recording
TTS_SAMPLING_RATE = 24000   # Kokoro outputs audio at 24 kHz
BUTTON_WIDTH = 70           # Fixed button size

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

logging.info("Loading Whisper ASR model on CUDA...")
try:
    asr_model = WhisperModel("small", device="cuda", compute_type="float16")
except Exception as e:
    logging.error(f"Error loading Whisper model: {e}")
    sys.exit(1)

try:
    tts_pipeline = KPipeline(lang_code='a')
except Exception as e:
    logging.error(f"Error initializing TTS pipeline: {e}")
    sys.exit(1)

template = """
You are a helpful AI assistant.

Conversation history:
{history}

User: {input}
Assistant:"""
prompt = PromptTemplate(input_variables=["history", "input"], template=template)
memory = ConversationBufferMemory(ai_prefix="Assistant:")
chain = ConversationChain(
    prompt=prompt,
    memory=memory,
    llm=Ollama(model="deepseek-r1:32b"),
    verbose=False,
)

# -------------------------------
# Helper Functions
# -------------------------------

def filter_thought_section(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def is_bullet_list(text):
    lines = text.splitlines()
    for line in lines:
        if re.match(r"^(?:\-|\*+|•|\d+\.)\s*", line.strip()):
            return True
    return False

def remove_code_sections(text):
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)

def save_audio_to_file(audio_np, sample_rate):
    file_path = tempfile.mktemp(suffix=".wav")
    sf.write(file_path, audio_np, sample_rate)
    return file_path

def transcribe_audio(audio_file):
    segments, _ = asr_model.transcribe(audio_file, beam_size=5)
    return " ".join([segment.text.strip() for segment in segments])

def convert_plain_bullet_list_to_html(text):
    lines = text.splitlines()
    list_items = []
    for line in lines:
        match = re.match(r"^\s*([-*•]|\d+\.)\s+(.*)", line)
        if match:
            item = match.group(2).strip()
            list_items.append(f"<li>{item}</li>")
        else:
            list_items.append(f"<p>{line.strip()}</p>")
    return "<ul>" + "".join(list_items) + "</ul>"

def render_code_blocks(text):
    def replacer(match):
        code_content = match.group(1)
        lines = code_content.splitlines()
        language = ""
        if lines and lines[0].strip().lower() in ["python", "html", "javascript"]:
            language = lines[0].strip().lower()
            code_content = "\n".join(lines[1:])
        escaped_code = html.escape(code_content)
        code_id = str(uuid.uuid4())
        CODE_BLOCKS[code_id] = (language, code_content)
        return (
            f'<div class="code-container">'
            f'<a href="copy:{code_id}" class="copy-link top">Copy</a> '
            f'<a href="save:{code_id}" class="save-link top">Save</a>'
            f'<pre><code>{escaped_code}</code></pre>'
            f'<a href="copy:{code_id}" class="copy-link bottom">Copy</a> '
            f'<a href="save:{code_id}" class="save-link bottom">Save</a>'
            f'</div>'
        )
    return re.sub(r"```(.*?)```", replacer, text, flags=re.DOTALL)

def format_message_text(text):
    processed = render_code_blocks(text)
    return processed if "<" in processed and ">" in processed else f"<p>{processed.replace('\n','<br>')}</p>"

def sanitize_tts_text(text):
    text = text.replace("—", " — ")
    text = re.sub(r'(\d+)\s*–\s*(\d+)', r'\1 to \2', text)
    text = text.replace("–", "-")
    text = re.sub(r'[\*\_]', '', text)
    return text

def check_speech(audio_np, sample_rate=16000, frame_duration_ms=20, threshold=0.1):
    int16_data = (audio_np * 32768).astype(np.int16)
    raw_bytes = int16_data.tobytes()
    frame_size = int(sample_rate * (frame_duration_ms / 1000.0))
    num_bytes_per_frame = frame_size * 2
    vad = webrtcvad.Vad(3)
    num_frames = len(raw_bytes) // num_bytes_per_frame
    if num_frames == 0:
        return False
    speech_frames = 0
    for i in range(num_frames):
        start = i * num_bytes_per_frame
        frame = raw_bytes[start:start+num_bytes_per_frame]
        if len(frame) == num_bytes_per_frame and vad.is_speech(frame, sample_rate):
            speech_frames += 1
    return (speech_frames / num_frames) >= threshold

# -------------------------------
# Global dictionary for code blocks
# -------------------------------
CODE_BLOCKS = {}

# -------------------------------
# Spell Checking Highlighter
# -------------------------------
if spell_dict:
    class SpellCheckHighlighter(QSyntaxHighlighter):
        def __init__(self, parent, dictionary):
            super().__init__(parent)
            self.dictionary = dictionary
            self.error_format = QTextCharFormat()
            self.error_format.setUnderlineColor(QColor("red"))
            self.error_format.setUnderlineStyle(QTextCharFormat.SpellCheckUnderline)
        def highlightBlock(self, text):
            for match in re.finditer(r'\b\w+\b', text):
                word = match.group()
                if not self.dictionary.check(word):
                    start = match.start()
                    length = match.end() - match.start()
                    self.setFormat(start, length, self.error_format)
else:
    SpellCheckHighlighter = None

# -------------------------------
# PocketSphinx Wake Word Listener (with always_enabled parameter)
# -------------------------------
from pocketsphinx import LiveSpeech

class PocketSphinxWakeWordListener(QThread):
    wakeWordDetected = pyqtSignal()
    def __init__(self, keyphrase="computer", threshold=1e-20, always_enabled=False, parent=None):
        super().__init__(parent)
        self.keyphrase = keyphrase.lower()
        self.threshold = threshold
        self.always_enabled = always_enabled
        self.running = True
        self.enabled = True
    def setEnabled(self, value):
        self.enabled = value
    def run(self):
        speech = LiveSpeech(
            verbose=False,
            sampling_rate=16000,
            buffer_size=2048,
            no_search=False,
            full_utt=False,
            keyphrase=self.keyphrase,
            kws_threshold=self.threshold
        )
        for phrase in speech:
            if not self.running:
                break
            if not self.always_enabled and not self.enabled:
                time.sleep(0.1)
                continue
            recognized = str(phrase).lower().strip()
            # Require exact match to reduce false triggers.
            if recognized == self.keyphrase:
                self.wakeWordDetected.emit()
                if not self.always_enabled and self.parent():
                    self.parent().playWakeSound()
                time.sleep(2)
    def stop(self):
        self.running = False

# -------------------------------
# Custom QTextBrowser subclass
# -------------------------------
class ChatBrowser(QTextBrowser):
    def setSource(self, url):
        scheme = url.scheme()
        if scheme in ("copy", "save"):
            return
        super().setSource(url)

# -------------------------------
# HTML Wrapper Function with Menu Bar CSS
# -------------------------------
def wrap_html(content):
    return f"""
    <html>
    <head>
      <style>
        QMenuBar {{ background: black; color: white; }}
        QMenuBar::item {{ background: black; color: white; padding: 4px 10px; }}
        QMenuBar::item:selected {{ background: yellow; color: black; }}
        QMenu {{ background: black; color: white; }}
        QMenu::item {{ background: black; color: white; padding: 4px 20px; }}
        QMenu::item:selected {{ background: yellow; color: black; }}
        body {{
          background-color: black;
          color: white;
          margin: 0;
          padding: 5px;
          font-family: sans-serif;
          line-height: 1.6;
        }}
        p {{ margin: 10px 0; }}
        ul {{ list-style-type: disc; margin-left: 20px; padding-left: 20px; }}
        ol {{ list-style-type: decimal; margin-left: 20px; padding-left: 20px; }}
        li {{ margin: 5px 0; }}
        pre {{
          background-color: black;
          color: white;
          margin: 0;
          white-space: pre-wrap;
          font-family: Consolas, "Courier New", monospace;
          padding: 5px;
        }}
        .code-container {{
          position: relative;
          background-color: #333;
          padding: 10px;
          margin: 10px 0;
          border-radius: 5px;
          border: 1px solid #555;
        }}
        .copy-link.top {{
          position: absolute;
          top: 5px;
          right: 5px;
          background-color: #555;
          color: #fff;
          padding: 2px 5px;
          text-decoration: none;
          font-size: 10pt;
          border-radius: 3px;
        }}
        .copy-link.bottom {{
          position: absolute;
          bottom: 5px;
          right: 5px;
          background-color: #555;
          color: #fff;
          padding: 2px 5px;
          text-decoration: none;
          font-size: 10pt;
          border-radius: 3px;
        }}
        .save-link.top {{
          position: absolute;
          top: 5px;
          right: 70px;
          background-color: #777;
          color: #fff;
          padding: 2px 5px;
          text-decoration: none;
          font-size: 10pt;
          border-radius: 3px;
        }}
        .save-link.bottom {{
          position: absolute;
          bottom: 5px;
          right: 70px;
          background-color: #777;
          color: #fff;
          padding: 2px 5px;
          text-decoration: none;
          font-size: 10pt;
          border-radius: 3px;
        }}
      </style>
    </head>
    <body>{content}</body>
    </html>
    """

# -------------------------------
# TTS Streaming Worker (Updated to accept voice parameter)
# -------------------------------
class TTSStreamingWorker(QThread):
    finishedTTS = pyqtSignal()
    def __init__(self, text: str, voice: str, parent=None):
        super().__init__(parent)
        self.text = sanitize_tts_text(remove_code_sections(text))
        self.voice = voice
        self.audio_queue = queue.Queue()
        self.current_chunk = np.array([], dtype=np.float32)
        self.done_generating = False
        self.stop_requested = False
        self.paused = False
        self.stream = None
    def pause_tts(self):
        self.paused = True
    def resume_tts(self):
        self.paused = False
    def stop_tts(self):
        self.stop_requested = True
        self.paused = True
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
    def run(self):
        logging.info("TTS generation started.")
        def generate_audio():
            try:
                generator = tts_pipeline(self.text, voice=self.voice, speed=1.25, split_pattern=r'\n+')
                for _, _, audio in generator:
                    audio_np = np.array(audio, dtype=np.float32)
                    self.audio_queue.put(audio_np)
            except Exception as e:
                logging.error(f"Error during TTS generation: {e}")
            self.done_generating = True
            logging.info("TTS generation finished.")
        generation_thread = threading.Thread(target=generate_audio)
        generation_thread.start()
        try:
            self.stream = sd.OutputStream(samplerate=TTS_SAMPLING_RATE, channels=CHANNELS, callback=self.callback)
            self.stream.start()
            while ((not self.done_generating) or (not self.audio_queue.empty()) or (self.current_chunk.size > 0)) and (not self.stop_requested):
                self.msleep(50)
            self.stream.stop()
            self.stream.close()
        except Exception as e:
            logging.error(f"Error in audio output stream: {e}")
        generation_thread.join()
        self.finishedTTS.emit()
    def callback(self, outdata, frames, time_info, status):
        if status:
            logging.warning(f"Output stream status: {status}")
        if self.paused:
            outdata[:] = np.zeros((frames, CHANNELS), dtype=np.float32)
            return
        samples = []
        while len(samples) < frames:
            if self.current_chunk.size == 0:
                try:
                    self.current_chunk = self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            needed = frames - len(samples)
            if self.current_chunk.size <= needed:
                samples.extend(self.current_chunk.tolist())
                self.current_chunk = np.array([], dtype=np.float32)
            else:
                samples.extend(self.current_chunk[:needed].tolist())
                self.current_chunk = self.current_chunk[needed:]
        if len(samples) < frames:
            samples.extend([0.0] * (frames - len(samples)))
        outdata[:] = np.array(samples, dtype=np.float32).reshape(frames, CHANNELS)

# -------------------------------
# Recorder Thread (Unchanged)
# -------------------------------
class RecorderThread(QThread):
    finishedRecording = pyqtSignal(np.ndarray)
    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop_event = threading.Event()
        self.audio_queue = queue.Queue()
        self.last_voice_time = time.time()
        self.silence_threshold = 0.01
    def run(self):
        logging.info("Recording started.")
        self.last_voice_time = time.time()
        silence_duration = 3.0
        try:
            with sd.RawInputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16", callback=self.callback):
                while not self._stop_event.is_set():
                    self.msleep(100)
                    if time.time() - self.last_voice_time > silence_duration:
                        logging.info("Silence detected for 3 seconds, stopping recording.")
                        self._stop_event.set()
        except Exception as e:
            logging.error(f"Error in audio input stream: {e}")
        audio_data = b"".join(list(self.audio_queue.queue))
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        logging.info("Recording finished.")
        self.finishedRecording.emit(audio_np)
    def callback(self, indata, frames, time_info, status):
        if status:
            logging.warning(f"Recording status: {status}")
        audio_frame = np.frombuffer(indata, dtype=np.int16)
        audio_frame_float = audio_frame.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(audio_frame_float**2))
        if rms > self.silence_threshold:
            self.last_voice_time = time.time()
        self.audio_queue.put(bytes(indata))
    def stop(self):
        self._stop_event.set()

# -------------------------------
# Custom Text Input Widget
# -------------------------------
class MessageInput(QTextEdit):
    sendMessage = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPlaceholderText("Type a message...")
        self.setStyleSheet("background-color: black; color: white;")
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setFixedHeight(100)
        if SpellCheckHighlighter and spell_dict:
            self.highlighter = SpellCheckHighlighter(self.document(), spell_dict)
        else:
            self.highlighter = None
    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not (event.modifiers() & Qt.ShiftModifier):
            if self.toPlainText().strip():
                self.sendMessage.emit()
            event.accept()
        else:
            super().keyPressEvent(event)
    def contextMenuEvent(self, event):
        menu = self.createStandardContextMenu()
        cursor = self.cursorForPosition(event.pos())
        cursor.select(QTextCursor.WordUnderCursor)
        word = cursor.selectedText()
        if word and spell_dict and not spell_dict.check(word):
            suggestions = spell_dict.suggest(word)
            if suggestions:
                suggestion_menu = menu.addMenu("Spelling Suggestions")
                start = cursor.selectionStart()
                end = cursor.selectionEnd()
                for suggestion in suggestions[:5]:
                    action = suggestion_menu.addAction(suggestion)
                    def replace_word(checked, suggestion=suggestion, start=start, end=end):
                        text_cursor = self.textCursor()
                        text_cursor.setPosition(start)
                        text_cursor.setPosition(end, QTextCursor.KeepAnchor)
                        text_cursor.insertText(suggestion)
                    action.triggered.connect(replace_word)
        menu.exec_(event.globalPos())

# -------------------------------
# LLM Query Worker
# -------------------------------
class LLMQueryWorker(QThread):
    resultReady = pyqtSignal(str)
    def __init__(self, query, parent=None):
        super().__init__(parent)
        self.query = query
    def run(self):
        try:
            result = chain.predict(input=self.query)
        except Exception as e:
            result = "Error during LLM query: " + str(e)
        self.resultReady.emit(result)

# -------------------------------
# Main GUI Window
# -------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hef-Bot")
        self.resize(700, 500)
        self.setStyleSheet("background-color: #222222;")
        self.chat_messages = []
        self.current_tts_message_index = None
        self.recording = False
        self.cancelled_recording = False
        self.recorder = None
        self.tts_thread = None
        self.auto_recording_enabled = True
        self.current_voice = "af_heart"
        self.conversationActive = False
        self.interrupted = False
        self.chatText = ChatBrowser()
        self.chatText.setReadOnly(True)
        self.chatText.setStyleSheet("background-color: black; color: white; padding: 5px;")
        self.chatText.setOpenExternalLinks(False)
        self.chatText.anchorClicked.connect(self.handleAnchorClicked)
        self.textInput = MessageInput()
        self.textInput.sendMessage.connect(self.sendText)
        self.textInput.textChanged.connect(self.onTextChanged)
        self.statusLabel = QLabel("Idle")
        self.statusLabel.setAlignment(Qt.AlignCenter)
        self.statusLabel.setStyleSheet("color: yellow; font-size: 18pt;")
        
        # Initialize start and end conversation sound effects.
        self.wakeSound = QSoundEffect()
        start_sound_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "start_conversation.wav")
        self.wakeSound.setSource(QUrl.fromLocalFile(start_sound_path))
        self.wakeSound.setVolume(0.5)
        
        self.endSound = QSoundEffect()
        end_sound_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "end_conversation.wav")
        self.endSound.setSource(QUrl.fromLocalFile(end_sound_path))
        self.endSound.setVolume(0.5)
        
        # Menu Bar and Menus with Hover Styles
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar { background: black; color: white; }
            QMenuBar::item { background: black; color: white; padding: 4px 10px; }
            QMenuBar::item:selected { background: yellow; color: black; }
            QMenu { background: black; color: white; }
            QMenu::item { background: black; color: white; padding: 4px 20px; }
            QMenu::item:selected { background: yellow; color: black; }
        """)
        file_menu = menubar.addMenu("File")
        edit_menu = menubar.addMenu("Edit")
        
        save_chat_action = file_menu.addAction("Save Chat")
        save_chat_action.setShortcut("Ctrl+S")
        save_chat_action.triggered.connect(self.saveChat)
        
        clear_chat_action = file_menu.addAction("Clear Chat")
        clear_chat_action.setShortcut("Ctrl+L")
        clear_chat_action.triggered.connect(self.clearChat)
        
        exit_action = file_menu.addAction("Exit")
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        
        increase_font_action = edit_menu.addAction("Increase Font Size")
        increase_font_action.setShortcut("Ctrl+=")
        increase_font_action.triggered.connect(self.increaseFont)
        
        decrease_font_action = edit_menu.addAction("Decrease Font Size")
        decrease_font_action.setShortcut("Ctrl+-")
        decrease_font_action.triggered.connect(self.decreaseFont)
        
        change_wake_action = edit_menu.addAction("Change Wake Word...")
        change_wake_action.triggered.connect(self.changeWakeWord)
        
        # Primary action buttons
        self.playPauseButton = QPushButton("Pause")
        self.playPauseButton.setFixedSize(BUTTON_WIDTH, BUTTON_WIDTH)
        self.playPauseButton.setStyleSheet("background-color: black; border: 3px solid orange; color: white; font-weight: bold; font-size: 10pt; padding: 5px;")
        self.playPauseButton.clicked.connect(self.toggleTtsPause)
        
        self.stopButton = QPushButton("Cancel")
        self.stopButton.setFixedSize(BUTTON_WIDTH, BUTTON_WIDTH)
        self.stopButton.setStyleSheet("background-color: black; border: 3px solid red; color: white; font-weight: bold; font-size: 10pt; padding: 5px;")
        self.stopButton.clicked.connect(self.stopEverything)
        
        self.recordButton = QPushButton("Voice\nChat")
        self.recordButton.setFixedSize(BUTTON_WIDTH, BUTTON_WIDTH)
        self.recordButton.setStyleSheet("background-color: black; border: 3px solid green; color: white; font-weight: bold; font-size: 10pt; padding: 5px;")
        self.recordButton.clicked.connect(self.startRecording)
        
        self.sendButton = QPushButton("Send\nText")
        self.sendButton.setFixedSize(BUTTON_WIDTH, BUTTON_WIDTH)
        self.sendButton.setStyleSheet("background-color: black; border: 3px solid green; color: white; font-weight: bold; font-size: 10pt; padding: 5px;")
        self.sendButton.clicked.connect(self.sendText)
        self.sendButton.setEnabled(False)
        
        rightLayout = QVBoxLayout()
        rightLayout.addWidget(self.recordButton)
        rightLayout.addWidget(self.stopButton)
        rightLayout.addWidget(self.playPauseButton)
        rightLayout.addWidget(self.sendButton)
        rightLayout.addStretch()
        self.rightWidget = QWidget()
        self.rightWidget.setLayout(rightLayout)
        self.rightWidget.setFixedWidth(BUTTON_WIDTH + 20)
        
        leftLayout = QVBoxLayout()
        leftLayout.addWidget(self.chatText, stretch=1)
        leftLayout.addWidget(self.textInput)
        leftLayout.addWidget(self.statusLabel)
        self.leftWidget = QWidget()
        self.leftWidget.setLayout(leftLayout)
        
        mainLayout = QHBoxLayout()
        mainLayout.setContentsMargins(10, 10, 10, 10)
        mainLayout.setSpacing(10)
        mainLayout.addWidget(self.leftWidget, stretch=1)
        mainLayout.addWidget(self.rightWidget, stretch=0)
        container = QWidget()
        container.setLayout(mainLayout)
        self.setCentralWidget(container)
        
        self.currentFontSize = 12
        self.updateChatAndInputFontSizes()
        
        # Instantiate wake word listener for "computer" (only active when idle)
        self.wakeWordListener = PocketSphinxWakeWordListener(keyphrase="computer", parent=self, always_enabled=False)
        self.wakeWordListener.wakeWordDetected.connect(self.onWakeWordDetected)
        self.wakeWordListener.start()
        
        # Instantiate cancel word listener for "cancel that" (always active)
        self.cancelListener = PocketSphinxWakeWordListener(keyphrase="cancel that", parent=self, always_enabled=True)
        self.cancelListener.wakeWordDetected.connect(self.onCancelWordDetected)
        self.cancelListener.start()
        
        # Instantiate interrupt word listener for "hold on" (always active) with a higher threshold to reduce false triggers.
        self.interruptListener = PocketSphinxWakeWordListener(keyphrase="wait a minute", threshold=1e-15, parent=self, always_enabled=True)
        self.interruptListener.wakeWordDetected.connect(self.onInterruptWordDetected)
        self.interruptListener.start()
    
    def playWakeSound(self):
        self.wakeSound.play()
    
    def playEndSound(self):
        self.endSound.play()
    
    def onCancelWordDetected(self):
        if self.recording:
            self.cancelled_recording = True
        self.stopEverything()
        self.statusLabel.setText("Cancelled by voice.")
    
    def onInterruptWordDetected(self):
        if self.tts_thread and self.tts_thread.isRunning():
            self.tts_thread.stop_tts()
            self.tts_thread.wait()
            self.tts_thread = None
            self.interrupted = True
            self.playWakeSound()  # Play start conversation sound on interrupt.
            self.statusLabel.setText("Interrupted TTS. Recording...")
            self.startRecording()
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space and not self.textInput.hasFocus():
            self.toggleTtsPause()
        else:
            super().keyPressEvent(event)
    
    def onTextChanged(self):
        if self.textInput.toPlainText().strip():
            self.sendButton.setEnabled(True)
        else:
            self.sendButton.setEnabled(False)
    
    def onWakeWordDetected(self):
        if not self.recording and (self.tts_thread is None or not self.tts_thread.isRunning()):
            self.statusLabel.setText("Wake word detected! Starting recording...")
            self.disableWakeWordTemporarily()
            self.startRecording()
    
    def disableWakeWordTemporarily(self):
        self.wakeWordListener.setEnabled(False)
        QTimer.singleShot(5000, lambda: self.wakeWordListener.setEnabled(True))
    
    def clearChat(self):
        self.chat_messages = []
        chain.memory.clear()
        self.update_chat_display()
    
    def toggleTtsPause(self):
        if self.tts_thread and self.tts_thread.isRunning():
            if not self.tts_thread.paused:
                self.tts_thread.pause_tts()
                self.playPauseButton.setText("Resume")
            else:
                self.tts_thread.resume_tts()
                self.playPauseButton.setText("Pause")
    
    def handleAnchorClicked(self, url):
        scheme = url.scheme()
        code_id = url.toString()[len(scheme)+1:]
        if scheme == "copy":
            code_info = CODE_BLOCKS.get(code_id)
            if code_info:
                _, code_text = code_info
                clipboard = QGuiApplication.clipboard()
                clipboard.setText(code_text)
                self.statusLabel.setText("Code copied!")
                QTimer.singleShot(500, lambda: self.statusLabel.setText("Idle"))
        elif scheme == "save":
            code_info = CODE_BLOCKS.get(code_id)
            if code_info:
                language, code_text = code_info
                filters = "Text Files (*.txt);;All Files (*)"
                if language == "python":
                    filters = "Python Files (*.py);;Text Files (*.txt);;All Files (*)"
                elif language == "html":
                    filters = "HTML Files (*.html);;Text Files (*.txt);;All Files (*)"
                filename, _ = QFileDialog.getSaveFileName(self, "Save Code", "", filters)
                if filename:
                    try:
                        with open(filename, "w", encoding="utf-8") as f:
                            f.write(code_text)
                        self.statusLabel.setText("Code saved!")
                    except Exception as e:
                        self.statusLabel.setText(f"Error saving code: {e}")
                    QTimer.singleShot(1000, lambda: self.statusLabel.setText("Idle"))
        return
    
    def updateChatAndInputFontSizes(self):
        font = QFont()
        font.setPointSize(self.currentFontSize)
        self.chatText.setFont(font)
        self.textInput.setFont(font)
    
    def increaseFont(self):
        self.currentFontSize += 2
        self.updateChatAndInputFontSizes()
    
    def decreaseFont(self):
        if self.currentFontSize > 6:
            self.currentFontSize -= 2
            self.updateChatAndInputFontSizes()
    
    def update_chat_display(self):
        html_content = ""
        for msg in self.chat_messages:
            if msg["sender"] == "User":
                html_content += f'<div style="color: #00FFFF; margin: 10px 0;"><b>User:</b> {format_message_text(msg["text"])}</div>'
            elif msg["sender"] == "Assistant":
                color = "#FFFF00" if msg.get("highlight", False) else "white"
                html_content += f'<div style="color: {color}; margin: 10px 0;"><b>Assistant:</b> {format_message_text(msg["text"])}</div>'
        full_html = wrap_html(html_content)
        self.chatText.setHtml(full_html)
        self.chatText.moveCursor(self.chatText.textCursor().End)
    
    def append_message(self, sender, text, highlight=False):
        self.chat_messages.append({"sender": sender, "text": text, "highlight": highlight})
        self.update_chat_display()
    
    def startRecording(self):
        if self.recording:
            return
        if self.tts_thread and self.tts_thread.isRunning():
            self.tts_thread.stop_tts()
            self.tts_thread.wait()
            self.tts_thread = None
        self.recording = True
        self.auto_recording_enabled = True
        self.conversationActive = True
        self.statusLabel.setText("Recording...")
        self.recordButton.setEnabled(False)
        self.wakeWordListener.setEnabled(False)
        self.recordButton.setStyleSheet("background-color: green; border: 3px solid green; color: white; font-weight: bold; font-size: 10pt; padding: 5px;")
        self.recorder = RecorderThread()
        self.recorder.finishedRecording.connect(self.processRecording)
        self.recorder.start()
    
    def stopRecording(self):
        if self.recording and self.recorder:
            self.cancelled_recording = True
            self.recording = False
            self.recorder.stop()
            self.statusLabel.setText("Recording cancelled.")
            self.recordButton.setEnabled(True)
    
    def processRecording(self, audio_np):
        self.recording = False
        self.recordButton.setEnabled(True)
        if self.cancelled_recording:
            self.cancelled_recording = False
            self.statusLabel.setText("Idle")
            QTimer.singleShot(1000, self.checkIdleState)
            return
        if not check_speech(audio_np, SAMPLE_RATE, frame_duration_ms=20, threshold=0.1):
            self.append_message("User", "[No speech detected]")
            self.statusLabel.setText("Idle")
            QTimer.singleShot(1000, self.checkIdleState)
            return
        self.statusLabel.setText("Transcribing...")
        audio_file = save_audio_to_file(audio_np, SAMPLE_RATE)
        transcript = ""
        try:
            transcript = transcribe_audio(audio_file)
        except Exception as e:
            transcript = "[Error during transcription]"
        finally:
            try:
                os.remove(audio_file)
            except Exception as e:
                logging.warning(f"Could not remove temporary file {audio_file}: {e}")
        if transcript.strip() == "":
            self.append_message("User", "[No speech detected]")
            self.statusLabel.setText("Idle")
            QTimer.singleShot(1000, self.checkIdleState)
            return
        self.append_message("User", transcript)
        self.queryLLM(transcript)
    
    def sendText(self):
        if self.tts_thread and self.tts_thread.isRunning():
            self.tts_thread.stop_tts()
            self.tts_thread.wait()
            self.tts_thread = None
        plain_text = self.textInput.toPlainText().strip()
        if not plain_text:
            return
        if is_bullet_list(plain_text):
            text = convert_plain_bullet_list_to_html(plain_text)
        else:
            if plain_text.count("\n") >= 2 and not plain_text.startswith("```"):
                text = "```" + "\n" + plain_text + "\n```"
            else:
                text = plain_text
            text = f"<p>{text.replace('\n','<br>')}</p>"
        self.auto_recording_enabled = False
        self.append_message("User", text)
        self.textInput.clear()
        QApplication.processEvents()
        self.queryLLM(text)
    
    def queryLLM(self, user_input):
        self.statusLabel.setText("Querying LLM...")
        self.queryWorker = LLMQueryWorker(user_input)
        self.queryWorker.resultReady.connect(self.handleLLMResponse)
        self.queryWorker.start()
    
    def handleLLMResponse(self, response):
        filtered_response = filter_thought_section(response.strip())
        self.append_message("Assistant", filtered_response, highlight=True)
        tts_text = filtered_response
        self.current_tts_message_index = len(self.chat_messages) - 1
        self.statusLabel.setText("Speaking...")
        self.recordButton.setEnabled(False)
        self.wakeWordListener.setEnabled(False)
        self.tts_thread = TTSStreamingWorker(tts_text, self.current_voice)
        self.tts_thread.finishedTTS.connect(self.on_tts_finished)
        self.tts_thread.start()
    
    def on_tts_finished(self):
        if self.interrupted:
            self.interrupted = False
            return
        self.statusLabel.setText("Idle")
        self.recordButton.setEnabled(True)
        if self.current_tts_message_index is not None:
            self.chat_messages[self.current_tts_message_index]["highlight"] = False
            self.update_chat_display()
        self.current_tts_message_index = None
        QTimer.singleShot(5000, lambda: self.wakeWordListener.setEnabled(True))
        if self.auto_recording_enabled:
            QTimer.singleShot(500, self.startRecording)
        QTimer.singleShot(1000, self.checkIdleState)
    
    def checkIdleState(self):
        if not self.recording and (self.tts_thread is None or not self.tts_thread.isRunning()):
            self.recordButton.setStyleSheet("background-color: black; border: 3px solid green; color: white; font-weight: bold; font-size: 10pt; padding: 5px;")
            self.wakeWordListener.setEnabled(True)
            if self.conversationActive:
                self.playEndSound()
                self.conversationActive = False
    
    def stopEverything(self):
        if self.tts_thread and self.tts_thread.isRunning():
            self.tts_thread.stop_tts()
            self.tts_thread.wait()
            self.tts_thread = None
            self.playPauseButton.setText("Pause")
            self.statusLabel.setText("TTS stopped.")
            self.current_tts_message_index = None
            self.recordButton.setEnabled(True)
            self.auto_recording_enabled = False
        if self.recording:
            self.stopRecording()
            self.auto_recording_enabled = False
        self.statusLabel.setText("Idle")
        self.checkIdleState()
    
    def saveChat(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Chat", "", "Text Files (*.txt);;All Files (*)", options=options)
        if fileName:
            try:
                with open(fileName, 'w', encoding='utf-8') as f:
                    for msg in self.chat_messages:
                        f.write(f'{msg["sender"]}: {msg["text"]}\n')
                self.statusLabel.setText(f"Chat saved to {fileName}")
            except Exception as e:
                self.statusLabel.setText(f"Error saving chat: {e}")
    
    def changeWakeWord(self):
        dialog = QInputDialog(self)
        dialog.setWindowTitle("Change Wake Word")
        current = f'Currently: "{self.wakeWordListener.keyphrase}"'
        dialog.setLabelText("Enter new wake word:\n" + current)
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        dialog.setStyleSheet("""
            QLabel { color: white; font-size: 12pt; }
            QLineEdit { color: black; background: white; font-size: 12pt; }
            QDialogButtonBox QPushButton { color: black; background: yellow; font-size: 12pt; }
        """)
        if dialog.exec_():
            new_word = dialog.textValue().strip().lower()
            if new_word:
                if self.wakeWordListener:
                    self.wakeWordListener.stop()
                    self.wakeWordListener.wait()
                self.wakeWordListener = PocketSphinxWakeWordListener(keyphrase=new_word, parent=self, always_enabled=False)
                self.wakeWordListener.wakeWordDetected.connect(self.onWakeWordDetected)
                self.wakeWordListener.start()
                self.statusLabel.setText(f"Wake word changed to '{new_word}'")
    
    def closeEvent(self, event):
        self.stopEverything()
        for listener in [self.wakeWordListener, self.cancelListener, self.interruptListener]:
            if listener is not None and listener.isRunning():
                listener.quit()
                if not listener.wait(1000):
                    listener.terminate()
                    listener.wait(1000)
        event.accept()
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space and not self.textInput.hasFocus():
            self.toggleTtsPause()
        else:
            super().keyPressEvent(event)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
