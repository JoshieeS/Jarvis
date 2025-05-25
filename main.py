# main.py

import sys
import random
import cv2
import speech_recognition as sr
import pyttsx3

from model import ask_openai   # your existing API wrapper
from memory import MemoryManager

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QPushButton, QLineEdit, QTextEdit, QLabel,
    QVBoxLayout, QHBoxLayout
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt, QRect
from PyQt5.QtGui import QPainter, QColor, QRadialGradient, QBrush, QPixmap

# ─── Instantiate memory ─────────────────────────────────────────────────────
memory = MemoryManager()

# ─── Face‐Tracking Thread ─────────────────────────────────────────────────────
class FaceTrackerThread(QThread):
    faceMoved = pyqtSignal(float, float)
    def __init__(self, parent=None):
        super().__init__(parent)
        cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade)
        self.running  = True
    def run(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if not ret: break
            small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
            gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(gray, 1.2, 4)
            if len(faces) > 0:
                x,y,w,h = max(faces, key=lambda f: f[2]*f[3])
                cx, cy  = x + w/2, y + h/2
                h_s, w_s = gray.shape
                self.faceMoved.emit(cx/w_s, cy/h_s)
            self.msleep(5)
        cap.release()
    def stop(self):
        self.running = False
        self.wait()

# ─── Continuous Speech Recognition Thread ─────────────────────────────────────
class ContinuousSRThread(QThread):
    listening = pyqtSignal(bool)
    result    = pyqtSignal(str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.recognizer = sr.Recognizer()
        self.running    = True
    def run(self):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            while self.running:
                self.listening.emit(True)
                audio = self.recognizer.listen(source)
                self.listening.emit(False)
                try:
                    txt = self.recognizer.recognize_google(audio)
                except (sr.UnknownValueError, sr.RequestError):
                    txt = ""
                if txt:
                    self.result.emit(txt)
    def stop(self):
        self.running = False
        self.wait()

# ─── GPT Worker Thread (injects memory into system prompt) ────────────────────
class GPTWorker(QThread):
    responseReady = pyqtSignal(str)
    def __init__(self, prompt, parent=None):
        super().__init__(parent)
        self.prompt = prompt
    def run(self):
        # 1) update memory from this input
        memory.update_from_input(self.prompt)
        # 2) build system prompt with memory summary
        mem_sum = memory.summary()
        system_content = "You are Jarvis, a helpful assistant."
        if mem_sum:
            system_content += " User info: " + mem_sum + "."
        # 3) call OpenAI
        try:
            reply = ask_openai(
                prompt=self.prompt,
                system_prompt=system_content
            )
        except Exception as e:
            reply = f"[Error: {e}]"
        self.responseReady.emit(reply)

# ─── Face Widget with Blinking & Pupil Tracking ──────────────────────────────
class FaceWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.blink = False
        self.px, self.py = 0.5, 0.5
        self._init_blink()

    def attach_tracker(self, tracker):
        tracker.faceMoved.connect(self._on_face_moved)

    def _init_blink(self):
        self.blink_timer = QTimer(self)
        self.blink_timer.timeout.connect(self._do_blink)
        self._schedule_blink()

    def _schedule_blink(self):
        self.blink_timer.start(random.randint(2000, 6000))

    def _do_blink(self):
        self.blink = True
        self.update()
        QTimer.singleShot(random.randint(100, 250), self._end_blink)

    def _end_blink(self):
        self.blink = False
        self.update()
        self._schedule_blink()

    def _on_face_moved(self, xr, yr):
        α = 0.8
        self.px = α * xr + (1 - α) * self.px
        self.py = α * yr + (1 - α) * self.py
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(20, 20, 20))

        # Widget‐level eye size
        eye_w, eye_h = 100, 60
        y = self.height() // 3
        max_off_x = eye_w * 0.25
        max_off_y = eye_h * 0.25

        # Offscreen “2K” resolution multiplier
        OFFSCREEN_SCALE = 20
        off_w = eye_w * OFFSCREEN_SCALE    # ~2000 px
        off_h = eye_h * OFFSCREEN_SCALE    # ~1200 px

        # Number of blocks in the pixel grid
        GRID_W = 64
        GRID_H = int(GRID_W * off_h / off_w)

        for cx_base in (self.width()//3, 2*self.width()//3):
            x = cx_base - eye_w//2

            if self.blink:
                p.setBrush(QColor(20, 20, 20))
                p.drawRoundedRect(x, y + eye_h//2 - 5, eye_w, 10, 5, 5)
                continue

            # 1) Draw sclera
            p.setBrush(QColor(245, 245, 245))
            p.drawEllipse(x, y, eye_w, eye_h)

            # 2) Create high‐res iris
            iris_pix = QPixmap(off_w, off_h)
            iris_pix.fill(Qt.transparent)
            tmp = QPainter(iris_pix)
            tmp.setRenderHint(QPainter.Antialiasing)

            # gradient center moves with face tracking
            center_x = off_w/2 + (0.5 - self.px) * max_off_x * OFFSCREEN_SCALE * 2
            center_y = off_h/2 + (self.py - 0.5) * max_off_y * OFFSCREEN_SCALE * 2
            grad = QRadialGradient(center_x, center_y, off_h * 0.4)
            grad.setColorAt(0.0, QColor(30, 160, 200))
            grad.setColorAt(0.7, QColor(10, 80, 120))
            grad.setColorAt(1.0, QColor(0, 0, 0))
            tmp.setBrush(QBrush(grad))
            tmp.drawEllipse(0, 0, off_w, off_h)
            tmp.end()

            # 3) Pixelate: downscale then upscale
            small = iris_pix.scaled(GRID_W, GRID_H,
                                    Qt.IgnoreAspectRatio,
                                    Qt.FastTransformation)
            pixelated = small.scaled(off_w, off_h,
                                     Qt.IgnoreAspectRatio,
                                     Qt.FastTransformation)

            # 4) Blit pixelated iris into widget eye area
            eye_rect = QRect(x, y, eye_w, eye_h)
            p.drawPixmap(eye_rect, pixelated, 
                         QRect(0, 0, off_w, off_h))

            # 5) Draw blocky pupil
            pupil_w = eye_w * 0.2
            pupil_h = eye_h * 0.2
            px = x + eye_w/2 - pupil_w/2 \
                 + (0.5-self.px)*max_off_x*2
            py = y + eye_h/2 - pupil_h/2 \
                 + (self.py-0.5)*max_off_y*2
            p.setBrush(QColor(0, 0, 0))
            p.drawRect(int(px), int(py), int(pupil_w), int(pupil_h))

            # 6) Block‐style highlight
            hl_w = int(pupil_w * 0.5)
            hl_h = int(pupil_h * 0.5)
            p.setBrush(QColor(255, 255, 255, 180))
            p.drawRect(int(px + pupil_w*0.1),
                       int(py + pupil_h*0.1),
                       hl_w, hl_h)

# ─── Main Application Window ─────────────────────────────────────────────────
class JarvisUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Jarvis"); self.setFixedSize(400,550)
        self._build_ui()
        # face tracking
        self.tracker = FaceTrackerThread(); self.face.attach_tracker(self.tracker); self.tracker.start()
        # speech recog.
        self.sr = ContinuousSRThread()
        self.sr.listening.connect(self._on_listen)
        self.sr.result.connect(self._on_speech)
        self.sr.start()

    def _build_ui(self):
        w=QWidget(); v=QVBoxLayout(w)
        self.face = FaceWidget(); v.addWidget(self.face, stretch=2)
        self.status = QLabel("Idle"); v.addWidget(self.status)
        # input row
        h=QHBoxLayout()
        self.input = QLineEdit(); self.input.setPlaceholderText("Say or type something…")
        self.input.returnPressed.connect(self._on_send)
        btn = QPushButton("Send"); btn.clicked.connect(self._on_send)
        h.addWidget(self.input, stretch=3); h.addWidget(btn, stretch=1)
        v.addLayout(h)
        # log & controls
        self.log = QTextEdit(); self.log.setReadOnly(True); v.addWidget(self.log, stretch=1)
        ctrl=QHBoxLayout()
        r=QPushButton("Reset"); r.clicked.connect(self._on_reset)
        o=QPushButton("Off");   o.clicked.connect(self.close)
        ctrl.addWidget(r); ctrl.addWidget(o); v.addLayout(ctrl)
        self.setCentralWidget(w)

    def _on_listen(self, listening):
        self.status.setText("Listening…" if listening else "Processing…")

    def _on_speech(self, txt):
        self.log.append(f"<you> {txt}")
        self.input.setText(txt)
        self._on_send()

    def _on_send(self):
        txt = self.input.text().strip()
        if not txt: return
        self.log.append(f"<you> {txt}")
        self.input.clear()
        self.status.setText("Thinking…")
        self.worker = GPTWorker(txt)
        self.worker.responseReady.connect(self._on_gpt)
        self.worker.start()

    def _on_gpt(self, reply):
        self.log.append(f"<Jarvis> {reply}")
        self.status.setText("Idle")
        try:
            engine = pyttsx3.init()
            engine.say(reply)
            engine.runAndWait()
        except:
            pass

    def _on_reset(self):
        self.log.clear()
        self.log.append("<System> Conversation reset.")

    def closeEvent(self, ev):
        self.tracker.stop()
        self.sr.stop()
        super().closeEvent(ev)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = JarvisUI()
    win.show()
    sys.exit(app.exec_())
