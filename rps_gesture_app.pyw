from __future__ import annotations

import math
import threading
import time
import tkinter as tk
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk

REQUIRED_IMPORT_ERROR: ImportError | None = None
try:
    import cv2
    import numpy as np
    from PIL import Image, ImageTk
except ImportError as exc:
    REQUIRED_IMPORT_ERROR = exc
    cv2 = None
    np = None
    Image = None
    ImageTk = None

from tas_kagit_makas_bot import AdaptiveRPSBot, DISPLAY, round_result

try:
    import mediapipe as mp
except ImportError:
    mp = None


VIDEO_SIZE = (820, 560)
CAMERA_SIZE = (960, 540)
FRAME_INTERVAL_MS = 33
GESTURE_BUFFER_SIZE = 10
GESTURE_STABLE_COUNT = 6
COUNTDOWN_SECONDS = 3.0
ROUND_MIN_SAMPLES = 3
NEXT_ROUND_DELAY_MS = 1800
MODEL_PATH = Path(__file__).with_name("models") / "hand_landmarker.task"


@dataclass(frozen=True)
class GestureResult:
    move: str
    confidence: float
    finger_scores: dict[str, float]
    landmarks_seen: int


@dataclass(frozen=True)
class TimedGesture:
    timestamp: float
    gesture: GestureResult


class CameraStream:
    def __init__(self, camera_index: int = 0) -> None:
        self.camera_index = camera_index
        self.lock = threading.Lock()
        self.frame: np.ndarray | None = None
        self.last_frame_at = 0.0
        self.thread: threading.Thread | None = None
        self.capture = self._open_capture()
        self.running = self.capture is not None

        if self.running:
            self.thread = threading.Thread(target=self._read_loop, daemon=True)
            self.thread.start()

    @property
    def is_opened(self) -> bool:
        return self.capture is not None

    def read(self) -> np.ndarray | None:
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def close(self) -> None:
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=0.7)
        if self.capture is not None:
            self.capture.release()

    def _read_loop(self) -> None:
        while self.running and self.capture is not None:
            ok, frame = self.capture.read()
            if ok and frame is not None:
                with self.lock:
                    self.frame = frame
                    self.last_frame_at = time.monotonic()
            else:
                time.sleep(0.05)

    def _open_capture(self) -> object | None:
        backends = []
        if hasattr(cv2, "CAP_MSMF"):
            backends.append(cv2.CAP_MSMF)
        if hasattr(cv2, "CAP_DSHOW"):
            backends.append(cv2.CAP_DSHOW)
        backends.append(None)

        for backend in backends:
            if backend is None:
                capture = cv2.VideoCapture(self.camera_index)
            else:
                capture = cv2.VideoCapture(self.camera_index, backend)

            if not capture.isOpened():
                capture.release()
                continue

            self._configure_capture(capture)
            ok, frame = capture.read()
            if ok and frame is not None:
                self.frame = frame
                self.last_frame_at = time.monotonic()
                return capture

            capture.release()

        return None

    def _configure_capture(self, capture: object) -> None:
        width, height = CAMERA_SIZE
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_FPS, 30)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)


class HandGestureModel:
    def __init__(self) -> None:
        if mp is None:
            raise RuntimeError("mediapipe paketi kurulu degil")
        if not MODEL_PATH.exists():
            raise RuntimeError(f"model dosyasi bulunamadi: {MODEL_PATH}")

        vision = mp.tasks.vision
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.45,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._drawer = vision.drawing_utils
        self._styles = vision.drawing_styles
        self._connections = vision.HandLandmarksConnections
        self._last_timestamp_ms = 0
        self.last_hand_seen = False
        self.last_landmark_count = 0

    def close(self) -> None:
        self._landmarker.close()

    def detect(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, GestureResult | None]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(int(time.monotonic() * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        results = self._landmarker.detect_for_video(image, timestamp_ms)

        annotated = frame_bgr.copy()
        if not results.hand_landmarks:
            self.last_hand_seen = False
            self.last_landmark_count = 0
            return annotated, None

        hand_landmarks = results.hand_landmarks[0]
        self.last_hand_seen = True
        self.last_landmark_count = len(hand_landmarks)
        self._drawer.draw_landmarks(
            annotated,
            hand_landmarks,
            self._connections.HAND_CONNECTIONS,
            self._styles.get_default_hand_landmarks_style(),
            self._styles.get_default_hand_connections_style(),
        )

        return annotated, self._classify(hand_landmarks)

    def _classify(self, landmarks: list[object]) -> GestureResult | None:
        points = np.array([[mark.x, mark.y, mark.z] for mark in landmarks], dtype=float)
        finger_scores = {
            "index": self._finger_extension_score(points, 5, 6, 7, 8),
            "middle": self._finger_extension_score(points, 9, 10, 11, 12),
            "ring": self._finger_extension_score(points, 13, 14, 15, 16),
            "pinky": self._finger_extension_score(points, 17, 18, 19, 20),
        }

        rock_score = self._mean(1.0 - value for value in finger_scores.values())
        paper_score = self._mean(finger_scores.values())
        scissors_score = self._mean(
            [
                finger_scores["index"],
                finger_scores["middle"],
                1.0 - finger_scores["ring"],
                1.0 - finger_scores["pinky"],
            ]
        )

        candidates = {
            "tas": rock_score,
            "kagit": paper_score,
            "makas": scissors_score,
        }
        move, confidence = max(candidates.items(), key=lambda item: item[1])

        ordered_scores = sorted(candidates.values(), reverse=True)
        margin = ordered_scores[0] - ordered_scores[1]
        if confidence < 0.58 or margin < 0.08:
            return None

        return GestureResult(
            move=move,
            confidence=confidence,
            finger_scores=finger_scores,
            landmarks_seen=len(landmarks),
        )

    def _finger_extension_score(
        self,
        points: np.ndarray,
        mcp_index: int,
        pip_index: int,
        dip_index: int,
        tip_index: int,
    ) -> float:
        wrist = points[0]
        mcp = points[mcp_index]
        pip = points[pip_index]
        dip = points[dip_index]
        tip = points[tip_index]

        pip_angle = self._angle(mcp, pip, dip)
        dip_angle = self._angle(pip, dip, tip)
        angle_score = self._clamp((min(pip_angle, dip_angle) - 125.0) / 50.0)

        reach = self._distance(wrist, tip) / max(self._distance(wrist, pip), 1e-6)
        reach_score = self._clamp((reach - 1.05) / 0.35)

        return self._clamp((angle_score * 0.65) + (reach_score * 0.35))

    @staticmethod
    def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ba = a - b
        bc = c - b
        denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
        if denominator <= 1e-9:
            return 0.0

        cosine = float(np.dot(ba, bc) / denominator)
        cosine = max(-1.0, min(1.0, cosine))
        return math.degrees(math.acos(cosine))

    @staticmethod
    def _distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))

    @staticmethod
    def _mean(values: object) -> float:
        values = list(values)
        return sum(values) / len(values)


class RockPaperScissorsApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Tas Kagit Makas - Kamera Botu")
        self.root.geometry("1180x680")
        self.root.minsize(1060, 620)

        self.bot = AdaptiveRPSBot()
        self.score = Counter()
        self.round_number = 1

        self.camera = CameraStream()
        self.model = self._open_model()
        self.closed = False
        self.next_round_after_id: str | None = None

        self.frame_image: ImageTk.PhotoImage | None = None
        self.gesture_buffer: deque[GestureResult] = deque(maxlen=GESTURE_BUFFER_SIZE)
        self.no_gesture_frames = 0
        self.stable_move: str | None = None
        self.stable_confidence = 0.0

        self.round_active = False
        self.locked_bot_move: str | None = None
        self.locked_prediction = "?"
        self.locked_method = "random"
        self.locked_confidence = 0.0
        self.round_started_at = 0.0
        self.round_ends_at = 0.0
        self.round_gesture_samples: list[TimedGesture] = []

        self.round_var = tk.StringVar(value="Otomatik tur basliyor.")
        self.gesture_var = tk.StringVar(value="Algilanan: -")
        self.score_var = tk.StringVar(value="Sen: 0 | Bot: 0 | Berabere: 0")
        self.status_var = tk.StringVar(value="Kamera hazir.")
        self.last_round_var = tk.StringVar(value="Son tur: -")
        self.result_var = tk.StringVar(value="Sonuc bekleniyor.")

        self._build_ui()
        self._set_initial_status()

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(FRAME_INTERVAL_MS, self.update_frame)
        self.next_round_after_id = self.root.after(450, self.start_round)

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#101417")
        style.configure("Panel.TFrame", background="#161c20")
        style.configure("TLabel", background="#101417", foreground="#edf2f4")
        style.configure("Panel.TLabel", background="#161c20", foreground="#edf2f4")
        style.configure("Muted.TLabel", background="#161c20", foreground="#9da8ae")
        style.configure("Title.TLabel", background="#161c20", foreground="#ffffff", font=("Segoe UI", 18, "bold"))
        style.configure("Value.TLabel", background="#161c20", foreground="#ffffff", font=("Segoe UI", 13, "bold"))
        style.configure("TButton", font=("Segoe UI", 11), padding=(12, 8))
        style.configure("TCheckbutton", background="#161c20", foreground="#edf2f4", font=("Segoe UI", 10))

        self.root.configure(bg="#101417")
        main = ttk.Frame(self.root, padding=16)
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=0)
        main.rowconfigure(0, weight=1)

        video_frame = ttk.Frame(main)
        video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        video_frame.rowconfigure(0, weight=1)
        video_frame.columnconfigure(0, weight=1)

        self.video_label = ttk.Label(video_frame, anchor=tk.CENTER)
        self.video_label.grid(row=0, column=0, sticky="nsew")

        panel = ttk.Frame(main, style="Panel.TFrame", padding=18)
        panel.grid(row=0, column=1, sticky="ns")
        panel.columnconfigure(0, weight=1)

        ttk.Label(panel, text="Kamera Botu", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(panel, textvariable=self.status_var, style="Muted.TLabel", wraplength=280).grid(row=1, column=0, sticky="ew", pady=(4, 18))

        ttk.Label(panel, textvariable=self.round_var, style="Value.TLabel", wraplength=280).grid(row=2, column=0, sticky="ew", pady=6)
        ttk.Label(panel, textvariable=self.gesture_var, style="Panel.TLabel", wraplength=280).grid(row=3, column=0, sticky="ew", pady=(6, 16))
        ttk.Label(panel, textvariable=self.result_var, style="Value.TLabel", wraplength=280).grid(row=4, column=0, sticky="ew", pady=(0, 16))

        ttk.Separator(panel).grid(row=5, column=0, sticky="ew", pady=12)

        ttk.Label(panel, text="Skor", style="Muted.TLabel").grid(row=6, column=0, sticky="w")
        ttk.Label(panel, textvariable=self.score_var, style="Value.TLabel").grid(row=7, column=0, sticky="ew", pady=(4, 12))
        ttk.Label(panel, textvariable=self.last_round_var, style="Panel.TLabel", wraplength=280).grid(row=8, column=0, sticky="ew", pady=(0, 18))

        ttk.Button(panel, text="Skoru Sifirla", command=self.reset_game).grid(row=9, column=0, sticky="ew", pady=(16, 0))

    def _open_model(self) -> HandGestureModel | None:
        if mp is None:
            return None

        try:
            return HandGestureModel()
        except Exception as exc:
            messagebox.showwarning("Model baslatilamadi", str(exc))
            return None

    def _set_initial_status(self) -> None:
        if not self.camera.is_opened:
            self.status_var.set("Kamera bulunamadi. Webcam baglantisini kontrol et.")
        elif self.model is None:
            self.status_var.set("MediaPipe kurulu degil. requirements.txt ile bagimliliklari kur.")
        else:
            self.status_var.set("Tur otomatik baslayacak; geri sayim bitene kadar hamleni goster.")

    def start_round(self) -> None:
        if self.closed:
            return
        self.next_round_after_id = None
        self.locked_bot_move, predicted_move, method, confidence = self.bot.choose_bot_move()
        self.locked_prediction = predicted_move
        self.locked_method = method
        self.locked_confidence = confidence
        self.round_active = True
        self.round_started_at = time.monotonic()
        self.round_ends_at = self.round_started_at + COUNTDOWN_SECONDS
        self.round_gesture_samples.clear()
        self.gesture_buffer.clear()
        self.no_gesture_frames = 0
        self.stable_move = None
        self.stable_confidence = 0.0

        self.round_var.set("Geri sayim basladi.")
        self.gesture_var.set("Algilanan: -")
        self.result_var.set("Sonuc bekleniyor.")
        self.status_var.set("3 saniye boyunca hamlen analiz ediliyor.")

    def finish_round(self) -> None:
        if not self.round_active or self.locked_bot_move is None:
            return

        selected_move = self._select_countdown_move()
        if selected_move is None:
            self.round_active = False
            self.locked_bot_move = None
            self.round_var.set("Hamle okunamadi. Yeni tur basliyor.")
            self.result_var.set("Geri sayim icinde net hamle algilanamadi.")
            self.last_round_var.set("Son tur: hamle okunamadi")
            self.status_var.set("Elini daha aydinlik ve kameranin ortasinda goster.")
            self._schedule_next_round()
            return

        user_move, user_confidence = selected_move
        bot_move = self.locked_bot_move
        result = round_result(user_move, bot_move)
        self.score[result] += 1
        self.bot.observe_round(user_move, bot_move)
        self.round_active = False

        result_text = {
            "draw": "Berabere",
            "user": "Sen kazandin",
            "bot": "Bot kazandi",
        }[result]
        self.last_round_var.set(
            f"Son tur: Sen {DISPLAY[user_move]} | Bot {DISPLAY[bot_move]} | {result_text}"
        )
        self.score_var.set(
            f"Sen: {self.score['user']} | Bot: {self.score['bot']} | Berabere: {self.score['draw']}"
        )
        self.round_var.set("Tur bitti. Yeni tur basliyor.")
        self.result_var.set(
            f"Bot {DISPLAY[bot_move]} hamlesini yapti!\n"
            f"{result_text}. Senin hamlen: {DISPLAY[user_move]} (%{user_confidence * 100:.0f})."
        )
        self.status_var.set("Tur kesinlesti.")
        self.round_number += 1
        self._schedule_next_round()

    def _schedule_next_round(self) -> None:
        self._cancel_next_round()
        if not self.closed:
            self.next_round_after_id = self.root.after(
                NEXT_ROUND_DELAY_MS, self.start_round
            )

    def _cancel_next_round(self) -> None:
        if self.next_round_after_id is None:
            return
        try:
            self.root.after_cancel(self.next_round_after_id)
        except tk.TclError:
            pass
        self.next_round_after_id = None

    def _select_countdown_move(self) -> tuple[str, float] | None:
        if len(self.round_gesture_samples) < ROUND_MIN_SAMPLES:
            return None

        scores: Counter[str] = Counter()
        counts: Counter[str] = Counter()
        confidences: dict[str, list[float]] = {move: [] for move in DISPLAY}

        for sample in self.round_gesture_samples:
            progress = (sample.timestamp - self.round_started_at) / COUNTDOWN_SECONDS
            progress = max(0.0, min(1.0, progress))
            weight = 0.4 + (progress * 0.6)
            move = sample.gesture.move
            scores[move] += sample.gesture.confidence * weight
            counts[move] += 1
            confidences[move].append(sample.gesture.confidence)

        move, _ = scores.most_common(1)[0]
        if counts[move] < ROUND_MIN_SAMPLES:
            return None

        confidence = sum(confidences[move]) / len(confidences[move])
        return move, confidence

    def reset_game(self) -> None:
        self._cancel_next_round()
        self.bot = AdaptiveRPSBot()
        self.score.clear()
        self.round_number = 1
        self.last_round_var.set("Son tur: -")
        self.score_var.set("Sen: 0 | Bot: 0 | Berabere: 0")
        self.round_active = False
        self.locked_bot_move = None
        self.gesture_buffer.clear()
        self.round_gesture_samples.clear()
        self.stable_move = None
        self.stable_confidence = 0.0
        self.round_var.set("Oyun sifirlandi. Yeni tur basliyor.")
        self.gesture_var.set("Algilanan: -")
        self.result_var.set("Sonuc bekleniyor.")
        self.status_var.set("Oyun sifirlandi.")
        self.next_round_after_id = self.root.after(450, self.start_round)

    def update_frame(self) -> None:
        if self.closed:
            return

        try:
            self._update_frame_once()
        except Exception as exc:
            self.status_var.set(f"Canli goruntu dongusu hatasi: {exc}")
        finally:
            if not self.closed:
                self.root.after(FRAME_INTERVAL_MS, self.update_frame)

    def _update_frame_once(self) -> None:
        now = time.monotonic()
        frame = self._read_frame()
        gesture = None

        if frame is None:
            frame = self._placeholder_frame("Kamera goruntusu yok")
        else:
            frame = cv2.flip(frame, 1)
            if self.model is not None:
                try:
                    frame, gesture = self.model.detect(frame)
                    self._update_detection_status(gesture)
                except Exception as exc:
                    self._disable_model(exc)
            else:
                self._draw_overlay(frame, "MediaPipe modeli bulunamadi")

        self._record_countdown_gesture(gesture, now)
        self._update_stable_gesture(gesture)
        self._draw_gesture_overlay(frame)
        self._draw_countdown_overlay(frame, now)
        self._render_frame(frame)
        self._maybe_finish_round(now)

    def _read_frame(self) -> np.ndarray | None:
        return self.camera.read()

    def _disable_model(self, exc: Exception) -> None:
        model = self.model
        self.model = None
        if model is not None:
            try:
                model.close()
            except Exception:
                pass
        self.gesture_buffer.clear()
        self.stable_move = None
        self.stable_confidence = 0.0
        self.status_var.set(f"Model hatasi nedeniyle yalnizca kamera goruntusu acik: {exc}")

    def _record_countdown_gesture(
        self, gesture: GestureResult | None, timestamp: float
    ) -> None:
        if not self.round_active or gesture is None:
            return
        if timestamp > self.round_ends_at:
            return
        self.round_gesture_samples.append(TimedGesture(timestamp, gesture))

    def _update_detection_status(self, gesture: GestureResult | None) -> None:
        if self.model is None:
            return
        if not self.round_active:
            return
        if gesture is not None:
            self.status_var.set(
                f"El algilandi: {self.model.last_landmark_count} nokta. Hareket: {DISPLAY[gesture.move]}"
            )
        elif self.model.last_hand_seen:
            self.status_var.set(
                f"El algilandi: {self.model.last_landmark_count} nokta. Hamle net degil."
            )
        elif self.round_active:
            self.status_var.set("El araniyor. Avucunu kameraya net ve aydinlik goster.")

    def _update_stable_gesture(self, gesture: GestureResult | None) -> None:
        if gesture is None:
            self.no_gesture_frames += 1
            if self.no_gesture_frames >= 5:
                self.gesture_buffer.clear()
                self.stable_move = None
                self.stable_confidence = 0.0
                self.gesture_var.set("Algilanan: -")
            return

        self.no_gesture_frames = 0
        self.gesture_buffer.append(gesture)

        if not self.gesture_buffer:
            self.stable_move = None
            self.stable_confidence = 0.0
            self.gesture_var.set("Algilanan: -")
            return

        counts = Counter(item.move for item in self.gesture_buffer)
        move, count = counts.most_common(1)[0]
        matching = [item.confidence for item in self.gesture_buffer if item.move == move]
        confidence = sum(matching) / len(matching)

        if count >= GESTURE_STABLE_COUNT:
            self.stable_move = move
            self.stable_confidence = confidence
            self.gesture_var.set(f"Algilanan: {DISPLAY[move]} | Guven: %{confidence * 100:.0f}")
        else:
            latest = self.gesture_buffer[-1]
            self.stable_move = None
            self.stable_confidence = 0.0
            self.gesture_var.set(
                f"Algilanan: {DISPLAY[latest.move]} | Sabitleniyor..."
            )

    def _maybe_finish_round(self, now: float) -> None:
        if not self.round_active:
            return
        if now < self.round_ends_at:
            return
        self.finish_round()

    def _draw_countdown_overlay(self, frame: np.ndarray, now: float) -> None:
        if not self.round_active:
            return

        remaining = max(0.0, self.round_ends_at - now)
        number = max(1, math.ceil(remaining))
        self.round_var.set(f"Geri sayim: {number}")

        height, width = frame.shape[:2]
        center = (width // 2, height // 2)
        text = str(number)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 4.4
        thickness = 10
        text_size, _ = cv2.getTextSize(text, font, scale, thickness)
        x = center[0] - (text_size[0] // 2)
        y = center[1] + (text_size[1] // 2)

        overlay = frame.copy()
        cv2.circle(overlay, center, 112, (16, 20, 23), -1)
        cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)
        cv2.putText(
            frame,
            text,
            (x, y),
            font,
            scale,
            (10, 14, 16),
            thickness + 8,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (x, y),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        label = "Hamleni goster"
        label_scale = 0.78
        label_thickness = 2
        label_size, _ = cv2.getTextSize(label, font, label_scale, label_thickness)
        label_x = center[0] - (label_size[0] // 2)
        label_y = center[1] + 150
        cv2.putText(
            frame,
            label,
            (label_x, label_y),
            font,
            label_scale,
            (255, 255, 255),
            label_thickness,
            cv2.LINE_AA,
        )

    def _draw_gesture_overlay(self, frame: np.ndarray) -> None:
        text = "Algilanan: -"
        if self.stable_move is not None:
            text = f"Algilanan: {DISPLAY[self.stable_move]} (%{self.stable_confidence * 100:.0f})"

        self._draw_overlay(frame, text)

    def _draw_overlay(self, frame: np.ndarray, text: str) -> None:
        cv2.rectangle(frame, (18, 18), (520, 70), (16, 20, 23), -1)
        cv2.putText(
            frame,
            text,
            (34, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (237, 242, 244),
            2,
            cv2.LINE_AA,
        )

    def _render_frame(self, frame_bgr: np.ndarray) -> None:
        width, height = VIDEO_SIZE
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail(VIDEO_SIZE, Image.Resampling.LANCZOS)

        canvas = Image.new("RGB", VIDEO_SIZE, (10, 14, 16))
        x = (width - image.width) // 2
        y = (height - image.height) // 2
        canvas.paste(image, (x, y))

        self.frame_image = ImageTk.PhotoImage(canvas)
        self.video_label.configure(image=self.frame_image)

    def _placeholder_frame(self, message: str) -> np.ndarray:
        width, height = VIDEO_SIZE
        frame = np.full((height, width, 3), (18, 22, 25), dtype=np.uint8)
        cv2.putText(
            frame,
            message,
            (34, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (237, 242, 244),
            2,
            cv2.LINE_AA,
        )
        return frame

    def close(self) -> None:
        self.closed = True
        self._cancel_next_round()
        self.camera.close()
        if self.model is not None:
            self.model.close()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    if REQUIRED_IMPORT_ERROR is not None:
        root.withdraw()
        messagebox.showerror(
            "Eksik bagimlilik",
            f"{REQUIRED_IMPORT_ERROR}\n\nrequirements.txt dosyasindaki paketleri kur.",
        )
        root.destroy()
        return

    try:
        RockPaperScissorsApp(root)
    except Exception as exc:
        messagebox.showerror("Uygulama baslatilamadi", str(exc))
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
